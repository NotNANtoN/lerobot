"""Control-Aware JEPA Latent World Model and MuZero/MPPI Latent Planner.

Combines:
1. V-JEPA 2 / H-JEPA variable-horizon action-conditioned latent prediction:
   ``P_phi(h_t, a_{t:t+H}, H) -> (h_hat_{t+H}, s_hat_{t+H})`` against an EMA target encoder.
2. Inverse-dynamics regularization from Toso, LeCun et al. (Oct 2026, arXiv:2610.07540,
   "Preserving Unstable Modes Through Inverse Dynamics in JEPA World Models"):
   ``I_psi(h_t, h_{t+H}, H) -> a_hat_{t:t+H}`` makes the latent encoder injective on the
   finite-horizon reachable subspace so controllable modes cannot collapse.
3. Variance-covariance anti-collapse regularization (VICReg / LeVJEPA arXiv:2608.27395).
4. MuZero value/progress head ``V_omega(h_t) -> [0, 1]`` + goal-state prototype scoring.
5. Cycle-consistent MuZero/MPPI latent search over candidate action chunks from a policy prior.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


@dataclass(frozen=True)
class JEPAWorldModelConfig:
    """Configuration for ControlAwareJEPAWorldModel."""

    vis_tokens: int = 600
    vis_dim: int = 2048
    state_dim: int = 6
    action_dim: int = 6
    max_horizon: int = 30
    latent_dim: int = 512
    num_spatial_queries: int = 8
    hidden_dim: int = 512
    predictor_layers: int = 3
    dropout: float = 0.1
    ema_decay: float = 0.99


class SpatialTemporalPooler(nn.Module):
    """Compress [B, 600, 2048] (2 frames x 15x20 patches) + [B, 6] state into [B, latent_dim]."""

    def __init__(self, cfg: JEPAWorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        half = cfg.vis_tokens // 2
        self.half = half
        # Project per-token visual features down before attention pooling
        self.token_proj = nn.Sequential(
            nn.LayerNorm(cfg.vis_dim),
            nn.Linear(cfg.vis_dim, cfg.latent_dim // 2),
            nn.GELU(),
        )
        self.queries = nn.Parameter(torch.randn(1, cfg.num_spatial_queries, cfg.latent_dim // 2) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.latent_dim // 2, num_heads=4, dropout=cfg.dropout, batch_first=True
        )
        # Concatenate: [attn_pooled_t1, attn_pooled_t0_diff, mean_t1, state_emb]
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.latent_dim // 4),
            nn.GELU(),
            nn.Linear(cfg.latent_dim // 4, cfg.latent_dim // 4),
        )
        fused_in = (cfg.latent_dim // 2) * 3 + (cfg.latent_dim // 4)
        self.out_proj = nn.Sequential(
            nn.Linear(fused_in, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
        )

    def forward(self, vis_context: Tensor, state: Tensor) -> Tensor:
        """Encode ``vis_context [B, 600, 2048]`` and ``state [B, 6]`` -> ``h [B, latent_dim]``."""
        if vis_context.ndim != 3:
            raise ValueError(f"Expected vis_context [B, N, D], got {tuple(vis_context.shape)}")
        if state.ndim == 3:
            state = state.squeeze(1)
        b = vis_context.shape[0]
        tokens = self.token_proj(vis_context.float())
        t0, t1 = tokens[:, : self.half], tokens[:, self.half :]
        q = self.queries.expand(b, -1, -1)
        pooled_t1, _ = self.attn(q, t1, t1, need_weights=False)
        pooled_t0, _ = self.attn(q, t0, t0, need_weights=False)
        attn_t1 = pooled_t1.mean(dim=1)
        attn_diff = (pooled_t1 - pooled_t0).mean(dim=1)  # explicit velocity/motion cue
        mean_t1 = t1.mean(dim=1)
        s_emb = self.state_proj(state.float())
        fused = torch.cat([attn_t1, attn_diff, mean_t1, s_emb], dim=-1)
        return self.out_proj(fused)


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block used in the forward and inverse dynamics predictors."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class ActionSequenceEncoder(nn.Module):
    """Encode a variable-horizon action prefix ``a_{0:H}`` into a conditioning vector."""

    def __init__(self, cfg: JEPAWorldModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.step_proj = nn.Linear(cfg.action_dim, cfg.hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, cfg.max_horizon, cfg.hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=4,
            dim_feedforward=cfg.hidden_dim * 2,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2, enable_nested_tensor=False)
        self.horizon_emb = nn.Embedding(cfg.max_horizon + 1, cfg.hidden_dim)
        self.out_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, actions: Tensor, horizons: Tensor) -> Tensor:
        """Encode ``actions [B, max_H, 6]`` up to per-sample ``horizons [B]`` -> ``[B, hidden_dim]``."""
        b, max_h, _ = actions.shape
        device = actions.device
        steps = torch.arange(max_h, device=device).unsqueeze(0)  # [1, max_H]
        valid = steps < horizons.unsqueeze(1)  # [B, max_H]
        x = self.step_proj(actions.float()) + self.pos_emb[:, :max_h]
        x = self.transformer(x, src_key_padding_mask=~valid)
        weights = valid.float().unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.out_norm(pooled + self.horizon_emb(horizons.clamp(1, self.cfg.max_horizon)))


class ControlAwareJEPAWorldModel(nn.Module):
    """Latent World Model with Forward Dynamics, Inverse Dynamics (IDM), and MuZero Value Head."""

    def __init__(self, cfg: JEPAWorldModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or JEPAWorldModelConfig()
        c = self.cfg

        # Online encoder and EMA target encoder
        self.encoder = SpatialTemporalPooler(c)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # Action sequence encoder
        self.action_encoder = ActionSequenceEncoder(c)

        # Forward dynamics predictor P_phi(h_t, a_{t:t+H}, s_t, H) -> (delta_h, delta_s)
        self.forward_in = nn.Linear(c.latent_dim + c.hidden_dim + c.state_dim, c.hidden_dim)
        self.forward_blocks = nn.ModuleList(
            [ResidualBlock(c.hidden_dim, c.dropout) for _ in range(c.predictor_layers)]
        )
        self.forward_latent_head = nn.Sequential(
            nn.LayerNorm(c.hidden_dim),
            nn.Linear(c.hidden_dim, c.latent_dim),
        )
        self.forward_state_head = nn.Sequential(
            nn.LayerNorm(c.hidden_dim),
            nn.Linear(c.hidden_dim, c.state_dim),
        )

        # Inverse dynamics predictor I_psi(h_t, h_{t+H}, H) -> a_hat_{0:max_H}
        # Preserves controllable unstable modes (Toso & LeCun, Oct 2026, arXiv:2610.07540)
        self.horizon_emb_idm = nn.Embedding(c.max_horizon + 1, c.hidden_dim)
        self.inverse_in = nn.Linear(c.latent_dim * 3 + c.hidden_dim, c.hidden_dim)
        self.inverse_blocks = nn.ModuleList([ResidualBlock(c.hidden_dim, c.dropout) for _ in range(2)])
        self.inverse_action_head = nn.Sequential(
            nn.LayerNorm(c.hidden_dim),
            nn.Linear(c.hidden_dim, c.max_horizon * c.action_dim),
        )

        # MuZero Value / Progress head V_omega(h_t) -> (progress in [0, 1], goal_logit)
        self.value_head = nn.Sequential(
            nn.LayerNorm(c.latent_dim),
            nn.Linear(c.latent_dim, c.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(c.hidden_dim // 2, 2),
        )

        # Running goal-state prototype (updated during training from terminal demonstration frames)
        self.register_buffer("goal_prototype", torch.zeros(c.latent_dim))
        self.register_buffer("goal_prototype_initialized", torch.tensor(False))

    @torch.no_grad()
    def update_target_encoder(self, decay: float | None = None) -> None:
        """EMA update of the target encoder."""
        tau = self.cfg.ema_decay if decay is None else decay
        for p_online, p_target in zip(
            self.encoder.parameters(), self.target_encoder.parameters(), strict=True
        ):
            p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)

    @torch.no_grad()
    def update_goal_prototype(self, h_goal: Tensor, momentum: float = 0.95) -> None:
        """Update the latent goal prototype with terminal demonstration latents."""
        if h_goal.numel() == 0:
            return
        mean_goal = F.normalize(h_goal.detach().float().mean(dim=0), dim=-1)
        if not bool(self.goal_prototype_initialized.item()):
            self.goal_prototype.copy_(mean_goal)
            self.goal_prototype_initialized.fill_(True)
        else:
            updated = F.normalize(self.goal_prototype * momentum + mean_goal * (1.0 - momentum), dim=-1)
            self.goal_prototype.copy_(updated)

    def encode(self, vis_context: Tensor, state: Tensor, *, use_target: bool = False) -> Tensor:
        """Encode observation into latent ``h_t [B, latent_dim]``."""
        enc = self.target_encoder if use_target else self.encoder
        return enc(vis_context, state)

    def predict_forward(
        self, h_t: Tensor, state_t: Tensor, actions: Tensor, horizons: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Predict future latent ``h_hat_{t+H}`` and normalized state ``s_hat_{t+H}``."""
        if state_t.ndim == 3:
            state_t = state_t.squeeze(1)
        a_emb = self.action_encoder(actions, horizons)
        x = self.forward_in(torch.cat([h_t, a_emb, state_t.float()], dim=-1))
        for block in self.forward_blocks:
            x = block(x)
        h_next = F.layer_norm(h_t + self.forward_latent_head(x), (self.cfg.latent_dim,))
        s_next = state_t.float() + self.forward_state_head(x)
        return h_next, s_next

    def predict_inverse(self, h_t: Tensor, h_next: Tensor, horizons: Tensor) -> Tensor:
        """Predict the action sequence ``[B, max_H, action_dim]`` that caused ``h_t -> h_next``."""
        h_emb = self.horizon_emb_idm(horizons.clamp(1, self.cfg.max_horizon))
        x = self.inverse_in(torch.cat([h_t, h_next, h_next - h_t, h_emb], dim=-1))
        for block in self.inverse_blocks:
            x = block(x)
        return self.inverse_action_head(x).view(-1, self.cfg.max_horizon, self.cfg.action_dim)

    def predict_value(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(progress in [0, 1], goal_logit)`` each shaped ``[B]``."""
        out = self.value_head(h)
        return torch.sigmoid(out[:, 0]), out[:, 1]

    @staticmethod
    def variance_covariance_loss(h: Tensor, gamma: float = 1.0) -> Tensor:
        """VICReg / LeVJEPA anti-collapse loss ensuring unit variance and decorrelated features."""
        if h.shape[0] < 4:
            return h.new_zeros(())
        h_centered = h - h.mean(dim=0, keepdim=True)
        std = torch.sqrt(h_centered.var(dim=0) + 1e-4)
        var_loss = F.relu(gamma - std).mean()
        cov = (h_centered.T @ h_centered) / (h.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = off_diag.pow(2).sum() / h.shape[1]
        return var_loss + 0.04 * cov_loss

    @torch.no_grad()
    def score_action_candidates(
        self,
        h_t: Tensor,
        state_t: Tensor,
        candidate_actions: Tensor,
        *,
        horizons: tuple[int, ...] = (10, 20, 30),
        w_value: float = 1.0,
        w_goal: float = 0.5,
        w_cycle: float = 0.25,
        w_prior: float = 0.15,
    ) -> dict[str, Tensor]:
        """Score ``N`` candidate action chunks ``[N, max_H, 6]`` for a single observation ``h_t [1, D]``.

        Combines:
        - Predicted MuZero value/progress ``V(h_hat_{t+H})`` across lookahead horizons,
        - Cosine similarity to the terminal goal prototype ``cos(h_hat_{t+H}, h_goal)``,
        - Forward-inverse cycle consistency ``-||I_psi(h_t, h_hat_{t+H}) - a||^2`` (penalizes OOD hallucinations),
        - Proximity to the policy prior mean ``-||a - mean(a)||^2``.
        """
        n_cand = candidate_actions.shape[0]
        device = candidate_actions.device
        h_rep = h_t.expand(n_cand, -1)
        s_rep = state_t.reshape(1, -1).expand(n_cand, -1)

        total_value = torch.zeros(n_cand, device=device)
        total_goal = torch.zeros(n_cand, device=device)
        total_cycle_err = torch.zeros(n_cand, device=device)

        for h_step in horizons:
            h_tensor = torch.full((n_cand,), h_step, dtype=torch.long, device=device)
            h_pred, _ = self.predict_forward(h_rep, s_rep, candidate_actions, h_tensor)
            val_pred, goal_logit = self.predict_value(h_pred)
            total_value = total_value + val_pred + 0.25 * torch.sigmoid(goal_logit)
            if bool(self.goal_prototype_initialized.item()):
                goal_cos = F.cosine_similarity(
                    F.normalize(h_pred, dim=-1), self.goal_prototype.unsqueeze(0), dim=-1
                )
                total_goal = total_goal + goal_cos
            a_recon = self.predict_inverse(h_rep, h_pred, h_tensor)
            cycle_mse = (a_recon[:, :h_step] - candidate_actions[:, :h_step]).square().mean(dim=(1, 2))
            total_cycle_err = total_cycle_err + cycle_mse

        k = float(len(horizons))
        total_value = total_value / k
        total_goal = total_goal / k
        total_cycle_err = total_cycle_err / k

        prior_mean = candidate_actions.mean(dim=0, keepdim=True)
        prior_dev = (candidate_actions - prior_mean).square().mean(dim=(1, 2))

        score = w_value * total_value + w_goal * total_goal - w_cycle * total_cycle_err - w_prior * prior_dev
        return {
            "score": score,
            "value": total_value,
            "goal_cos": total_goal,
            "cycle_mse": total_cycle_err,
            "prior_dev": prior_dev,
        }

    @torch.no_grad()
    def plan_mppi(
        self,
        h_t: Tensor,
        state_t: Tensor,
        prior_actions: Tensor,
        *,
        num_iters: int = 2,
        num_perturbations: int = 32,
        noise_std: float = 0.08,
        temperature: float = 0.1,
        horizons: tuple[int, ...] = (10, 20, 30),
        w_value: float = 1.0,
        w_goal: float = 0.5,
        w_cycle: float = 0.25,
        w_prior: float = 0.20,
    ) -> tuple[Tensor, dict[str, float]]:
        """MuZero / TD-MPC2 style latent MPPI search anchored to ``prior_actions [N, H, 6]``."""
        device = prior_actions.device
        anchor = prior_actions.mean(dim=0, keepdim=True)  # [1, H, 6]
        mean_seq = prior_actions[0:1].clone()

        best_chunk = prior_actions[0].clone()
        best_score = -float("inf")
        initial_scores = self.score_action_candidates(
            h_t,
            state_t,
            prior_actions,
            horizons=horizons,
            w_value=w_value,
            w_goal=w_goal,
            w_cycle=w_cycle,
            w_prior=w_prior,
        )
        best_idx = int(initial_scores["score"].argmax().item())
        best_chunk = prior_actions[best_idx].clone()
        best_score = float(initial_scores["score"][best_idx].item())
        mean_seq = best_chunk.unsqueeze(0)

        for _ in range(num_iters):
            # Generate smooth temporally-correlated perturbations around current elite mean
            raw_noise = torch.randn(num_perturbations, *mean_seq.shape[1:], device=device) * noise_std
            smooth_noise = F.avg_pool1d(
                raw_noise.transpose(1, 2), kernel_size=5, stride=1, padding=2
            ).transpose(1, 2)
            candidates = torch.cat([prior_actions, mean_seq + smooth_noise], dim=0)
            scored = self.score_action_candidates(
                h_t,
                state_t,
                candidates,
                horizons=horizons,
                w_value=w_value,
                w_goal=w_goal,
                w_cycle=w_cycle,
                w_prior=w_prior,
            )
            # Add explicit anchor penalty to the original policy prior mean
            anchor_pen = (candidates - anchor).square().mean(dim=(1, 2))
            total_s = scored["score"] - w_prior * anchor_pen
            weights = torch.softmax(total_s / max(temperature, 1e-4), dim=0)
            mean_seq = (weights[:, None, None] * candidates).sum(dim=0, keepdim=True)
            top_idx = int(total_s.argmax().item())
            if float(total_s[top_idx].item()) > best_score:
                best_score = float(total_s[top_idx].item())
                best_chunk = candidates[top_idx].clone()

        # Score the MPPI-blended mean_seq as well and take the higher of (mean_seq, best_chunk)
        final_cands = torch.cat([mean_seq, best_chunk.unsqueeze(0)], dim=0)
        final_scored = self.score_action_candidates(
            h_t,
            state_t,
            final_cands,
            horizons=horizons,
            w_value=w_value,
            w_goal=w_goal,
            w_cycle=w_cycle,
            w_prior=w_prior,
        )
        pick = int(final_scored["score"].argmax().item())
        chosen = final_cands[pick]
        return chosen, {
            "initial_prior_score": float(initial_scores["score"][0].item()),
            "best_of_n_score": float(initial_scores["score"][best_idx].item()),
            "final_planned_score": float(final_scored["score"][pick].item()),
            "final_cycle_mse": float(final_scored["cycle_mse"][pick].item()),
            "final_value": float(final_scored["value"][pick].item()),
        }
