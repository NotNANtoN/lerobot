#!/usr/bin/env python3
"""Train and evaluate a Control-Aware JEPA Latent World Model + MuZero/MPPI Latent Planner.

Trains on multi-horizon latent transitions (H in {3, 6, ..., 30}) from cached Cosmos 3 Edge
features, ablates LeCun's Oct 2026 Inverse-Dynamics regularization (arXiv:2610.07540), and
evaluates MuZero/MPPI latent lookahead search on top of a trained SmolExpert policy prior.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file, save_file

from lerobot.policies.vam.jepa_world_model import (
    ControlAwareJEPAWorldModel,
    JEPAWorldModelConfig,
)
from lerobot.policies.vam.smol_expert import (
    ACTION_DIM,
    SMOLVLA_CHECKPOINT,
    SmolExpertActionDecoder,
    SmolVLANormalizer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("outputs/features/cosmos3-edge-lora/train/manifest.json"),
    )
    p.add_argument(
        "--val-manifest",
        type=Path,
        default=Path("outputs/features/cosmos3-edge-lora/val/manifest.json"),
    )
    p.add_argument(
        "--policy-run-dir",
        type=Path,
        default=Path("outputs/train/v1-cosmos3-edge-lora-online-unaugmented-smolexpert"),
        help="Trained SmolExpert run dir used as the MuZero policy prior during evaluation.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--idm-weight",
        type=float,
        default=1.0,
        help="Weight on inverse-dynamics action reconstruction loss (arXiv:2610.07540). Set 0 to ablate.",
    )
    p.add_argument("--state-weight", type=float, default=1.0)
    p.add_argument("--value-weight", type=float, default=0.5)
    p.add_argument("--vicreg-weight", type=float, default=0.1)
    p.add_argument("--num-candidates", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_split_in_memory(manifest_path: Path) -> dict[str, Any]:
    """Load all cached safetensors entries of a manifest into CPU memory for fast multi-horizon indexing."""
    payload = json.loads(manifest_path.read_text())
    base_dir = manifest_path.parent
    entries = payload["entries"]

    contexts: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    paddings: list[torch.Tensor] = []
    episodes: list[int] = []
    frames: list[int] = []
    sample_ids: list[str] = []

    by_ep_frame: dict[tuple[int, int], int] = {}
    ep_max_frame: dict[int, int] = {}

    for idx, entry in enumerate(entries):
        ep = int(entry["episode_index"])
        fr = int(entry["frame_index"])
        sd = load_file(str(base_dir / entry["safetensors"]), device="cpu")
        contexts.append(sd["context"].to(torch.bfloat16))
        states.append(sd["state"].reshape(ACTION_DIM).float())
        actions.append(sd["target_action"].reshape(-1, ACTION_DIM).float())
        paddings.append(sd["action_is_pad"].reshape(-1).bool())
        episodes.append(ep)
        frames.append(fr)
        sample_ids.append(entry["sample_id"])
        by_ep_frame[(ep, fr)] = idx
        ep_max_frame[ep] = max(ep_max_frame.get(ep, fr), fr)

    progress = torch.tensor(
        [float(fr) / max(float(ep_max_frame[ep]), 1.0) for ep, fr in zip(episodes, frames, strict=True)],
        dtype=torch.float32,
    )
    is_goal = torch.tensor(
        [1.0 if (ep_max_frame[ep] - fr) <= 30 else 0.0 for ep, fr in zip(episodes, frames, strict=True)],
        dtype=torch.float32,
    )

    return {
        "contexts": torch.stack(contexts, dim=0),  # [N, 600, 2048] bf16
        "states": torch.stack(states, dim=0),  # [N, 6] fp32
        "actions": torch.stack(actions, dim=0),  # [N, 30, 6] fp32
        "paddings": torch.stack(paddings, dim=0),  # [N, 30] bool
        "episodes": episodes,
        "frames": frames,
        "sample_ids": sample_ids,
        "by_ep_frame": by_ep_frame,
        "ep_max_frame": ep_max_frame,
        "progress": progress,
        "is_goal": is_goal,
    }


def build_transition_pairs(
    split_data: dict[str, Any], horizons: tuple[int, ...]
) -> list[tuple[int, int, int]]:
    """Return ``(idx_t, idx_next, H)`` for every valid in-episode transition in ``horizons``."""
    pairs: list[tuple[int, int, int]] = []
    by_ep_frame = split_data["by_ep_frame"]
    for idx_t, (ep, fr) in enumerate(zip(split_data["episodes"], split_data["frames"], strict=True)):
        for h in horizons:
            idx_next = by_ep_frame.get((ep, fr + h))
            if idx_next is not None and not bool(split_data["paddings"][idx_t, :h].any().item()):
                pairs.append((idx_t, idx_next, h))
    return pairs


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation without scipy."""
    if len(x) < 2:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 1e-12 else 0.0


@torch.no_grad()
def evaluate_world_model(
    model: ControlAwareJEPAWorldModel,
    val_data: dict[str, Any],
    val_pairs: list[tuple[int, int, int]],
    normalizer: SmolVLANormalizer,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate forward latent prediction, state prediction, inverse dynamics, and value correlation."""
    model.eval()
    contexts = val_data["contexts"].to(device)
    raw_states = val_data["states"].to(device)
    norm_states = normalizer.normalize_state(raw_states)
    raw_actions = val_data["actions"].to(device)
    norm_actions = normalizer.normalize_action(raw_actions)

    # Encode all validation anchors with both online and target encoders
    h_online = model.encode(contexts, norm_states, use_target=False)
    h_target = model.encode(contexts, norm_states, use_target=True)
    val_preds, _ = model.predict_value(h_online)

    # Per-episode Spearman rank correlation between predicted value and true frame progress
    ep_corrs = []
    for ep in sorted(set(val_data["episodes"])):
        mask = [i for i, e in enumerate(val_data["episodes"]) if e == ep]
        if len(mask) >= 3:
            c = spearman_corr(
                val_preds[mask].cpu().numpy(),
                val_data["progress"][mask].cpu().numpy(),
            )
            ep_corrs.append(c)

    idx_t = torch.tensor([p[0] for p in val_pairs], dtype=torch.long, device=device)
    idx_next = torch.tensor([p[1] for p in val_pairs], dtype=torch.long, device=device)
    horizons = torch.tensor([p[2] for p in val_pairs], dtype=torch.long, device=device)

    h_t = h_online[idx_t]
    s_t_norm = norm_states[idx_t]
    s_t_raw = raw_states[idx_t]
    a_t_norm = norm_actions[idx_t]
    a_t_raw = raw_actions[idx_t]

    h_next_true = h_target[idx_next]
    s_next_true_raw = raw_states[idx_next]

    # 1. Forward prediction with true actions vs zero actions vs identity
    h_pred, s_pred_norm = model.predict_forward(h_t, s_t_norm, a_t_norm, horizons)
    h_pred_no_act, s_pred_no_act_norm = model.predict_forward(
        h_t, s_t_norm, torch.zeros_like(a_t_norm), horizons
    )

    cos_true = F.cosine_similarity(h_pred, h_next_true, dim=-1).mean().item()
    cos_no_act = F.cosine_similarity(h_pred_no_act, h_next_true, dim=-1).mean().item()
    cos_identity = F.cosine_similarity(h_t, h_next_true, dim=-1).mean().item()

    # Retrieval accuracy among the 80 validation future targets
    sim_matrix = F.normalize(h_pred, dim=-1) @ F.normalize(h_next_true, dim=-1).T
    ranks = (
        sim_matrix.argsort(dim=-1, descending=True) == torch.arange(len(val_pairs), device=device)[:, None]
    ).nonzero()[:, 1]
    top1_acc = float((ranks == 0).float().mean().item())
    top5_acc = float((ranks < 5).float().mean().item())

    sim_id = F.normalize(h_t, dim=-1) @ F.normalize(h_next_true, dim=-1).T
    ranks_id = (
        sim_id.argsort(dim=-1, descending=True) == torch.arange(len(val_pairs), device=device)[:, None]
    ).nonzero()[:, 1]
    top1_id = float((ranks_id == 0).float().mean().item())

    # 2. Future proprioceptive state RMSE in physical degrees (at H=20 = 2.0s ahead)
    s_std = normalizer.state_std.to(s_pred_norm)
    s_mean = normalizer.state_mean.to(s_pred_norm)
    s_pred_raw = s_pred_norm * s_std + s_mean
    s_pred_no_act_raw = s_pred_no_act_norm * s_std + s_mean
    state_rmse_deg = math.sqrt(float((s_pred_raw - s_next_true_raw).square().mean().item()))
    state_no_act_rmse_deg = math.sqrt(float((s_pred_no_act_raw - s_next_true_raw).square().mean().item()))
    state_repeat_rmse_deg = math.sqrt(float((s_t_raw - s_next_true_raw).square().mean().item()))

    # 3. Inverse dynamics action reconstruction RMSE in physical degrees on held-out transitions
    a_idm_norm = model.predict_inverse(h_t, h_online[idx_next], horizons)
    a_idm_raw = normalizer.denormalize_action(a_idm_norm)
    h_val = int(horizons[0].item())
    idm_rmse_deg = math.sqrt(float((a_idm_raw[:, :h_val] - a_t_raw[:, :h_val]).square().mean().item()))

    return {
        "num_val_transitions": len(val_pairs),
        "horizon_steps": h_val,
        "latent_cos_sim": {
            "jepa_action_conditioned": cos_true,
            "jepa_zero_action": cos_no_act,
            "identity_baseline": cos_identity,
        },
        "latent_retrieval": {
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "identity_top1_acc": top1_id,
        },
        "future_state_rmse_deg": {
            "jepa_action_conditioned": state_rmse_deg,
            "jepa_zero_action": state_no_act_rmse_deg,
            "state_repeat_baseline": state_repeat_rmse_deg,
        },
        "inverse_dynamics_action_rmse_deg": idm_rmse_deg,
        "value_progress_spearman_rho": float(np.mean(ep_corrs)) if ep_corrs else 0.0,
    }


@torch.no_grad()
def evaluate_muzero_planning_with_smolexpert(
    model: ControlAwareJEPAWorldModel,
    val_data: dict[str, Any],
    policy_run_dir: Path,
    device: torch.device,
    *,
    num_candidates: int = 16,
) -> dict[str, Any]:
    """Evaluate MuZero/MPPI latent search on top of a trained SmolExpert policy prior across all 88 val anchors."""
    ckpt_path = policy_run_dir / "best.safetensors"
    norm_path = policy_run_dir / "normalizer.safetensors"
    if not (ckpt_path.is_file() and norm_path.is_file()):
        return {"skipped": f"Policy checkpoint not found at {policy_run_dir}"}

    norm_sd = load_file(str(norm_path), device="cpu")
    normalizer = SmolVLANormalizer(
        state_mean=norm_sd["state_mean"],
        state_std=norm_sd["state_std"],
        action_mean=norm_sd["action_mean"],
        action_std=norm_sd["action_std"],
        eps=1e-8,
        source_split="train",
    )
    decoder = SmolExpertActionDecoder.from_training_checkpoint(
        ckpt_path,
        normalizer=normalizer,
        expert_checkpoint=SMOLVLA_CHECKPOINT,
        device=device,
        num_steps=10,
        input_channels=2048,
    )
    decoder.eval()
    model.eval()

    contexts = val_data["contexts"].to(device)
    raw_states = val_data["states"].to(device)
    norm_states = normalizer.normalize_state(raw_states)
    target_actions = val_data["actions"].to(device)
    paddings = val_data["paddings"].to(device)

    n_anchors = len(contexts)
    single_preds = []
    prior_mean_preds = []
    best_of_n_preds = []
    mppi_preds = []
    latencies_ms = []

    for i in range(n_anchors):
        ctx_i = contexts[i : i + 1]  # [1, 600, 2048]
        state_i = raw_states[i : i + 1]  # [1, 6]
        s_norm_i = norm_states[i : i + 1]  # [1, 6]

        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()

        # 1. Sample N candidate action chunks from the SmolExpert flow-matching prior in one batched pass
        # Candidate 0 uses the canonical deterministic seed; candidates 1..N-1 explore nearby modes
        noises = []
        base_noise: torch.Tensor | None = None
        for k in range(num_candidates):
            gen = torch.Generator(device=device).manual_seed(42 + i * 1000 + k)
            n_k = torch.randn((1, 30, 32), generator=gen, device=device, dtype=torch.float32)
            if base_noise is not None:
                n_k = 0.85 * base_noise + 0.35 * n_k
            else:
                base_noise = n_k.clone()
            noises.append(n_k[0])
        noise_batch = torch.stack(noises, dim=0)  # [N, 30, 32]
        cands_raw_t = decoder.sample_actions(
            state_i.expand(num_candidates, -1),
            ctx_i.float().expand(num_candidates, -1, -1),
            noise=noise_batch,
            num_steps=10,
            use_cuda_graph=False,
        )  # [N, 30, 6] in physical degrees
        cands_norm_t = normalizer.normalize_action(cands_raw_t)  # [N, 30, 6] normalized

        # 2. Encode latent state h_t once and score / refine candidates in the JEPA world model
        h_t = model.encode(ctx_i, s_norm_i, use_target=False)
        scored = model.score_action_candidates(
            h_t,
            s_norm_i,
            cands_norm_t,
            horizons=(10, 20, 30),
            w_value=1.0,
            w_goal=0.5,
            w_cycle=0.3,
            w_prior=0.25,
        )
        best_idx = int(scored["score"].argmax().item())
        bon_raw = cands_raw_t[best_idx]

        mppi_norm, _ = model.plan_mppi(
            h_t,
            s_norm_i,
            cands_norm_t,
            num_iters=2,
            num_perturbations=24,
            noise_std=0.05,
            temperature=0.08,
            horizons=(10, 20, 30),
            w_value=1.0,
            w_goal=0.5,
            w_cycle=0.3,
            w_prior=0.35,
        )
        mppi_raw = normalizer.denormalize_action(mppi_norm)

        torch.cuda.synchronize() if device.type == "cuda" else None
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        single_preds.append(cands_raw_t[0])
        prior_mean_preds.append(cands_raw_t.mean(dim=0))
        best_of_n_preds.append(bon_raw)
        mppi_preds.append(mppi_raw)

    valid = (~paddings).unsqueeze(-1).double()
    tgt = target_actions.double()

    def calc_metrics(pred_list: list[torch.Tensor]) -> dict[str, float]:
        p = torch.stack(pred_list, dim=0).double()
        sq = (p - tgt).square() * valid
        n_tok = valid.sum()
        full30 = math.sqrt(float(sq.sum() / (n_tok * ACTION_DIM)))
        h1 = math.sqrt(float(sq[:, 0].sum() / (valid[:, 0].sum() * ACTION_DIM)))
        f5 = sum(math.sqrt(float(sq[:, h].sum() / (valid[:, h].sum() * ACTION_DIM))) for h in range(5)) / 5.0
        return {"full30": full30, "h1": h1, "f5": f5}

    return {
        "num_anchors": n_anchors,
        "num_candidates": num_candidates,
        "median_planning_ms": float(np.median(latencies_ms[1:])),
        "single_sample_smolexpert": calc_metrics(single_preds),
        "best_of_n_jepa_world_model": calc_metrics(best_of_n_preds),
        "mppi_jepa_world_model": calc_metrics(mppi_preds),
        "prior_ensemble_mean": calc_metrics(prior_mean_preds),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cached feature splits from {args.train_manifest} and {args.val_manifest}...")
    train_data = load_split_in_memory(args.train_manifest)
    val_data = load_split_in_memory(args.val_manifest)

    # Fit state/action normalizer strictly on training split
    normalizer = SmolVLANormalizer.from_training_tensors(
        state=train_data["states"],
        action=train_data["actions"],
        action_is_pad=train_data["paddings"],
    )

    train_horizons = (3, 6, 9, 12, 15, 18, 21, 24, 30)
    train_pairs = build_transition_pairs(train_data, train_horizons)
    val_pairs = build_transition_pairs(val_data, (20,))
    print(
        f"Built {len(train_pairs)} multi-horizon training transitions {train_horizons} "
        f"and {len(val_pairs)} held-out validation transitions (H=20)."
    )

    cfg = JEPAWorldModelConfig()
    model = ControlAwareJEPAWorldModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Move compact tensors to GPU for fast multi-horizon training
    tr_ctx = train_data["contexts"].to(device)
    tr_states_norm = normalizer.normalize_state(train_data["states"].to(device))
    tr_actions_norm = normalizer.normalize_action(train_data["actions"].to(device))
    tr_progress = train_data["progress"].to(device)
    tr_goal = train_data["is_goal"].to(device)
    goal_indices = (tr_goal > 0.5).nonzero(as_tuple=True)[0]

    pair_tensor = torch.tensor(train_pairs, dtype=torch.long, device=device)
    best_val_score = -float("inf")
    best_metrics: dict[str, Any] = {}
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(pair_tensor), device=device)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(pair_tensor), args.batch_size):
            batch_pairs = pair_tensor[perm[start : start + args.batch_size]]
            idx_t = batch_pairs[:, 0]
            idx_next = batch_pairs[:, 1]
            horizons = batch_pairs[:, 2]

            h_t = model.encode(tr_ctx[idx_t], tr_states_norm[idx_t], use_target=False)
            h_next_online = model.encode(tr_ctx[idx_next], tr_states_norm[idx_next], use_target=False)
            with torch.no_grad():
                h_next_target = model.encode(tr_ctx[idx_next], tr_states_norm[idx_next], use_target=True)

            # 1. Forward JEPA prediction loss (cosine + smooth L1 in latent space + state MSE)
            h_pred, s_pred = model.predict_forward(
                h_t, tr_states_norm[idx_t], tr_actions_norm[idx_t], horizons
            )
            cos_loss = (1.0 - F.cosine_similarity(h_pred, h_next_target.detach(), dim=-1)).mean()
            l1_loss = F.smooth_l1_loss(h_pred, h_next_target.detach())
            fwd_latent_loss = cos_loss + l1_loss
            fwd_state_loss = F.mse_loss(s_pred, tr_states_norm[idx_next])

            # 2. Inverse dynamics loss (Toso, LeCun et al. Oct 2026 arXiv:2610.07540)
            a_idm = model.predict_inverse(h_t, h_next_online, horizons)
            steps_mask = (
                torch.arange(cfg.max_horizon, device=device).unsqueeze(0) < horizons.unsqueeze(1)
            ).unsqueeze(-1)
            idm_loss = (
                (a_idm - tr_actions_norm[idx_t]).square() * steps_mask
            ).sum() / steps_mask.sum().clamp(min=1)

            # 3. Value / progress + goal head loss on both current and future states
            v_t, g_t = model.predict_value(h_t)
            v_next, g_next = model.predict_value(h_next_online)
            val_loss = (
                F.mse_loss(v_t, tr_progress[idx_t])
                + F.mse_loss(v_next, tr_progress[idx_next])
                + 0.2 * F.binary_cross_entropy_with_logits(g_t, tr_goal[idx_t])
                + 0.2 * F.binary_cross_entropy_with_logits(g_next, tr_goal[idx_next])
            )
            # Temporal ranking margin: future state in a demo must have higher value than current state
            rank_loss = F.relu(v_t - v_next + 0.03 * (horizons.float() / 30.0)).mean()

            # 4. Variance-covariance anti-collapse loss
            vic_loss = model.variance_covariance_loss(torch.cat([h_t, h_next_online], dim=0))

            loss = (
                fwd_latent_loss
                + args.state_weight * fwd_state_loss
                + args.idm_weight * idm_loss
                + args.value_weight * (val_loss + rank_loss)
                + args.vicreg_weight * vic_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_target_encoder()

            epoch_loss += float(loss.item())
            n_batches += 1

        # Update terminal goal prototype from training goal states
        with torch.no_grad():
            if len(goal_indices) > 0:
                h_goals = model.encode(tr_ctx[goal_indices], tr_states_norm[goal_indices], use_target=True)
                model.update_goal_prototype(h_goals, momentum=0.8)

        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            val_metrics = evaluate_world_model(model, val_data, val_pairs, normalizer, device)
            val_cos = val_metrics["latent_cos_sim"]["jepa_action_conditioned"]
            val_no_act = val_metrics["latent_cos_sim"]["jepa_zero_action"]
            top1 = val_metrics["latent_retrieval"]["top1_acc"]
            rho = val_metrics["value_progress_spearman_rho"]
            # Select checkpoint maximizing retrieval + action-conditioning gain + value correlation
            sel_score = top1 + 2.0 * (val_cos - val_no_act) + 0.5 * rho
            if sel_score > best_val_score:
                best_val_score = sel_score
                best_metrics = val_metrics
                save_file(model.state_dict(), str(args.output_dir / "best_jepa_world_model.safetensors"))
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {epoch_loss / max(n_batches, 1):.4f} | "
                f"Val CosSim: {val_cos:.4f} (NoAct: {val_metrics['latent_cos_sim']['jepa_zero_action']:.4f}, "
                f"ID: {val_metrics['latent_cos_sim']['identity_baseline']:.4f}) | "
                f"Top1: {val_metrics['latent_retrieval']['top1_acc'] * 100:.1f}% | "
                f"State RMSE: {val_metrics['future_state_rmse_deg']['jepa_action_conditioned']:.2f}° | "
                f"IDM RMSE: {val_metrics['inverse_dynamics_action_rmse_deg']:.2f}° | "
                f"Val rho: {val_metrics['value_progress_spearman_rho']:.3f}",
                flush=True,
            )

    train_s = time.time() - t0
    model.load_state_dict(
        load_file(str(args.output_dir / "best_jepa_world_model.safetensors"), device=str(device))
    )
    planning_metrics = evaluate_muzero_planning_with_smolexpert(
        model,
        val_data,
        args.policy_run_dir,
        device,
        num_candidates=args.num_candidates,
    )

    report = {
        "config": {
            "idm_weight": args.idm_weight,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_candidates": args.num_candidates,
            "seed": args.seed,
            "train_seconds": train_s,
        },
        "world_model_eval": best_metrics,
        "muzero_planning_eval": planning_metrics,
    }
    (args.output_dir / "jepa_world_model_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
