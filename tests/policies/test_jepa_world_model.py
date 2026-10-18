"""Unit tests for ControlAwareJEPAWorldModel and MuZero/MPPI latent search."""

from __future__ import annotations

import torch

from lerobot.policies.vam.jepa_world_model import (
    ControlAwareJEPAWorldModel,
    JEPAWorldModelConfig,
)


def _tiny_cfg() -> JEPAWorldModelConfig:
    return JEPAWorldModelConfig(
        vis_tokens=20,
        vis_dim=32,
        state_dim=6,
        action_dim=6,
        max_horizon=30,
        latent_dim=32,
        num_spatial_queries=4,
        hidden_dim=32,
        predictor_layers=2,
    )


def test_jepa_world_model_forward_inverse_and_value_shapes() -> None:
    cfg = _tiny_cfg()
    model = ControlAwareJEPAWorldModel(cfg)
    vis = torch.randn(4, cfg.vis_tokens, cfg.vis_dim)
    state = torch.randn(4, cfg.state_dim)
    actions = torch.randn(4, cfg.max_horizon, cfg.action_dim)
    horizons = torch.tensor([6, 12, 20, 30], dtype=torch.long)

    h_t = model.encode(vis, state, use_target=False)
    h_target = model.encode(vis, state, use_target=True)
    assert h_t.shape == (4, cfg.latent_dim)
    assert h_target.shape == (4, cfg.latent_dim)

    h_pred, s_pred = model.predict_forward(h_t, state, actions, horizons)
    assert h_pred.shape == (4, cfg.latent_dim)
    assert s_pred.shape == (4, cfg.state_dim)

    a_idm = model.predict_inverse(h_t, h_pred, horizons)
    assert a_idm.shape == (4, cfg.max_horizon, cfg.action_dim)

    v_pred, g_logit = model.predict_value(h_t)
    assert v_pred.shape == (4,)
    assert g_logit.shape == (4,)
    assert bool(((v_pred >= 0.0) & (v_pred <= 1.0)).all().item())

    vic = model.variance_covariance_loss(h_t)
    assert vic.ndim == 0 and torch.isfinite(vic)


def test_jepa_world_model_mppi_planning_and_goal_prototype() -> None:
    cfg = _tiny_cfg()
    model = ControlAwareJEPAWorldModel(cfg).eval()
    vis = torch.randn(1, cfg.vis_tokens, cfg.vis_dim)
    state = torch.randn(1, cfg.state_dim)
    h_t = model.encode(vis, state)
    model.update_goal_prototype(torch.randn(3, cfg.latent_dim))
    assert bool(model.goal_prototype_initialized.item())

    prior_actions = torch.randn(8, cfg.max_horizon, cfg.action_dim)
    scored = model.score_action_candidates(h_t, state, prior_actions)
    assert scored["score"].shape == (8,)

    planned, info = model.plan_mppi(
        h_t,
        state,
        prior_actions,
        num_iters=2,
        num_perturbations=8,
    )
    assert planned.shape == (cfg.max_horizon, cfg.action_dim)
    assert torch.isfinite(planned).all()
    assert info["final_planned_score"] >= info["initial_prior_score"] - 1e-5
