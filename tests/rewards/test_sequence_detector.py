# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from lerobot.policies.vam.expo_residual import (
    ExpoResidualCritic,
    compute_dr_grpo_advantages,
    compute_marek_score_centered_loss,
)
from lerobot.rewards import (
    MultiFrameSuccessDetector,
    SharedFeatureSequenceDetector,
    SingleFrameResNet50Detector,
    extract_sequence_window,
)


def test_extract_sequence_window():
    frames = torch.randint(0, 255, (50, 3, 64, 64), dtype=torch.uint8)
    states = torch.randn(50, 6)
    seq_f, seq_s = extract_sequence_window(frames, states, end_idx=40, seq_len=8, stride=6)
    assert seq_f.shape == (8, 3, 64, 64)
    assert seq_f.dtype == torch.float32
    assert seq_s.shape == (8, 6)


def test_multi_frame_and_single_frame_detectors():
    seq_frames = torch.rand(4, 8, 3, 128, 128)
    seq_states = torch.randn(4, 8, 6)

    seq_det = MultiFrameSuccessDetector(seq_len=8, state_dim=6, pretrained=False)
    logits, pre_logits = seq_det.forward_with_aux(seq_frames, seq_states)
    probs = seq_det.predict_proba(seq_frames, seq_states)
    assert logits.shape == (4,)
    assert pre_logits.shape == (4,)
    assert probs.shape == (4,)
    assert torch.all((probs >= 0.0) & (probs <= 1.0))

    sf_det = SingleFrameResNet50Detector(pretrained=False)
    sf_logits = sf_det(seq_frames, seq_states)
    sf_probs = sf_det.predict_proba(seq_frames, seq_states)
    assert sf_logits.shape == (4,)
    assert sf_probs.shape == (4,)

    # Test modality dropout and token dropout during training
    seq_det.train()
    logits_drop, pre_drop = seq_det.forward_with_aux(
        seq_frames, seq_states, proprio_dropout=0.5, token_dropout=0.2
    )
    assert logits_drop.shape == (4,)
    assert pre_drop.shape == (4,)

    # Test SharedFeatureSequenceDetector operating directly on pooled Video-VAM tokens [B, T, 24, D]
    shared_det = SharedFeatureSequenceDetector(ctx_dim=64, seq_len=12, state_dim=6)
    seq_ctx = torch.randn(4, 12, 24, 64)
    seq_st_12 = torch.randn(4, 12, 6)
    sh_logits, sh_pre = shared_det.forward_with_aux(
        seq_ctx, seq_st_12, proprio_dropout=0.4, token_dropout=0.2
    )
    sh_probs = shared_det.predict_proba(seq_ctx, seq_st_12)
    assert sh_logits.shape == (4,)
    assert sh_pre.shape == (4,)
    assert sh_probs.shape == (4,)
    assert torch.all((sh_probs >= 0.0) & (sh_probs <= 1.0))


def test_expo_residual_critic_and_dr_grpo_advantages():
    rewards = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    groups = [0, 0, 0, 1, 1, 1]
    adv = compute_dr_grpo_advantages(rewards, groups)
    assert torch.allclose(adv[:3], torch.tensor([1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0]), atol=1e-5)
    assert torch.allclose(adv[3:], torch.tensor([2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0]), atol=1e-5)

    model = ExpoResidualCritic(ctx_dim=64, state_dim=6, chunk_len=30, use_init_anchor=True)
    ctx_raw = torch.randn(2, 600, 64)
    state_deg = torch.randn(2, 6)
    t_norm = torch.tensor([[0.0], [0.5]])
    base_actions = torch.randn(2, 30, 6)

    edited = model.forward_edit(ctx_raw, state_deg, t_norm, base_actions)
    assert edited.shape == (2, 30, 6)
    q_vals = model.forward_q(ctx_raw, state_deg, t_norm, edited, edited)
    assert q_vals.shape == (2,)
    assert torch.all((q_vals >= 0.0) & (q_vals <= 1.0))

    episodes = [
        {
            "steps": [
                {
                    "ctx": torch.randn(24, 64),
                    "state": torch.randn(6),
                    "t_norm": torch.tensor([0.2]),
                    "base_action": torch.randn(30, 6),
                    "exec_action": torch.randn(30, 6),
                }
                for _ in range(4)
            ],
            "reward": 1.0 if ep_i % 2 == 0 else 0.0,
            "spawn_group": 0,
        }
        for ep_i in range(4)
    ]
    stats = model.update_from_episode_buffer(episodes, steps=10, batch_size=8)
    assert stats["num_episodes"] == 4.0
    assert stats["mean_reward"] == 0.5
    assert stats["loss"] >= 0.0

    # Test SmolVLA 64-token (8x8) spatial pooling + Marek Score Centering loss
    smol_model = ExpoResidualCritic(ctx_dim=960, state_dim=6, chunk_len=30, use_init_anchor=True)
    smol_ctx = torch.randn(2, 64, 960)
    smol_edited = smol_model.forward_edit(smol_ctx, state_deg, t_norm, base_actions)
    assert smol_edited.shape == (2, 30, 6)
    sc_loss = compute_marek_score_centered_loss(
        smol_edited,
        base_actions + 2.0,
        base_actions,
        torch.tensor([1.0, 0.0]),
    )
    assert sc_loss.ndim == 0 and sc_loss.item() >= 0.0
    pooled_seq = SharedFeatureSequenceDetector.pool_ctx_sequence(torch.randn(2, 12, 64, 960))
    assert pooled_seq.shape == (2, 12, 24, 960)
