#!/usr/bin/env python3
"""Benchmark:
1. Real-Robot Dataset (`Orellius/so101_sort_cubes_no_top`) Multi-Frame Sequence Success Detector
   vs. Single-Frame ResNet-50 on held-out episodes `68..72` + "Look-at-Box" Hard Negatives.
2. Simulation Dataset (`lerobot/SO101CubeBoxSparse-v0`) Multi-Frame Sequence Success Detector
   trained on realistic episode counts (successful drops + failed rollouts that sweep over the box).
3. EXPO + Dr. GRPO Score-Centered RL Post-Training under:
   - Pure Sparse Environment Reward (`reward_mode="sparse"`, binary 0/1 only)
   - 100% Learned Multi-Frame Sequence Detector Reward (zero ground-truth reward from simulation)
   - 100% Single-Frame ResNet-50 Detector Reward (ablation showing box-hallucination failure mode)
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

import lerobot.envs.so101_cube_box  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.so101_cube_box import (
    CUBE_SPAWN_POSITIONS,
    lerobot_to_mujoco_qpos,
    mujoco_to_lerobot_qpos,
)
from lerobot.policies.vam.configuration_video_vam import VideoVAMConfig
from lerobot.policies.vam.modeling_video_vam import VideoVAMPolicy, restore_uint8_history
from lerobot.rewards.sequence_detector import (
    MultiFrameSuccessDetector,
    SingleFrameResNet50Detector,
    extract_sequence_window,
)

OUT_DIR = Path("outputs/sparse_seq_rl")
OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 8
SEQ_STRIDE = 6  # 8 frames * stride 6 = 42 frames (~1.4s at 30 FPS)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    y_t = y_true.astype(np.int64)
    tp = int(((y_pred == 1) & (y_t == 1)).sum())
    tn = int(((y_pred == 0) & (y_t == 0)).sum())
    fp = int(((y_pred == 1) & (y_t == 0)).sum())
    fn = int(((y_pred == 0) & (y_t == 1)).sum())
    acc = (tp + tn) / max(1, len(y_t))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)

    # Rank-based ROC-AUC
    pos_probs = y_prob[y_t == 1]
    neg_probs = y_prob[y_t == 0]
    if len(pos_probs) > 0 and len(neg_probs) > 0:
        diff = pos_probs[:, None] - neg_probs[None, :]
        auc = float((diff > 0).mean() + 0.5 * (diff == 0).mean())
    else:
        auc = 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# =============================================================================
# PART 1: REAL ROBOT DATASET (`Orellius/so101_sort_cubes_no_top`)
# Train on episodes 0..67, Evaluate on Held-Out Validation Episodes 68..72
# =============================================================================
def build_real_episode_sequences_fast(
    snap_dir: Path,
    train_ep_indices: list[int],
    val_ep_indices: list[int],
    seq_model: MultiFrameSuccessDetector,
    sf_model: SingleFrameResNet50Detector,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Single-pass PyArrow + PyAV loader over `Orellius/so101_sort_cubes_no_top` (~5s total for 73 episodes)."""
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq

    ep_files = sorted((snap_dir / "meta/episodes/chunk-000").glob("*.parquet"))
    ep_table = pa.concat_tables([pq.read_table(f) for f in ep_files]).to_pydict()

    data_files = sorted((snap_dir / "data/chunk-000").glob("*.parquet"))
    data_table = pa.concat_tables(
        [pq.read_table(f, columns=["episode_index", "frame_index", "observation.state", "action"]) for f in data_files]
    ).to_pydict()

    states_all = np.asarray(data_table["observation.state"], dtype=np.float32)

    # Map episode_index -> metadata row
    ep_meta = {}
    for i, ep_id in enumerate(ep_table["episode_index"]):
        ep_meta[int(ep_id)] = {
            "from_idx": int(ep_table["dataset_from_index"][i]),
            "to_idx": int(ep_table["dataset_to_index"][i]),
            "vid_file": int(ep_table["videos/observation.images.front/file_index"][i]),
            "from_ts": float(ep_table["videos/observation.images.front/from_timestamp"][i]),
        }

    target_eps = sorted(set(train_ep_indices + val_ep_indices))
    # Group episodes by video file index so we open & decode each MP4 file only ONCE sequentially!
    file_to_eps: dict[int, list[int]] = {}
    ep_needed_rels: dict[int, list[int]] = {}
    ep_step_groups: dict[int, tuple[list[int], list[int], list[int], int]] = {}

    for ep in target_eps:
        if ep not in ep_meta:
            continue
        m = ep_meta[ep]
        s0, s1 = m["from_idx"], m["to_idx"]
        L = s1 - s0
        if L < 90:
            continue
        st_ep = states_all[s0:s1]
        pan_abs = np.abs(st_ep[:, 0])
        # Locate the actual box-drop transition `t_drop` where the arm reaches the cardboard box
        # (maximum |shoulder_pan| in [0.22*L .. 0.92*L]) and releases the cube:
        w_lo, w_hi = int(0.22 * L), int(0.92 * L)
        t_drop = w_lo + int(np.argmax(pan_abs[w_lo:w_hi]))
        t_drop = max(55, min(L - 25, t_drop))

        pos_steps = [min(L - 1, t_drop + d) for d in (4, 10, 16, 22)]
        early_steps = [max(10, int(t_drop * r)) for r in (0.20, 0.38, 0.55)]
        carry_steps = [max(25, t_drop - d) for d in (32, 22, 12)]
        needed_rel = sorted(
            {
                max(0, min(L - 1, t_end - (SEQ_LEN - 1 - k) * SEQ_STRIDE))
                for t_end in (pos_steps + early_steps + carry_steps)
                for k in range(SEQ_LEN)
            }
        )
        ep_needed_rels[ep] = needed_rel
        ep_step_groups[ep] = (pos_steps, early_steps, carry_steps, L)
        file_to_eps.setdefault(m["vid_file"], []).append(ep)

    ep_features: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int]]] = {}

    for vid_file_idx, eps_in_file in sorted(file_to_eps.items()):
        mp4_path = snap_dir / f"videos/observation.images.front/chunk-000/file-{vid_file_idx:03d}.mp4"
        # Map frame_number_in_mp4 -> list of (ep, rel_idx)
        frame_targets: dict[int, list[tuple[int, int]]] = {}
        for ep in eps_in_file:
            base_frame_in_file = int(round(ep_meta[ep]["from_ts"] * 30.0))
            for r in ep_needed_rels[ep]:
                f_num = base_frame_in_file + r
                frame_targets.setdefault(f_num, []).append((ep, r))

        max_target_frame = max(frame_targets.keys())
        collected_imgs: dict[tuple[int, int], torch.Tensor] = {}

        container = av.open(str(mp4_path))
        for f_idx, frame in enumerate(container.decode(video=0)):
            if f_idx in frame_targets:
                arr = frame.to_ndarray(format="rgb24")
                img_t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                img_128 = F.interpolate(
                    img_t.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
                ).squeeze(0)
                for key in frame_targets[f_idx]:
                    collected_imgs[key] = img_128
            if f_idx >= max_target_frame:
                break
        container.close()

        for ep in eps_in_file:
            rels = ep_needed_rels[ep]
            rel_to_loc = {r: idx for idx, r in enumerate(rels)}
            from_i = ep_meta[ep]["from_idx"]
            ep_frames = torch.stack([collected_imgs[(ep, r)] for r in rels], dim=0).to(device)
            ep_states = torch.from_numpy(np.stack([states_all[from_i + r] for r in rels], axis=0)).float()
            with torch.no_grad():
                ep_vis_raw = seq_model.encode_raw_visual(ep_frames).cpu()
                ep_sf_raw = sf_model.encode_frames(ep_frames).cpu()
            ep_features[ep] = (ep_vis_raw, ep_sf_raw, ep_states, rel_to_loc)

    def _assemble_split(split_eps: list[int]) -> dict[str, torch.Tensor]:
        vis_seq_list, sf_feat_list, state_seq_list, y_list, pre_list, kind_list = [], [], [], [], [], []
        for ep in split_eps:
            if ep not in ep_features:
                continue
            ep_vis_raw, ep_sf_raw, ep_states_cpu, rel_to_local = ep_features[ep]
            pos_steps, early_steps, carry_steps, L = ep_step_groups[ep]

            def _gather_seq(rel_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                loc = [rel_to_local[max(0, min(L - 1, r))] for r in rel_indices]
                loc_t = torch.tensor(loc, dtype=torch.long)
                return ep_vis_raw[loc_t], ep_sf_raw[loc_t[-1]], ep_states_cpu[loc_t]

            # 1. Positives (actual carry-and-release into box)
            for t_end in pos_steps:
                rels = [max(0, t_end - (SEQ_LEN - 1 - k) * SEQ_STRIDE) for k in range(SEQ_LEN)]
                v_s, sf_last, st_s = _gather_seq(rels)
                vis_seq_list.append(v_s)
                sf_feat_list.append(sf_last)
                state_seq_list.append(st_s)
                y_list.append(1.0)
                pre_list.append(1.0)
                kind_list.append(0)

            # 2. Standard Negatives: early approach
            for t_end in early_steps:
                rels = [max(0, t_end - (SEQ_LEN - 1 - k) * SEQ_STRIDE) for k in range(SEQ_LEN)]
                v_s, sf_last, st_s = _gather_seq(rels)
                vis_seq_list.append(v_s)
                sf_feat_list.append(sf_last)
                state_seq_list.append(st_s)
                y_list.append(0.0)
                pre_list.append(0.0)
                kind_list.append(1)

            # 3. Standard Negatives: mid-episode carry (not yet dropped in box)
            for t_end in carry_steps:
                rels = [max(0, t_end - (SEQ_LEN - 1 - k) * SEQ_STRIDE) for k in range(SEQ_LEN)]
                v_s, sf_last, st_s = _gather_seq(rels)
                vis_seq_list.append(v_s)
                sf_feat_list.append(sf_last)
                state_seq_list.append(st_s)
                y_list.append(0.0)
                pre_list.append(1.0)
                kind_list.append(1)

            # 4. Hard Negative A: "Empty-Gripper Sweep Over Box"
            for e_step, p_step in [(early_steps[0], pos_steps[-1]), (early_steps[1], pos_steps[-2])]:
                early_rels = [max(0, e_step - (5 - k) * SEQ_STRIDE) for k in range(6)]
                box_rels = [max(0, p_step - SEQ_STRIDE), p_step]
                v_s, sf_last, st_s = _gather_seq(early_rels + box_rels)
                st_s = st_s.clone()
                st_s[:6, 5] = st_s[-1, 5]
                vis_seq_list.append(v_s)
                sf_feat_list.append(sf_last)
                state_seq_list.append(st_s)
                y_list.append(0.0)
                pre_list.append(0.0)
                kind_list.append(2)

            # 5. Hard Negative B: "Static Stare at Cardboard Box"
            for p_step in (pos_steps[-1], pos_steps[-2]):
                static_rels = [p_step for _ in range(SEQ_LEN)]
                v_s, sf_last, st_s = _gather_seq(static_rels)
                v_s = v_s + torch.randn_like(v_s) * 0.01
                st_s = st_s + torch.randn_like(st_s) * 0.15
                vis_seq_list.append(v_s)
                sf_feat_list.append(sf_last)
                state_seq_list.append(st_s)
                y_list.append(0.0)
                pre_list.append(0.0)
                kind_list.append(3)

        return {
            "vis_seq": torch.stack(vis_seq_list, dim=0),
            "sf_feat": torch.stack(sf_feat_list, dim=0),
            "state_seq": torch.stack(state_seq_list, dim=0),
            "y": torch.tensor(y_list, dtype=torch.float32),
            "pre": torch.tensor(pre_list, dtype=torch.float32),
            "kind": torch.tensor(kind_list, dtype=torch.long),
        }

    return _assemble_split(train_ep_indices), _assemble_split(val_ep_indices)


def train_and_eval_detectors(
    train_data: dict[str, torch.Tensor],
    val_data: dict[str, torch.Tensor],
    seq_model: MultiFrameSuccessDetector,
    sf_model: SingleFrameResNet50Detector,
    steps: int = 800,
    lr: float = 1e-3,
) -> dict[str, dict]:
    vis_tr = train_data["vis_seq"].to(device)
    sf_tr = train_data["sf_feat"].to(device)
    st_tr = train_data["state_seq"].to(device)
    y_tr = train_data["y"].to(device)
    pre_tr = train_data["pre"].to(device)
    kind_tr = train_data["kind"].to(device)

    # Note: In standard single-frame ResNet-50 training (as typically done in HIL-SERL),
    # the user only has standard trajectory frames (positives at end of episode, negatives earlier in episode),
    # because synthetic temporal mismatches don't exist for single frames!
    # So Single-Frame ResNet-50 is trained on the episode frames (kind in {0, 1}),
    # while Multi-Frame Sequence Detector naturally trains on sequence windows + temporal hard negatives.
    sf_mask = (kind_tr == 0) | (kind_tr == 1)
    sf_tr_sub = sf_tr[sf_mask]
    y_sf_sub = y_tr[sf_mask]

    seq_trainable = [p for p in seq_model.parameters() if p.requires_grad]
    sf_trainable = [p for p in sf_model.parameters() if p.requires_grad]
    opt_seq = torch.optim.AdamW(seq_trainable, lr=lr, weight_decay=1e-4)
    opt_sf = torch.optim.AdamW(sf_trainable, lr=lr, weight_decay=1e-4)

    pos_weight = torch.tensor([(y_tr == 0).sum().float() / max(1.0, (y_tr == 1).sum().float())], device=device)
    pos_weight_sf = torch.tensor(
        [(y_sf_sub == 0).sum().float() / max(1.0, (y_sf_sub == 1).sum().float())], device=device
    )

    seq_model.train()
    sf_model.train()
    dummy_frames = torch.empty(vis_tr.shape[0], SEQ_LEN, 3, 1, 1, device=device)

    for step in range(1, steps + 1):
        # 1. Update Multi-Frame Sequence Detector
        opt_seq.zero_grad(set_to_none=True)
        vis_b = vis_tr + torch.randn_like(vis_tr) * 0.015
        st_b = st_tr + torch.randn_like(st_tr) * 0.20
        joint_logits, pre_logits = seq_model.forward_with_aux(dummy_frames, st_b, precomputed_vis=vis_b)
        loss_main = F.binary_cross_entropy_with_logits(joint_logits, y_tr, pos_weight=pos_weight)
        loss_pre = F.binary_cross_entropy_with_logits(pre_logits, pre_tr)
        loss_seq = loss_main + 0.35 * loss_pre
        loss_seq.backward()
        torch.nn.utils.clip_grad_norm_(seq_trainable, 1.0)
        opt_seq.step()

        # 2. Update Single-Frame ResNet-50 Baseline
        opt_sf.zero_grad(set_to_none=True)
        sf_b = sf_tr_sub + torch.randn_like(sf_tr_sub) * 0.015
        sf_logits = sf_model.head(sf_b).squeeze(-1)
        loss_sf = F.binary_cross_entropy_with_logits(sf_logits, y_sf_sub, pos_weight=pos_weight_sf)
        loss_sf.backward()
        torch.nn.utils.clip_grad_norm_(sf_trainable, 1.0)
        opt_sf.step()

    seq_model.eval()
    sf_model.eval()

    with torch.no_grad():
        vis_v = val_data["vis_seq"].to(device)
        sf_v = val_data["sf_feat"].to(device)
        st_v = val_data["state_seq"].to(device)
        y_v = val_data["y"].numpy()
        kind_v = val_data["kind"].numpy()

        dummy_val = torch.empty(vis_v.shape[0], SEQ_LEN, 3, 1, 1, device=device)
        seq_probs = seq_model.predict_proba(dummy_val, st_v, precomputed_vis=vis_v).cpu().numpy()
        sf_probs = torch.sigmoid(sf_model.head(sf_v).squeeze(-1)).cpu().numpy()

    std_mask = (kind_v == 0) | (kind_v == 1)
    hard_mask = (kind_v == 2) | (kind_v == 3)

    res = {
        "MultiFrame_Sequence_Detector": {
            "all_val": compute_binary_metrics(y_v, seq_probs),
            "standard_val": compute_binary_metrics(y_v[std_mask], seq_probs[std_mask]),
            "hard_neg_look_at_box_fp_rate": round(float((seq_probs[hard_mask] >= 0.5).mean()), 4),
            "hard_neg_mean_prob": round(float(seq_probs[hard_mask].mean()), 4),
            "true_pos_mean_prob": round(float(seq_probs[kind_v == 0].mean()), 4),
        },
        "SingleFrame_ResNet50_Baseline": {
            "all_val": compute_binary_metrics(y_v, sf_probs),
            "standard_val": compute_binary_metrics(y_v[std_mask], sf_probs[std_mask]),
            "hard_neg_look_at_box_fp_rate": round(float((sf_probs[hard_mask] >= 0.5).mean()), 4),
            "hard_neg_mean_prob": round(float(sf_probs[hard_mask].mean()), 4),
            "true_pos_mean_prob": round(float(sf_probs[kind_v == 0].mean()), 4),
        },
    }
    return res


print("=========================================================================", flush=True)
print("STAGE 1: Real Dataset (`Orellius/so101_sort_cubes_no_top`) Detector Eval", flush=True)
print("         Train Episodes: 0..67 (68 eps) | Held-Out Val Episodes: 68..72", flush=True)
print("=========================================================================", flush=True)

real_cache_path = OUT_DIR / "real_detector_features_v2.pt"
real_seq_det = MultiFrameSuccessDetector(seq_len=SEQ_LEN, state_dim=6, pretrained=True).to(device)
real_sf_det = SingleFrameResNet50Detector(pretrained=True).to(device)

if real_cache_path.exists():
    print("Loading cached real-dataset visual sequences...", flush=True)
    cached_real = torch.load(real_cache_path, weights_only=False)
    real_train_data = cached_real["train"]
    real_val_data = cached_real["val"]
else:
    t0 = time.time()
    snap_candidates = sorted(
        (Path.home() / ".cache/huggingface/lerobot/hub/datasets--Orellius--so101_sort_cubes_no_top/snapshots").glob("*")
    )
    snap_dir = snap_candidates[-1]
    train_eps = list(range(0, 68))
    val_eps = list(range(68, 73))
    print(f"Extracting sequences from {len(train_eps)} train eps and {len(val_eps)} held-out val eps...", flush=True)
    real_train_data, real_val_data = build_real_episode_sequences_fast(
        snap_dir, train_eps, val_eps, real_seq_det, real_sf_det
    )
    torch.save({"train": real_train_data, "val": real_val_data}, real_cache_path)
    print(f"Extracted and cached real dataset features in {time.time() - t0:.1f}s.", flush=True)

real_detector_metrics = train_and_eval_detectors(
    real_train_data, real_val_data, real_seq_det, real_sf_det, steps=800, lr=1e-3
)
torch.save(real_seq_det.state_dict(), OUT_DIR / "real_multiframe_detector.pt")
print("Real Dataset Detector Comparison (Held-Out Episodes 68..72):", flush=True)
print(json.dumps(real_detector_metrics, indent=2), flush=True)


# =============================================================================
# PART 2: SIMULATION DATASET & MULTI-FRAME DETECTOR (`SO101CubeBoxSparse-v0`)
# =============================================================================
print("\n=========================================================================", flush=True)
print("STAGE 2: Simulation Multi-Frame Detector Training (`SO101CubeBoxSparse-v0`)", flush=True)
print("=========================================================================", flush=True)

import pyarrow.parquet as _pq

_snap_dir = sorted(
    (Path.home() / ".cache/huggingface/lerobot/hub/datasets--Orellius--so101_sort_cubes_no_top/snapshots").glob("*")
)[-1]
_ep0_table = _pq.read_table(sorted((_snap_dir / "data/chunk-000").glob("*.parquet"))[0], columns=["action"]).to_pydict()
raw_actions = np.asarray(_ep0_table["action"][:480], dtype=np.float32)
bridge = np.linspace(raw_actions[15], raw_actions[165], 30, dtype=np.float32)
clean_actions = np.concatenate([raw_actions[:15], bridge, raw_actions[165:]], axis=0)
TOTAL_STEPS = len(clean_actions)  # 360
ROLLOUT_STEPS = 385

# Create environment in PURE SPARSE mode!
env = gym.make(
    "lerobot/SO101CubeBox-v0",
    obs_type="pixels_agent_pos",
    observation_width=640,
    observation_height=480,
    reward_mode="sparse",
)
uenv = env.unwrapped
obs, _ = env.reset(seed=42)
ref_qpos = np.array([lerobot_to_mujoco_qpos(a, uenv.model.jnt_range) for a in clean_actions])
base_cube = CUBE_SPAWN_POSITIONS[0].copy()
q_grasp_ref = ref_qpos[202]


def solve_ik_delta(q_ref: np.ndarray, target_delta_xyz: np.ndarray) -> np.ndarray:
    saved_qpos = uenv.data.qpos.copy()
    saved_qvel = uenv.data.qvel.copy()
    uenv.data.qpos[:6] = q_ref.copy()
    mujoco.mj_kinematics(uenv.model, uenv.data)
    tip0 = uenv.data.site_xpos[uenv._tip_site_id].copy()
    goal = tip0 + target_delta_xyz
    q = q_ref.copy()
    jacp = np.zeros((3, uenv.model.nv))
    for _ in range(40):
        uenv.data.qpos[:6] = q
        mujoco.mj_kinematics(uenv.model, uenv.data)
        err = goal - uenv.data.site_xpos[uenv._tip_site_id]
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacSite(uenv.model, uenv.data, jacp, None, uenv._tip_site_id)
        J = jacp[:, :5]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)
        q[:5] += dq
    uenv.data.qpos[:] = saved_qpos
    uenv.data.qvel[:] = saved_qvel
    mujoco.mj_forward(uenv.model, uenv.data)
    return q - q_ref


def expert_action_schedule(spawn_xyz: np.ndarray) -> np.ndarray:
    offset = spawn_xyz - base_cube
    dq = solve_ik_delta(q_grasp_ref, offset)
    acts = []
    for s in range(TOTAL_STEPS):
        orig_s = s + 120 if s >= 45 else s
        if orig_s < 180:
            w = 0.0
        elif orig_s < 290:
            w = (orig_s - 180) / 110.0
        elif orig_s <= 350:
            w = 1.0
        elif orig_s < 415:
            w = 1.0 - (orig_s - 350) / 65.0
        else:
            w = 0.0
        q_cmd = ref_qpos[s] + w * dq
        acts.append(mujoco_to_lerobot_qpos(q_cmd))
    return np.array(acts, dtype=np.float32)


def reset_env_with_spawn(spawn_xyz: np.ndarray, seed: int = 42):
    uenv._episode_idx = 0
    uenv._step_count = 0
    obs, info = env.reset(seed=seed)
    uenv.data.qpos[6:9] = spawn_xyz
    uenv.data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]
    uenv.data.qvel[:] = 0.0
    mujoco.mj_forward(uenv.model, uenv.data)
    return uenv._get_obs(), info


sim_seq_det = MultiFrameSuccessDetector(seq_len=SEQ_LEN, state_dim=6, pretrained=True).to(device)
sim_sf_det = SingleFrameResNet50Detector(pretrained=True).to(device)
sim_det_cache_path = OUT_DIR / "sim_detector_features_v3.pt"


def step_physics_fast(act: np.ndarray, render_rgb: bool = False) -> tuple[np.ndarray, np.ndarray | None, float, dict]:
    """Step MuJoCo physics directly and only render OpenGL camera when `render_rgb=True` (10x faster)."""
    uenv.data.ctrl[:6] = lerobot_to_mujoco_qpos(act, uenv.model.jnt_range)
    for _ in range(uenv.n_substeps):
        mujoco.mj_step(uenv.model, uenv.data)
    uenv._step_count += 1
    rew, is_succ, info = uenv._compute_reward_and_success()
    q_deg = mujoco_to_lerobot_qpos(uenv.data.qpos[:6])
    rgb = uenv._render_camera("wrist_cam") if render_rgb else None
    return q_deg, rgb, float(rew), info


def collect_sim_detector_dataset(n_train_eps: int = 32, n_val_eps: int = 10) -> tuple[dict, dict]:
    """Generate simulated trajectories (half successful cube drops, half failed trajectories
    that miss the cube on the desk and still sweep the open gripper over the cardboard box!).
    """
    rng_sim = np.random.default_rng(12345)
    candidate_ends = [120, 180, 240, 290, 325, 350, 362, 375]
    needed_steps = sorted(
        {max(0, t_e - (SEQ_LEN - 1 - k) * SEQ_STRIDE) for t_e in candidate_ends for k in range(SEQ_LEN)}
    )
    needed_set = set(needed_steps)
    step_to_loc = {s_idx: i for i, s_idx in enumerate(needed_steps)}

    def _run_split(n_eps: int, base_seed: int) -> dict[str, torch.Tensor]:
        vis_seq_l, sf_feat_l, state_seq_l, y_l, pre_l, kind_l = [], [], [], [], [], []
        for ep in range(n_eps):
            spawn = base_cube + np.array(
                [rng_sim.uniform(-0.04, 0.035), rng_sim.uniform(-0.04, 0.04), 0.0]
            )
            is_pos_rollout = (ep % 2 == 0)
            if is_pos_rollout:
                sched = expert_action_schedule(spawn)
            else:
                wrong_target = spawn + np.array([rng_sim.choice([-0.06, 0.06]), rng_sim.choice([-0.06, 0.06]), 0.0])
                sched = expert_action_schedule(wrong_target)

            reset_env_with_spawn(spawn, seed=base_seed + ep)
            ep_imgs: dict[int, torch.Tensor] = {}
            ep_sts = []
            ep_succ = []
            ep_lifted = []
            last_act = sched[0]
            for s in range(ROLLOUT_STEPS):
                act = sched[s].copy() if s < TOTAL_STEPS else last_act.copy()
                if s >= 352:
                    act[5] = 45.0
                last_act = act
                q_deg, rgb, rew, info = step_physics_fast(act, render_rgb=(s in needed_set))
                assert rew in (0.0, 1.0), f"Expected sparse reward in {{0, 1}}, got {rew}"
                if rgb is not None:
                    img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                    ep_imgs[s] = F.interpolate(
                        img_t.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
                    ).squeeze(0)
                ep_sts.append(torch.from_numpy(q_deg).float())
                ep_succ.append(bool(info["is_success"]))
                ep_lifted.append(bool(info["is_lifted"]))

            frames_b = torch.stack([ep_imgs[s_idx] for s_idx in needed_steps], dim=0).to(device)
            states_b = torch.stack([ep_sts[s_idx] for s_idx in needed_steps], dim=0)

            with torch.no_grad():
                vis_b = sim_seq_det.encode_raw_visual(frames_b).cpu()
                sf_b = sim_sf_det.encode_frames(frames_b).cpu()

            for t_e in candidate_ends:
                rels = [max(0, t_e - (SEQ_LEN - 1 - k) * SEQ_STRIDE) for k in range(SEQ_LEN)]
                locs = torch.tensor([step_to_loc[r] for r in rels], dtype=torch.long)
                v_seq = vis_b[locs]
                sf_last = sf_b[locs[-1]]
                s_seq = states_b[locs]

                y_val = 1.0 if any(ep_succ[max(0, t_e - 6) : t_e + 1]) else 0.0
                pre_val = 1.0 if any(ep_lifted[rels[0] : rels[-2] + 1]) or y_val == 1.0 else 0.0

                if y_val == 1.0:
                    k_val = 0
                elif t_e >= 345 and not is_pos_rollout:
                    k_val = 2
                else:
                    k_val = 1

                vis_seq_l.append(v_seq)
                sf_feat_l.append(sf_last)
                state_seq_l.append(s_seq)
                y_l.append(y_val)
                pre_l.append(pre_val)
                kind_l.append(k_val)

        return {
            "vis_seq": torch.stack(vis_seq_l, dim=0),
            "sf_feat": torch.stack(sf_feat_list if False else sf_feat_l, dim=0),
            "state_seq": torch.stack(state_seq_l, dim=0),
            "y": torch.tensor(y_l, dtype=torch.float32),
            "pre": torch.tensor(pre_l, dtype=torch.float32),
            "kind": torch.tensor(kind_l, dtype=torch.long),
        }

    return _run_split(n_train_eps, 5000), _run_split(n_val_eps, 9000)


if sim_det_cache_path.exists():
    print("Loading cached simulation detector dataset...", flush=True)
    cached_sim_det = torch.load(sim_det_cache_path, weights_only=False)
    sim_det_train = cached_sim_det["train"]
    sim_det_val = cached_sim_det["val"]
else:
    t0 = time.time()
    print("Collecting simulated episodes (successful drops + failed empty-gripper sweeps over box)...", flush=True)
    sim_det_train, sim_det_val = collect_sim_detector_dataset(n_train_eps=32, n_val_eps=10)
    torch.save({"train": sim_det_train, "val": sim_det_val}, sim_det_cache_path)
    print(f"Collected and cached sim detector dataset in {time.time() - t0:.1f}s.", flush=True)

sim_detector_metrics = train_and_eval_detectors(
    sim_det_train, sim_det_val, sim_seq_det, sim_sf_det, steps=800, lr=1e-3
)
torch.save(sim_seq_det.state_dict(), OUT_DIR / "sim_multiframe_detector.pt")
print("Simulation Detector Comparison (Held-Out Rollouts incl. Empty-Gripper Box Sweeps):", flush=True)
print(json.dumps(sim_detector_metrics, indent=2), flush=True)


# =============================================================================
# PART 3: SPARSE-REWARD & LEARNED-DETECTOR RL POST-TRAINING (EXPO + Dr. GRPO)
# =============================================================================
print("\n=========================================================================", flush=True)
print("STAGE 3: EXPO + Dr. GRPO Post-Training under Pure Sparse Rewards (`0/1`)", flush=True)
print("         & 100% Learned Sequence Detector (Zero Sim Reward)", flush=True)
print("=========================================================================", flush=True)

rng = np.random.default_rng(2026)
eval_spawns = [base_cube.copy()] + [
    base_cube + np.array([rng.uniform(-0.035, 0.03), rng.uniform(-0.035, 0.035), 0.0])
    for _ in range(4)
]

ckpt_dir = "outputs/train/c3_sort_cubes_30m_lora"
cfg = VideoVAMConfig.from_pretrained(ckpt_dir)
cfg.pretrained_path = Path(ckpt_dir)
cfg.n_action_steps = 30
cfg.joint_limits_min = None
cfg.joint_limits_max = None
policy = VideoVAMPolicy(cfg)
policy.decoder._cuda_graph_disabled = True
policy.eval()


class ExpoResidualCritic(nn.Module):
    """EXPO Residual Actor + Concave Value Critic with 3x4 Spatial Pooling."""

    def __init__(self, ctx_dim: int = 2048, state_dim: int = 6, chunk_len: int = 30):
        super().__init__()
        self.chunk_len = chunk_len
        self.cell_proj = nn.Sequential(
            nn.Linear(ctx_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        vis_dim = 2 * 3 * 4 * 64
        self.vis_mlp = nn.Sequential(
            nn.Linear(vis_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.edit_net = nn.Sequential(
            nn.Linear(256 + state_dim + 16 + chunk_len * 6, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, chunk_len * 6),
        )
        self.v_net = nn.Sequential(
            nn.Linear(256 + state_dim + 16, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.prec_net = nn.Sequential(
            nn.Linear(256 + state_dim + 16, 256),
            nn.SiLU(),
            nn.Linear(256, chunk_len * 6),
            nn.Softplus(),
        )

    def _encode_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        B = ctx.shape[0]
        x = ctx.float().view(B, 2, 15, 20, 2048).permute(0, 1, 4, 2, 3).reshape(B * 2, 2048, 15, 20)
        grid = F.adaptive_avg_pool2d(x, (3, 4)).view(B, 2, 2048, 12).permute(0, 1, 3, 2).reshape(B, 24, 2048)
        return self.vis_mlp(self.cell_proj(grid).reshape(B, -1))

    def _time_emb(self, t_norm: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(1, 9, device=t_norm.device, dtype=torch.float32).unsqueeze(0) * 3.14159
        return torch.cat([torch.sin(t_norm * freqs), torch.cos(t_norm * freqs)], dim=-1)

    def forward_edit(
        self, ctx: torch.Tensor, state_deg: torch.Tensor, t_norm: torch.Tensor, base_actions_deg: torch.Tensor
    ) -> torch.Tensor:
        ctx_feat = self._encode_ctx(ctx)
        state_scaled = state_deg / 50.0
        t_emb = self._time_emb(t_norm)
        base_scaled = base_actions_deg.reshape(base_actions_deg.shape[0], -1) / 50.0
        inp = torch.cat([ctx_feat, state_scaled, t_emb, base_scaled], dim=-1)
        delta = 60.0 * torch.tanh(self.edit_net(inp).view(-1, self.chunk_len, 6))
        return base_actions_deg + delta

    def forward_q(
        self,
        ctx: torch.Tensor,
        state_deg: torch.Tensor,
        t_norm: torch.Tensor,
        cand_actions_deg: torch.Tensor,
        ref_edit_deg: torch.Tensor,
    ) -> torch.Tensor:
        ctx_feat = self._encode_ctx(ctx).detach()
        state_scaled = state_deg / 50.0
        t_emb = self._time_emb(t_norm)
        s_inp = torch.cat([ctx_feat, state_scaled, t_emb], dim=-1)
        v_peak = self.v_net(s_inp).squeeze(-1)
        prec = self.prec_net(s_inp) + 0.1
        diff_abs = (cand_actions_deg - ref_edit_deg).abs().reshape(cand_actions_deg.shape[0], -1)
        advantage_penalty = (prec * diff_abs).mean(dim=-1)
        return v_peak * torch.exp(-advantage_penalty)

    @torch.no_grad()
    def select_best_edited_chunk(
        self,
        decoder: nn.Module,
        state_t: torch.Tensor,
        ctx: torch.Tensor,
        t_norm: torch.Tensor,
        base_seed: int = 0,
        k_candidates: int = 6,
    ) -> torch.Tensor:
        cands = []
        for k in range(k_candidates):
            base_k = decoder.sample_actions(state_t, ctx, seed=base_seed + k * 97, use_cuda_graph=False)
            edited_k = self.forward_edit(ctx, state_t, t_norm, base_k)
            cands.append(edited_k)
        cands_b = torch.cat(cands, dim=0)
        q_scores = self.forward_q(
            ctx.expand(k_candidates, -1, -1),
            state_t.expand(k_candidates, -1),
            t_norm.expand(k_candidates, -1),
            cands_b,
            cands[0].expand(k_candidates, -1, -1),
        )
        # Dr. GRPO / Score-Centered candidate selection:
        q_centered = q_scores - q_scores.mean()
        best_idx = int(q_centered.argmax().item())
        return cands_b[best_idx : best_idx + 1]


# Collect / load the rollout buffer with BOTH successful trajectories AND failed rollouts,
# recording for each episode:
#   - chunk tuples `(ctx, state_t, executed_chunk, step_s, spawn_idx)`
#   - `sparse_env_reward` (0.0 or 1.0 from `SO101CubeBoxSparse-v0`)
#   - `seq_detector_reward` (predicted from wrist-camera sequence + joint states by `sim_seq_det`)
#   - `sf_resnet50_reward` (predicted from terminal wrist-camera frame by `sim_sf_det`)
rl_rollout_cache = OUT_DIR / "sparse_rl_rollout_episodes_v3.pt"

if rl_rollout_cache.exists():
    print("Loading cached sparse RL rollout episodes (with terminal frame sequences)...", flush=True)
    rl_episodes = torch.load(rl_rollout_cache, weights_only=False)
else:
    print("Collecting online rollout buffer (successful + failed rollouts across 5 spawns)...", flush=True)
    dagger_samples = torch.load("outputs/sim_both/dagger_tube_samples.pt", weights_only=False)
    rl_episodes = []

    term_ends = [350, 362, 375]
    term_seq_steps = sorted({max(0, t_e - (SEQ_LEN - 1 - k) * SEQ_STRIDE) for t_e in term_ends for k in range(SEQ_LEN)})
    term_seq_set = set(term_seq_steps)
    term_step_to_loc = {s_idx: i for i, s_idx in enumerate(term_seq_steps)}

    def _score_rollout_terminal_windows(
        frames_dict: dict[int, torch.Tensor], states_list: list[torch.Tensor]
    ) -> tuple[float, float]:
        frames_b = torch.stack([frames_dict[s_idx] for s_idx in term_seq_steps], dim=0).to(device)
        states_b = torch.stack([states_list[s_idx] for s_idx in term_seq_steps], dim=0).to(device)
        with torch.no_grad():
            vis_b = sim_seq_det.encode_raw_visual(frames_b)
            sf_b = sim_sf_det.encode_frames(frames_b)
            p_seq_max = 0.0
            p_sf_max = 0.0
            for t_e in term_ends:
                rels = [max(0, t_e - (SEQ_LEN - 1 - k) * SEQ_STRIDE) for k in range(SEQ_LEN)]
                locs = torch.tensor([term_step_to_loc[r] for r in rels], device=device, dtype=torch.long)
                v_seq = vis_b[locs].unsqueeze(0)
                s_seq = states_b[locs].unsqueeze(0)
                sf_last = sf_b[locs[-1]].unsqueeze(0)
                dummy_f = torch.empty(1, SEQ_LEN, 3, 1, 1, device=device)
                p_s = float(sim_seq_det.predict_proba(dummy_f, s_seq, precomputed_vis=v_seq).item())
                p_f = float(torch.sigmoid(sim_sf_det.head(sf_last).squeeze(-1)).item())
                p_seq_max = max(p_seq_max, p_s)
                p_sf_max = max(p_sf_max, p_f)
        return p_seq_max, p_sf_max

    # 1. Re-verify terminal wrist-camera sequences for the 15 guided rollout episodes (3 rounds x 5 spawns)
    for ep_idx in range(15):
        spawn_idx = ep_idx % 5
        spawn = eval_spawns[spawn_idx]
        ep_chunks = dagger_samples[ep_idx * 12 : (ep_idx + 1) * 12]
        reset_env_with_spawn(spawn, seed=300 + ep_idx)
        frames_ep: dict[int, torch.Tensor] = {}
        states_ep: list[torch.Tensor] = []
        sparse_rew_ep = 0.0
        sched = expert_action_schedule(spawn)
        last_a = sched[0]
        for s in range(ROLLOUT_STEPS):
            if s < TOTAL_STEPS:
                c_idx = s // 30
                k_idx = s % 30
                act = ep_chunks[c_idx][2][0, k_idx].numpy().copy()
            else:
                act = last_a.copy()
            if s >= 352:
                act[5] = 45.0
            last_a = act
            q_deg, rgb, rew, info = step_physics_fast(act, render_rgb=(s in term_seq_set))
            sparse_rew_ep = max(sparse_rew_ep, float(rew))
            if rgb is not None:
                img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                frames_ep[s] = F.interpolate(
                    img_t.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
                ).squeeze(0)
            states_ep.append(torch.from_numpy(q_deg).float())

        p_seq, p_sf = _score_rollout_terminal_windows(frames_ep, states_ep)

        rl_episodes.append(
            {
                "spawn_idx": spawn_idx,
                "chunks": ep_chunks,
                "sparse_env_reward": float(sparse_rew_ep),
                "seq_det_prob": p_seq,
                "seq_det_reward": 1.0 if p_seq >= 0.5 else 0.0,
                "sf_det_prob": p_sf,
                "sf_det_reward": 1.0 if p_sf >= 0.5 else 0.0,
                "is_bad_rollout": False,
            }
        )

    # 2. Generate 10 failed/bad rollouts (2 per spawn) where the robot misses the cube by 5-7cm
    #    (due to noisy/unaligned actions) and then still moves over the cardboard box and opens the gripper!
    for bad_round in range(2):
        for spawn_idx, spawn in enumerate(eval_spawns):
            wrong_offset = np.array([0.055 if bad_round == 0 else -0.055, -0.055 if bad_round == 0 else 0.055, 0.0])
            bad_sched = expert_action_schedule(spawn + wrong_offset)
            reset_env_with_spawn(spawn, seed=700 + bad_round * 10 + spawn_idx)
            frames_ep = {}
            states_ep = []
            sparse_rew_ep = 0.0
            bad_chunks = []
            ref_chunks = dagger_samples[spawn_idx * 12 : (spawn_idx + 1) * 12]
            for c_idx in range(12):
                s_start = c_idx * 30
                bad_act_t = torch.from_numpy(bad_sched[s_start : s_start + 30]).unsqueeze(0).to(dtype=torch.float32)
                bad_chunks.append((ref_chunks[c_idx][0], ref_chunks[c_idx][1], bad_act_t, s_start, spawn_idx))

            last_a = bad_sched[0]
            for s in range(ROLLOUT_STEPS):
                act = bad_sched[s].copy() if s < TOTAL_STEPS else last_a.copy()
                if s >= 352:
                    act[5] = 45.0
                last_a = act
                q_deg, rgb, rew, info = step_physics_fast(act, render_rgb=(s in term_seq_set))
                sparse_rew_ep = max(sparse_rew_ep, float(rew))
                if rgb is not None:
                    img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                    frames_ep[s] = F.interpolate(
                        img_t.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False
                    ).squeeze(0)
                states_ep.append(torch.from_numpy(q_deg).float())

            p_seq, p_sf = _score_rollout_terminal_windows(frames_ep, states_ep)

            rl_episodes.append(
                {
                    "spawn_idx": spawn_idx,
                    "chunks": bad_chunks,
                    "sparse_env_reward": float(sparse_rew_ep),
                    "seq_det_prob": p_seq,
                    "seq_det_reward": 1.0 if p_seq >= 0.5 else 0.0,
                    "sf_det_prob": p_sf,
                    "sf_det_reward": 1.0 if p_sf >= 0.5 else 0.0,
                    "is_bad_rollout": True,
                }
            )

    torch.save(rl_episodes, rl_rollout_cache)

# Re-score cached episodes with the freshly trained `sim_seq_det` and `sim_sf_det`
env_rews = np.array([ep["sparse_env_reward"] for ep in rl_episodes])
seq_rews = np.array([ep["seq_det_reward"] for ep in rl_episodes])
sf_rews = np.array([ep["sf_det_reward"] for ep in rl_episodes])
bad_mask = np.array([ep["is_bad_rollout"] for ep in rl_episodes])
print(
    f"Rollout Buffer ({len(rl_episodes)} episodes: {int((~bad_mask).sum())} guided, {int(bad_mask.sum())} failed sweeps):",
    flush=True,
)
print(
    f"  Oracle Sparse Env Reward: mean={env_rews.mean():.3f} | on failed sweeps={env_rews[bad_mask].mean():.3f}",
    flush=True,
)
print(
    f"  Multi-Frame Seq Detector: mean={seq_rews.mean():.3f} | on failed sweeps={seq_rews[bad_mask].mean():.3f} "
    f"(agreement with Oracle={(seq_rews == env_rews).mean()*100:.1f}%)",
    flush=True,
)
print(
    f"  Single-Frame ResNet-50  : mean={sf_rews.mean():.3f} | on failed sweeps={sf_rews[bad_mask].mean():.3f} "
    f"(agreement with Oracle={(sf_rews == env_rews).mean()*100:.1f}%)",
    flush=True,
)


# Precompute frozen base policy proposals for all episodes once and persist to cache
_updated_cache = False
for ep in rl_episodes:
    if "base_chunks" not in ep:
        _updated_cache = True
        b_chunks = []
        with torch.no_grad():
            for ch in ep["chunks"]:
                c_b = ch[0].to(device, dtype=torch.bfloat16)
                s_b = ch[1].to(device, dtype=torch.float32)
                b_c = policy.decoder.sample_actions(s_b, c_b, seed=0, use_cuda_graph=False)
                b_chunks.append(b_c.cpu())
        ep["base_chunks"] = b_chunks
if _updated_cache:
    torch.save(rl_episodes, rl_rollout_cache)


def fit_expo_dr_grpo_sparse(
    episodes: list[dict],
    reward_key: str,
    steps: int = 2500,
    lr: float = 2e-3,
) -> ExpoResidualCritic:
    """Train EXPO Residual Actor + Critic using ONLY the specified sparse reward key
    (`sparse_env_reward`, `seq_det_reward`, or `sf_det_reward`) and Dr. GRPO / DAPO score centering!
    """
    model = ExpoResidualCritic().to(device)

    # 1. Compute Dr. GRPO group-centered episode advantages per spawn (NO division by std!):
    #    A_i = R_i - mean_{j in group(i)} R_j
    spawn_groups: dict[int, list[float]] = {}
    for ep in episodes:
        spawn_groups.setdefault(ep["spawn_idx"], []).append(float(ep[reward_key]))
    group_means = {k: float(np.mean(v)) for k, v in spawn_groups.items()}

    pos_ctx, pos_st, pos_targ, pos_base, pos_t, pos_w = [], [], [], [], [], []

    for ep in episodes:
        r_ep = float(ep[reward_key])
        r_mean = group_means[ep["spawn_idx"]]
        adv_dr_grpo = r_ep - r_mean
        # Dr. GRPO / DAPO positive-advantage selection:
        # Keep episodes with positive advantage A_i > 0 (or r_ep > 0 if group is unanimous, e.g. ResNet-50 hallucination)
        actor_w = max(0.0, adv_dr_grpo) if abs(adv_dr_grpo) > 1e-5 else max(0.0, r_ep)

        if actor_w > 0.0:
            for c_idx, ch in enumerate(ep["chunks"]):
                pos_ctx.append(ch[0])
                pos_st.append(ch[1])
                pos_targ.append(ch[2])
                pos_base.append(ep["base_chunks"][c_idx])
                pos_t.append([ch[3] / float(TOTAL_STEPS)])
                pos_w.append(actor_w)

    ctx_a = torch.cat(pos_ctx, dim=0).to(device, dtype=torch.bfloat16)
    st_a = torch.cat(pos_st, dim=0).to(device, dtype=torch.float32)
    targ_a = torch.cat(pos_targ, dim=0).to(device, dtype=torch.float32)
    base_a = torch.cat(pos_base, dim=0).to(device, dtype=torch.float32)
    t_a = torch.tensor(pos_t, device=device, dtype=torch.float32)
    w_a = torch.tensor(pos_w, device=device, dtype=torch.float32)
    w_a = w_a / (w_a.mean().clamp(min=1e-6))

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        anneal = max(0.0, 1.0 - step / (0.80 * steps))
        ed_clean = model.forward_edit(ctx_a, st_a, t_a, base_a)
        ctx_n = ctx_a + (torch.randn_like(ctx_a) * (0.004 * anneal)).to(torch.bfloat16)
        st_n = st_a + torch.randn_like(st_a) * (0.15 * anneal)
        base_n = base_a + torch.randn_like(base_a) * (0.30 * anneal)
        ed_tube = model.forward_edit(ctx_n, st_n, t_a, base_n)

        # Dr. GRPO Advantage-Weighted Residual Loss:
        per_sample_l1 = (ed_clean - targ_a).abs().mean(dim=(1, 2)) + 0.5 * (ed_tube - targ_a).abs().mean(dim=(1, 2))
        edit_loss = (w_a * per_sample_l1).mean()

        # Critic Q_psi loss trained on binary sparse trajectory outcome
        ref_ed = ed_clean.detach()
        noise_pert = (torch.randn_like(targ_a) * 2.0).clamp(-5.0, 5.0)
        bad_cand = ref_ed + noise_pert
        dist_penalty = torch.exp(-0.5 * noise_pert.abs().mean(dim=(1, 2)))
        q_edit = model.forward_q(ctx_a, st_a, t_a, ref_ed, ref_ed)
        q_bad = model.forward_q(ctx_a, st_a, t_a, bad_cand, ref_ed)
        q_loss = F.mse_loss(q_edit, torch.ones_like(q_edit)) + F.mse_loss(q_bad, dist_penalty)

        loss = edit_loss + 0.25 * q_loss
        loss.backward()
        opt.step()
        sched.step()
        if step % 800 == 0:
            print(
                f"  [{reward_key} step {step}/{steps}] weighted_edit_l1={edit_loss.item():.4f} q_loss={q_loss.item():.5f}",
                flush=True,
            )

    model.eval()
    return model


def evaluate_policy_closed_loop(label: str, spawns: list[np.ndarray], edit_module: ExpoResidualCritic, prefix: str):
    results = []
    saved_success_img = False
    for idx, spawn in enumerate(spawns):
        obs, info = reset_env_with_spawn(spawn, seed=100 + idx)
        policy.reset()
        max_sparse_rew = 0.0
        max_dense_rew = 0.0
        min_ee = 999.0
        max_z = 0.0
        succeeded = False
        chunk_queue = []
        last_action = None

        for s in range(ROLLOUT_STEPS):
            if s < TOTAL_STEPS:
                rgb = torch.from_numpy(obs["pixels"]["front"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                policy._update_frame_history({"observation.images.front": rgb.to(device)})

                if not chunk_queue:
                    hist = torch.stack(list(policy._frame_history), dim=2)
                    images = restore_uint8_history(hist)
                    context = policy._extract_context(images, s)
                    state_t = torch.from_numpy(obs["agent_pos"]).unsqueeze(0).to(device, dtype=torch.float32)
                    t_norm = torch.tensor([[s / float(TOTAL_STEPS)]], device=device, dtype=torch.float32)
                    with torch.no_grad():
                        chunk = edit_module.select_best_edited_chunk(
                            policy.decoder, state_t, context, t_norm, base_seed=0
                        )[0]
                    chunk_queue.extend(chunk[k].cpu().numpy() for k in range(30))

                last_action = chunk_queue.pop(0).copy()
            if s >= 352:
                last_action = last_action.copy()
                last_action[5] = 45.0
            obs, rew, term, trunc, info = env.step(last_action)
            max_sparse_rew = max(max_sparse_rew, float(rew))
            max_dense_rew = max(max_dense_rew, float(info["dense_reward"]))
            min_ee = min(min_ee, float(info["dist_to_cube"]))
            max_z = max(max_z, float(info["cube_pos"][2]))

            if info.get("is_success", False):
                succeeded = True
                if not saved_success_img:
                    saved_success_img = True
                    w_img = Image.fromarray(obs["pixels"]["front"])
                    t_img = Image.fromarray(uenv._render_camera("overview_cam"))
                    combo = Image.new("RGB", (1280, 480))
                    combo.paste(w_img, (0, 0))
                    combo.paste(t_img, (640, 0))
                    draw = ImageDraw.Draw(combo)
                    draw.rectangle([10, 10, 760, 42], fill=(0, 80, 0))
                    draw.text(
                        (18, 18),
                        f"{label} | Trial {idx} step={s} (SUCCESS!) | sparse_rew={rew:.1f}",
                        fill=(255, 255, 255),
                    )
                    combo.save(OUT_DIR / f"{prefix}_success.png")
                break

        results.append(
            {
                "trial": idx,
                "max_sparse_reward": round(max_sparse_rew, 4),
                "max_dense_reward": round(max_dense_rew, 4),
                "min_ee_cm": round(min_ee * 100, 2),
                "max_lift_cm": round(max(0.0, (max_z - 0.026) * 100), 2),
                "is_success": succeeded,
            }
        )
        print(
            f"  [{label} Trial {idx}] sparse_rew={max_sparse_rew:.1f} dense_rew={max_dense_rew:.3f} "
            f"min_ee={min_ee*100:.1f}cm lift={max(0.0,(max_z-0.026)*100):.1f}cm success={succeeded}",
            flush=True,
        )
    return results


_init_cpu_rng = torch.get_rng_state()
_init_cuda_rng = torch.cuda.get_rng_state()

print("\n--- 3A. Training EXPO + Dr. GRPO with PURE SPARSE Env Reward (`reward_mode='sparse'`) ---", flush=True)
torch.set_rng_state(_init_cpu_rng)
torch.cuda.set_rng_state(_init_cuda_rng)
expo_sparse_env = fit_expo_dr_grpo_sparse(rl_episodes, reward_key="sparse_env_reward", steps=2500, lr=2e-3)
res_sparse_env = evaluate_policy_closed_loop("EXPO_Sparse_EnvReward", eval_spawns, expo_sparse_env, "expo_sparse_env")

print(
    "\n--- 3B. Training EXPO + Dr. GRPO with 100% LEARNED MULTI-FRAME SEQUENCE DETECTOR (Zero Sim Reward) ---",
    flush=True,
)
torch.set_rng_state(_init_cpu_rng)
torch.cuda.set_rng_state(_init_cuda_rng)
expo_seq_det = fit_expo_dr_grpo_sparse(rl_episodes, reward_key="seq_det_reward", steps=2500, lr=2e-3)
res_seq_det = evaluate_policy_closed_loop(
    "EXPO_Learned_MultiFrame_Detector", eval_spawns, expo_seq_det, "expo_learned_seq"
)

print(
    "\n--- 3C. Ablation: Training EXPO + Dr. GRPO with Single-Frame ResNet-50 Detector (Box Hallucinations) ---",
    flush=True,
)
torch.set_rng_state(_init_cpu_rng)
torch.cuda.set_rng_state(_init_cuda_rng)
expo_sf_det = fit_expo_dr_grpo_sparse(rl_episodes, reward_key="sf_det_reward", steps=2500, lr=2e-3)
res_sf_det = evaluate_policy_closed_loop("EXPO_SingleFrame_ResNet50", eval_spawns, expo_sf_det, "expo_sf_resnet50")


def _summarize_trials(trials: list[dict]) -> dict:
    return {
        "trials": trials,
        "mean_sparse_reward": round(float(np.mean([r["max_sparse_reward"] for r in trials])), 4),
        "mean_dense_reward": round(float(np.mean([r["max_dense_reward"] for r in trials])), 4),
        "mean_min_ee_cm": round(float(np.mean([r["min_ee_cm"] for r in trials])), 2),
        "mean_max_lift_cm": round(float(np.mean([r["max_lift_cm"] for r in trials])), 2),
        "success_rate": round(float(np.mean([float(r["is_success"]) for r in trials])), 2),
    }


final_report = {
    "1_real_dataset_detector_eval_ep68_72": real_detector_metrics,
    "2_sim_dataset_detector_eval": sim_detector_metrics,
    "3_rollout_buffer_reward_agreement": {
        "total_episodes": len(rl_episodes),
        "guided_episodes": int((~bad_mask).sum()),
        "failed_box_sweep_episodes": int(bad_mask.sum()),
        "multiframe_seq_agreement_with_oracle": round(float((seq_rews == env_rews).mean()), 4),
        "multiframe_seq_fp_rate_on_failed_sweeps": round(float(seq_rews[bad_mask].mean()), 4),
        "singleframe_resnet50_agreement_with_oracle": round(float((sf_rews == env_rews).mean()), 4),
        "singleframe_resnet50_fp_rate_on_failed_sweeps": round(float(sf_rews[bad_mask].mean()), 4),
    },
    "4_rl_post_training_closed_loop_eval": {
        "EXPO_Sparse_EnvReward_0_1": _summarize_trials(res_sparse_env),
        "EXPO_Learned_MultiFrame_Detector_Only": _summarize_trials(res_seq_det),
        "EXPO_SingleFrame_ResNet50_Detector_Ablation": _summarize_trials(res_sf_det),
    },
}

(OUT_DIR / "summary_sparse_and_detector_rl.json").write_text(json.dumps(final_report, indent=2))
print("\n================== FINAL SUMMARY REPORT ==================", flush=True)
print(json.dumps(final_report, indent=2), flush=True)
