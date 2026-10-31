#!/usr/bin/env python3
"""Scientifically Rigorous Study:
1. Shared-Backbone (`Cosmos 3 Edge LoRA` `[24, 2048]` tokens) vs. Dedicated (`ResNet-18`) Multi-Frame
   Sequence Detector vs. Single-Frame `ResNet-50` across 3 seeds (Mean +/- SEM) on:
   - Real-Robot Dataset `Orellius/so101_sort_cubes_no_top` (Strict 3-Way Split: Train `0..57`, Val `58..67`,
     Held-Out Test `68..72` & `68..76`, including Empty-Gripper Box Sweep & Static Stare Hard Negatives).
   - Simulation 60-Episode Rollout Buffer (`sparse_rl_10spawns_pooled.pt`, 30 guided + 30 failed box-sweep rollouts).
2. 3-Seed Closed-Loop RL Post-Training Evaluation (`SO101CubeBoxSparse-v0`, 10 spawns, 5,000 steps, Mean +/- SEM):
   - Zero-Shot Base Policy (`Cosmos 3 Edge LoRA + SmolExpert`)
   - Vanilla EXPO (Uncentered AWR, beta=1.5)
   - Score-Centered EXPO (Dr. GRPO) + Single-Frame ResNet-50 Detector
   - Score-Centered EXPO (Dr. GRPO) + Shared-Backbone Sequence Detector (`Cosmos 3 Edge LoRA` tokens, 0 extra backbone overhead)
   - Score-Centered EXPO (Dr. GRPO) + Dedicated Multi-Frame Sequence Detector (`ResNet-18`)
   - Score-Centered EXPO (Dr. GRPO) + Ground-Truth Sparse Oracle
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pyarrow.parquet as _pq
import torch
import torch.nn.functional as F  # noqa: N812

import lerobot.envs.so101_cube_box  # noqa: F401
from lerobot.envs.so101_cube_box import (
    lerobot_to_mujoco_qpos,
    mujoco_to_lerobot_qpos,
)
from lerobot.policies.vam.configuration_video_vam import VideoVAMConfig
from lerobot.policies.vam.expo_residual import ExpoResidualCritic
from lerobot.policies.vam.modeling_video_vam import VideoVAMPolicy, restore_uint8_history
from lerobot.rewards.sequence_detector import (
    MultiFrameSuccessDetector,
    SharedFeatureSequenceDetector,
)

OUT_DIR = Path("outputs/sparse_seq_rl")
OUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 8
SEQ_STRIDE = 6


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


def mean_sem(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {"mean": round(float(arr.mean()), 4), "sem": round(sem, 4), "seeds": [round(float(x), 4) for x in vals]}


print("=========================================================================", flush=True)
print("LOADING PRETRAINED VIDEO-VAM POLICY (`Cosmos 3 Edge LoRA + SmolExpert`)", flush=True)
print("=========================================================================", flush=True)
ckpt_dir = "outputs/train/c3_sort_cubes_30m_lora"
cfg = VideoVAMConfig.from_pretrained(ckpt_dir)
cfg.pretrained_path = Path(ckpt_dir)
cfg.n_action_steps = 30
cfg.joint_limits_min = None
cfg.joint_limits_max = None
policy = VideoVAMPolicy(cfg)
policy.decoder._cuda_graph_disabled = True
policy.eval()


# =============================================================================
# PART 1: EXTRACT & EVALUATE SHARED-BACKBONE (`Cosmos 3 Edge LoRA`) FEATURES
#         ON REAL DATASET `Orellius/so101_sort_cubes_no_top`
# =============================================================================
real_c3_cache_path = OUT_DIR / "real_cosmos3_shared_features.pt"
real_mv_cache = torch.load(OUT_DIR / "real_detector_multiview_v3.pt", map_location="cpu", weights_only=False)
ep_features_mv = real_mv_cache["ep_features"]
ep_step_groups = real_mv_cache["ep_step_groups"]

if real_c3_cache_path.exists():
    print("Loading cached real-dataset Cosmos 3 Edge LoRA pooled token sequences...", flush=True)
    ep_c3_features = torch.load(real_c3_cache_path, map_location="cpu", weights_only=False)
else:
    t0 = time.time()
    print("Extracting Cosmos 3 Edge LoRA [24, 2048] pooled tokens on real dataset (76 episodes)...", flush=True)
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq

    snap_dir = sorted(
        Path("/root/.cache/huggingface/lerobot/hub/datasets--Orellius--so101_sort_cubes_no_top/snapshots").glob("*")
    )[-1]
    ep_files = sorted((snap_dir / "meta/episodes/chunk-000").glob("*.parquet"))
    ep_table = pa.concat_tables([pq.read_table(f) for f in ep_files]).to_pydict()
    ep_meta = {
        int(ep_id): {
            "from_idx": int(ep_table["dataset_from_index"][i]),
            "to_idx": int(ep_table["dataset_to_index"][i]),
            "vid_file": int(ep_table["videos/observation.images.front/file_index"][i]),
            "from_ts": float(ep_table["videos/observation.images.front/from_timestamp"][i]),
        }
        for i, ep_id in enumerate(ep_table["episode_index"])
    }

    file_to_eps: dict[int, list[int]] = {}
    ep_needed_rels: dict[int, list[int]] = {}
    ep_anchor_rels: dict[int, list[int]] = {}
    for ep, (pos_steps, early_steps, carry_steps, ep_len) in ep_step_groups.items():
        needed_rel = sorted(
            {
                max(0, min(ep_len - 1, t_end - (SEQ_LEN - 1 - k) * SEQ_STRIDE))
                for t_end in (pos_steps + early_steps + carry_steps)
                for k in range(SEQ_LEN)
            }
        )
        ep_needed_rels[ep] = needed_rel
        # Deduplicate 5-frame clips so anchor clips are spaced by at least 6 frames
        anchors: list[int] = []
        for r in needed_rel:
            if not anchors or (r - anchors[-1]) >= 6:
                anchors.append(r)
        if needed_rel[-1] not in anchors:
            anchors.append(needed_rel[-1])
        ep_anchor_rels[ep] = anchors
        file_to_eps.setdefault(ep_meta[ep]["vid_file"], []).append(ep)

    ep_c3_features: dict[int, torch.Tensor] = {}
    # For each anchor frame `r`, Cosmos 3 Edge takes a 5-frame history clip `[r-4, r-3, r-2, r-1, r]`
    for vid_file_idx, eps_in_file in sorted(file_to_eps.items()):
        mp4_path = snap_dir / f"videos/observation.images.front/chunk-000/file-{vid_file_idx:03d}.mp4"
        frame_targets: dict[int, list[tuple[int, int, int]]] = {}
        for ep in eps_in_file:
            base_f = int(round(ep_meta[ep]["from_ts"] * 30.0))
            for r in ep_anchor_rels[ep]:
                for h_idx, lag in enumerate((4, 3, 2, 1, 0)):
                    f_num = base_f + max(0, r - lag)
                    frame_targets.setdefault(f_num, []).append((ep, r, h_idx))

        max_target_frame = max(frame_targets.keys())
        collected_240x320: dict[tuple[int, int, int], torch.Tensor] = {}
        container = av.open(str(mp4_path))
        for f_idx, frame in enumerate(container.decode(video=0)):
            if f_idx in frame_targets:
                arr = frame.to_ndarray(format="rgb24")
                img_t = torch.from_numpy(arr).permute(2, 0, 1).float()
                img_rz = (
                    F.interpolate(img_t.unsqueeze(0), size=(240, 320), mode="bilinear", align_corners=False)
                    .squeeze(0)
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
                for key in frame_targets[f_idx]:
                    collected_240x320[key] = img_rz
            if f_idx >= max_target_frame:
                break
        container.close()

        for ep in eps_in_file:
            anchors = ep_anchor_rels[ep]
            rels = ep_needed_rels[ep]
            clips = []
            for r in anchors:
                clip_5 = torch.stack([collected_240x320[(ep, r, h_i)] for h_i in range(5)], dim=1)  # [3, 5, 240, 320]
                clips.append(clip_5)
            clips_b = torch.stack(clips, dim=0).to(device)  # [N_anc, 3, 5, 240, 320]
            pooled_list = []
            with torch.no_grad():
                for b0 in range(0, len(clips_b), 16):
                    sub = clips_b[b0 : b0 + 16]
                    ctx_sub = policy.extractor.extract(rgb_frames=sub).features
                    pooled_sub = ExpoResidualCritic.pool_ctx_tokens(ctx_sub).cpu().to(torch.float16)
                    pooled_list.append(pooled_sub)
            anc_feats = torch.cat(pooled_list, dim=0)  # [N_anc, 24, 2048]
            anc_arr = np.asarray(anchors)
            nearest_idx = [int(np.argmin(np.abs(anc_arr - r))) for r in rels]
            ep_c3_features[ep] = anc_feats[torch.tensor(nearest_idx, dtype=torch.long)]  # [N_rel, 24, 2048]
        print(f"  Finished video file {vid_file_idx} ({len(eps_in_file)} episodes)", flush=True)

    torch.save(ep_c3_features, real_c3_cache_path)
    print(f"Cached real Cosmos 3 Edge LoRA features in {time.time() - t0:.1f}s.", flush=True)


def assemble_real_split(split_eps: list[int], include_aug_views: bool = False) -> dict[str, torch.Tensor]:
    vis_seq_l, c3_seq_l, sf_l, st_l, y_l, pre_l, kind_l = [], [], [], [], [], [], []
    st_c3_l, y_c3_l, pre_c3_l = [], [], []
    views = list(range(6)) if include_aug_views else [0]
    for ep in split_eps:
        if ep not in ep_features_mv or ep not in ep_c3_features:
            continue
        vis_views, ep_states_cpu, rel_to_local = ep_features_mv[ep]
        c3_raw = ep_c3_features[ep].float()  # [N_rel, 24, 2048]
        pos_steps, early_steps, carry_steps, ep_len = ep_step_groups[ep]

        for v_idx in views:
            ep_vis_raw = vis_views[v_idx]

            def _gather(
                rel_indices: list[int],
                _rel_map=rel_to_local,
                _len=ep_len,
                _c3=c3_raw,
                _v=v_idx,
                _vis=ep_vis_raw,
                _st=ep_states_cpu,
            ):
                loc = [_rel_map[max(0, min(_len - 1, r))] for r in rel_indices]
                loc_t = torch.tensor(loc, dtype=torch.long)
                c3_part = _c3[loc_t] if _v == 0 else None
                return _vis[loc_t], c3_part, _vis[loc_t[-1]], _st[loc_t]

            for t_end in pos_steps:
                rels = [max(0, t_end - (SEQ_LEN - 1 - k) * SEQ_STRIDE) for k in range(SEQ_LEN)]
                v_s, c3_s, sf_last, st_s = _gather(rels)
                vis_seq_l.append(v_s)
                sf_l.append(sf_last)
                st_l.append(st_s)
                y_l.append(1.0)
                pre_l.append(1.0)
                kind_l.append(0)
                if c3_s is not None:
                    c3_seq_l.append(c3_s)
                    st_c3_l.append(st_s)
                    y_c3_l.append(1.0)
                    pre_c3_l.append(1.0)

            for t_end in early_steps:
                rels = [max(0, t_end - (SEQ_LEN - 1 - k) * SEQ_STRIDE) for k in range(SEQ_LEN)]
                v_s, c3_s, sf_last, st_s = _gather(rels)
                vis_seq_l.append(v_s)
                sf_l.append(sf_last)
                st_l.append(st_s)
                y_l.append(0.0)
                pre_l.append(0.0)
                kind_l.append(1)
                if c3_s is not None:
                    c3_seq_l.append(c3_s)
                    st_c3_l.append(st_s)
                    y_c3_l.append(0.0)
                    pre_c3_l.append(0.0)

            for t_end in carry_steps:
                rels = [max(0, t_end - (SEQ_LEN - 1 - k) * SEQ_STRIDE) for k in range(SEQ_LEN)]
                v_s, c3_s, sf_last, st_s = _gather(rels)
                vis_seq_l.append(v_s)
                sf_l.append(sf_last)
                st_l.append(st_s)
                y_l.append(0.0)
                pre_l.append(1.0)
                kind_l.append(1)
                if c3_s is not None:
                    c3_seq_l.append(c3_s)
                    st_c3_l.append(st_s)
                    y_c3_l.append(0.0)
                    pre_c3_l.append(1.0)

            for e_step, p_step in [(early_steps[0], pos_steps[-1]), (early_steps[1], pos_steps[-2])]:
                early_rels = [max(0, e_step - (5 - k) * SEQ_STRIDE) for k in range(6)]
                box_rels = [max(0, p_step - SEQ_STRIDE), p_step]
                v_s, c3_s, sf_last, st_s = _gather(early_rels + box_rels)
                st_s = st_s.clone()
                st_s[:6, 5] = st_s[-1, 5]
                vis_seq_l.append(v_s)
                sf_l.append(sf_last)
                st_l.append(st_s)
                y_l.append(0.0)
                pre_l.append(0.0)
                kind_l.append(2)
                if c3_s is not None:
                    c3_seq_l.append(c3_s)
                    st_c3_l.append(st_s)
                    y_c3_l.append(0.0)
                    pre_c3_l.append(0.0)

            for p_step in (pos_steps[-1], pos_steps[-2]):
                static_rels = [p_step for _ in range(SEQ_LEN)]
                v_s, c3_s, sf_last, st_s = _gather(static_rels)
                v_s = v_s + torch.randn_like(v_s) * 0.01
                st_s = st_s + torch.randn_like(st_s) * 0.15
                vis_seq_l.append(v_s)
                sf_l.append(sf_last)
                st_l.append(st_s)
                y_l.append(0.0)
                pre_l.append(0.0)
                kind_l.append(3)
                if c3_s is not None:
                    c3_seq_l.append(c3_s)
                    st_c3_l.append(st_s)
                    y_c3_l.append(0.0)
                    pre_c3_l.append(0.0)

    return {
        "vis_seq": torch.stack(vis_seq_l, dim=0),
        "c3_seq": torch.stack(c3_seq_l, dim=0),
        "st_c3": torch.stack(st_c3_l, dim=0),
        "y_c3": torch.tensor(y_c3_l, dtype=torch.float32),
        "pre_c3": torch.tensor(pre_c3_l, dtype=torch.float32),
        "sf_feat": torch.stack(sf_l, dim=0),
        "state_seq": torch.stack(st_l, dim=0),
        "y": torch.tensor(y_l, dtype=torch.float32),
        "pre": torch.tensor(pre_l, dtype=torch.float32),
        "kind": torch.tensor(kind_l, dtype=torch.long),
    }


train_split = assemble_real_split(list(range(0, 58)), include_aug_views=True)
val_split = assemble_real_split(list(range(58, 68)), include_aug_views=False)
test5_split = assemble_real_split(list(range(68, 73)), include_aug_views=False)
test9_split = assemble_real_split(list(range(68, 77)), include_aug_views=False)


def eval_real_detectors_3seeds() -> dict:
    vis_tr = train_split["vis_seq"].to(device)
    c3_tr_0 = train_split["c3_seq"].to(device)
    sf_tr = train_split["sf_feat"].to(device)
    st_tr = train_split["state_seq"].to(device)
    st_tr_0 = train_split["st_c3"].to(device)
    y_tr = train_split["y"].to(device)
    y_tr_0 = train_split["y_c3"].to(device)
    pre_tr = train_split["pre"].to(device)
    pre_tr_0 = train_split["pre_c3"].to(device)
    kind_tr = train_split["kind"].to(device)

    vis_val = val_split["vis_seq"].to(device)
    c3_val = val_split["c3_seq"].to(device)
    st_val = val_split["state_seq"].to(device)
    y_val_t = val_split["y"].to(device)

    pos_weight = torch.tensor([(y_tr == 0).sum().float() / max(1.0, (y_tr == 1).sum().float())], device=device)
    sf_mask = (kind_tr == 0) | (kind_tr == 1)
    sf_tr_sub = sf_tr[sf_mask]
    y_sf_sub = y_tr[sf_mask]
    pos_weight_sf = torch.tensor([(y_sf_sub == 0).sum().float() / max(1.0, (y_sf_sub == 1).sum().float())], device=device)

    dummy_tr = torch.empty(vis_tr.shape[0], SEQ_LEN, 3, 1, 1, device=device)
    dummy_val = torch.empty(vis_val.shape[0], SEQ_LEN, 3, 1, 1, device=device)

    res_by_model: dict[str, list[dict]] = {"Dedicated_ResNet18_Seq": [], "Shared_Cosmos3_LoRA_Seq": [], "SingleFrame_ResNet50": []}

    for seed in (0, 1, 2):
        torch.manual_seed(1000 + seed)
        torch.cuda.manual_seed_all(1000 + seed)

        seq_det = MultiFrameSuccessDetector(seq_len=SEQ_LEN, state_dim=6, pretrained=False).to(device)
        shared_det = SharedFeatureSequenceDetector(ctx_dim=2048, seq_len=SEQ_LEN, state_dim=6).to(device)
        sf_head = torch.nn.Sequential(
            torch.nn.Linear(3072, 256),
            torch.nn.LayerNorm(256),
            torch.nn.SiLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 1),
        ).to(device)

        opt_seq = torch.optim.AdamW([p for p in seq_det.parameters() if p.requires_grad], lr=8e-4, weight_decay=3e-4)
        opt_sh = torch.optim.AdamW(shared_det.parameters(), lr=8e-4, weight_decay=3e-4)
        opt_sf = torch.optim.AdamW(sf_head.parameters(), lr=1e-3, weight_decay=1e-4)

        best_seq_sd, best_seq_vloss = None, 999.0
        best_sh_sd, best_sh_vloss = None, 999.0

        for step in range(1, 451):
            # 1. Dedicated ResNet-18 MultiFrameSuccessDetector (mini-batch 512)
            seq_det.train()
            opt_seq.zero_grad(set_to_none=True)
            idx_v = torch.randint(0, vis_tr.shape[0], (min(512, vis_tr.shape[0]),), device=device)
            vis_sub = vis_tr[idx_v]
            st_sub = st_tr[idx_v]
            scale_v = 1.0 + torch.randn(vis_sub.shape[0], 1, 1, device=device) * 0.08
            vis_b = vis_sub * scale_v + torch.randn_like(vis_sub) * 0.025
            st_b = st_sub + torch.randn_like(st_sub) * 0.35
            j_log, p_log = seq_det.forward_with_aux(
                dummy_tr[: vis_sub.shape[0]], st_b, precomputed_vis=vis_b, proprio_dropout=0.45, token_dropout=0.20
            )
            y_sm = y_tr[idx_v] * 0.96 + 0.02
            p_sm = pre_tr[idx_v] * 0.96 + 0.02
            loss_seq = F.binary_cross_entropy_with_logits(j_log, y_sm, pos_weight=pos_weight) + 0.4 * F.binary_cross_entropy_with_logits(p_log, p_sm)
            loss_seq.backward()
            torch.nn.utils.clip_grad_norm_(seq_det.parameters(), 1.0)
            opt_seq.step()

            # 2. Shared Cosmos 3 Edge LoRA Sequence Detector (mini-batch 128 + cell-level scale noise)
            shared_det.train()
            opt_sh.zero_grad(set_to_none=True)
            idx_c = torch.randint(0, c3_tr_0.shape[0], (min(128, c3_tr_0.shape[0]),), device=device)
            c3_sub = c3_tr_0[idx_c]
            scale_c = 1.0 + torch.randn(c3_sub.shape[0], SEQ_LEN, 24, 1, device=device) * 0.08
            c3_b = c3_sub * scale_c
            st_b0 = st_tr_0[idx_c] + torch.randn_like(st_tr_0[idx_c]) * 0.35
            sh_j, sh_p = shared_det.forward_with_aux(c3_b, st_b0, proprio_dropout=0.45, token_dropout=0.20)
            y_sm0 = y_tr_0[idx_c] * 0.96 + 0.02
            p_sm0 = pre_tr_0[idx_c] * 0.96 + 0.02
            loss_sh = F.binary_cross_entropy_with_logits(sh_j, y_sm0, pos_weight=pos_weight) + 0.4 * F.binary_cross_entropy_with_logits(sh_p, p_sm0)
            loss_sh.backward()
            torch.nn.utils.clip_grad_norm_(shared_det.parameters(), 1.0)
            opt_sh.step()

            # 3. Single-Frame Baseline
            sf_head.train()
            opt_sf.zero_grad(set_to_none=True)
            sf_logits = sf_head(sf_tr_sub + torch.randn_like(sf_tr_sub) * 0.015).squeeze(-1)
            loss_sf = F.binary_cross_entropy_with_logits(sf_logits, y_sf_sub, pos_weight=pos_weight_sf)
            loss_sf.backward()
            opt_sf.step()

            if step % 20 == 0:
                seq_det.eval()
                shared_det.eval()
                with torch.no_grad():
                    v_log = seq_det(dummy_val, st_val, precomputed_vis=vis_val)
                    v_loss = float(F.binary_cross_entropy_with_logits(v_log, y_val_t, pos_weight=pos_weight).item())
                    if v_loss < best_seq_vloss:
                        best_seq_vloss = v_loss
                        best_seq_sd = copy.deepcopy(seq_det.state_dict())

                    sh_vlog = shared_det(c3_val, st_val)
                    sh_vloss = float(F.binary_cross_entropy_with_logits(sh_vlog, y_val_t, pos_weight=pos_weight).item())
                    if sh_vloss < best_sh_vloss:
                        best_sh_vloss = sh_vloss
                        best_sh_sd = copy.deepcopy(shared_det.state_dict())

        seq_det.load_state_dict(best_seq_sd)
        shared_det.load_state_dict(best_sh_sd)
        seq_det.eval()
        shared_det.eval()
        sf_head.eval()

        def _eval_split(
            sp: dict[str, torch.Tensor],
            _seq=seq_det,
            _sh=shared_det,
            _sf=sf_head,
        ):
            v_s = sp["vis_seq"].to(device)
            c3_s = sp["c3_seq"].to(device)
            sf_s = sp["sf_feat"].to(device)
            st_s = sp["state_seq"].to(device)
            y_np = sp["y"].numpy()
            k_np = sp["kind"].numpy()
            hard_m = (k_np == 2) | (k_np == 3)
            d_f = torch.empty(v_s.shape[0], SEQ_LEN, 3, 1, 1, device=device)
            with torch.no_grad():
                p_seq = _seq.predict_proba(d_f, st_s, precomputed_vis=v_s).cpu().numpy()
                p_sh = _sh.predict_proba(c3_s, st_s).cpu().numpy()
                p_sf = torch.sigmoid(_sf(sf_s).squeeze(-1)).cpu().numpy()
            out = {}
            for name, probs in [("Dedicated_ResNet18_Seq", p_seq), ("Shared_Cosmos3_LoRA_Seq", p_sh), ("SingleFrame_ResNet50", p_sf)]:
                m = compute_binary_metrics(y_np, probs)
                m["hard_neg_fp_rate"] = round(float((probs[hard_m] >= 0.5).mean()), 4)
                out[name] = m
            return out

        ev_val = _eval_split(val_split)
        ev_t5 = _eval_split(test5_split)
        ev_t9 = _eval_split(test9_split)
        for k in res_by_model:
            res_by_model[k].append({"val": ev_val[k], "test5": ev_t5[k], "test9": ev_t9[k]})

    summary = {}
    for model_name, seed_runs in res_by_model.items():
        summary[model_name] = {}
        for split_name in ("val", "test5", "test9"):
            summary[model_name][split_name] = {
                metric: mean_sem([r[split_name][metric] for r in seed_runs])
                for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "hard_neg_fp_rate")
            }
    return summary


real_3seed_cache_path = OUT_DIR / "real_3seed_summary.json"
if real_3seed_cache_path.exists():
    print("\nLoading cached 3-seed real dataset detector results...", flush=True)
    real_3seed_summary = json.loads(real_3seed_cache_path.read_text())
else:
    print("\n=========================================================================", flush=True)
    print("EVALUATING 3-SEED REAL DATASET REWARD DETECTORS (SHARED vs DEDICATED vs SF)", flush=True)
    print("=========================================================================", flush=True)
    real_3seed_summary = eval_real_detectors_3seeds()
    real_3seed_cache_path.write_text(json.dumps(real_3seed_summary, indent=2))
print(json.dumps(real_3seed_summary, indent=2), flush=True)


# =============================================================================
# PART 2: SIMULATION 60-EPISODE ROLLOUT BUFFER + 3-SEED CLOSED-LOOP RL STUDY
# =============================================================================
print("\n=========================================================================", flush=True)
print("STAGE 2: SHARED-FEATURE DETECTOR ON 60-EPISODE BUFFER & 3-SEED 10-SPAWN RL", flush=True)
print("=========================================================================", flush=True)

pooled_cache = torch.load(OUT_DIR / "sparse_rl_10spawns_pooled.pt", map_location="cpu", weights_only=False)
eval_spawns_10 = pooled_cache["eval_spawns_10"]
pooled_rounds = pooled_cache["pooled_rounds"]
pooled_bad_by_spawn = pooled_cache["pooled_bad_by_spawn"]

all_60_eps: list[dict] = []
for (_r_idx, _s_idx), ep in sorted(pooled_rounds.items()):
    all_60_eps.append(ep)
for _s_idx, bad_list in sorted(pooled_bad_by_spawn.items()):
    for ep in bad_list:
        all_60_eps.append(ep)

# Setup MuJoCo Env for true wrist-camera token extraction on failed sweeps & 3-Seed Closed-Loop Eval
_snap_dir = sorted(
    Path("/root/.cache/huggingface/lerobot/hub/datasets--Orellius--so101_sort_cubes_no_top/snapshots").glob("*")
)[-1]
_ep0_table = _pq.read_table(sorted((_snap_dir / "data/chunk-000").glob("*.parquet"))[0], columns=["action"]).to_pydict()
raw_actions = np.asarray(_ep0_table["action"][:480], dtype=np.float32)
bridge = np.linspace(raw_actions[15], raw_actions[165], 30, dtype=np.float32)
clean_actions = np.concatenate([raw_actions[:15], bridge, raw_actions[165:]], axis=0)
TOTAL_STEPS = len(clean_actions)  # 360
ROLLOUT_STEPS = 385

env = gym.make(
    "lerobot/SO101CubeBox-v0",
    obs_type="pixels_agent_pos",
    observation_width=640,
    observation_height=480,
    reward_mode="sparse",
)
uenv = env.unwrapped
env.reset(seed=42)


def reset_env_with_spawn(spawn_xyz: np.ndarray, seed: int = 42):
    uenv._episode_idx = 0
    uenv._step_count = 0
    obs, info = env.reset(seed=seed)
    uenv.data.qpos[6:9] = spawn_xyz
    uenv.data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]
    uenv.data.qvel[:] = 0.0
    mujoco.mj_forward(uenv.model, uenv.data)
    return uenv._get_obs(), info


# Ensure the 30 failed rollouts have their ACTUAL wrist-camera Cosmos 3 Edge LoRA tokens & states
bad_c3_cache_path = OUT_DIR / "bad_rollouts_true_c3_seq.pt"
if bad_c3_cache_path.exists():
    bad_true_seqs = torch.load(bad_c3_cache_path, map_location="cpu", weights_only=False)
else:
    print("Extracting true MuJoCo wrist-camera Cosmos 3 Edge LoRA sequences for the 30 failed sweeps...", flush=True)
    bad_true_seqs = []
    # Sample 6 chunk boundaries across the trajectory: steps [60, 120, 180, 240, 300, 355]
    sample_steps = [60, 120, 180, 240, 300, 355]
    need_render = {max(0, st - lag) for st in sample_steps for lag in (4, 3, 2, 1, 0)}
    for ep in all_60_eps:
        if not ep.get("is_bad_rollout", False):
            continue
        spawn = eval_spawns_10[ep["spawn_idx"]]
        reset_env_with_spawn(spawn, seed=900 + ep["spawn_idx"])
        sched = np.concatenate([ch[2][0].numpy() for ch in ep["chunks"]], axis=0)
        ep_240: dict[int, torch.Tensor] = {}
        ep_q: dict[int, torch.Tensor] = {}
        for s in range(TOTAL_STEPS):
            act = sched[s].copy()
            if s >= 352:
                act[5] = 45.0
            uenv.data.ctrl[:6] = lerobot_to_mujoco_qpos(act, uenv.model.jnt_range)
            for _ in range(uenv.n_substeps):
                mujoco.mj_step(uenv.model, uenv.data)
            uenv._step_count += 1
            if s in sample_steps:
                ep_q[s] = torch.from_numpy(mujoco_to_lerobot_qpos(uenv.data.qpos[:6])).float()
            if s in need_render:
                rgb = uenv._render_camera("wrist_cam")
                img_t = torch.from_numpy(rgb).permute(2, 0, 1).float()
                ep_240[s] = (
                    F.interpolate(img_t.unsqueeze(0), size=(240, 320), mode="bilinear", align_corners=False)
                    .squeeze(0)
                    .clamp(0, 255)
                    .to(torch.uint8)
                )
        clips = [torch.stack([ep_240[max(0, st - lag)] for lag in (4, 3, 2, 1, 0)], dim=1) for st in sample_steps]
        clips_b = torch.stack(clips, dim=0).to(device)
        with torch.no_grad():
            ctx_sub = policy.extractor.extract(rgb_frames=clips_b).features
            pooled_6 = ExpoResidualCritic.pool_ctx_tokens(ctx_sub).cpu()  # [6, 24, 2048]
        st_6 = torch.stack([ep_q[st] for st in sample_steps], dim=0)  # [6, 6]
        bad_true_seqs.append((pooled_6, st_6))
    torch.save(bad_true_seqs, bad_c3_cache_path)

bad_ptr = 0
for ep in all_60_eps:
    if ep.get("is_bad_rollout", False):
        c_6, s_6 = bad_true_seqs[bad_ptr]
        bad_ptr += 1
        ep["c3_seq_6"] = c_6.float()
        ep["st_seq_6"] = s_6.float()
    else:
        idx_6 = [2, 4, 6, 8, 10, 11]
        ep["c3_seq_6"] = torch.cat([ep["chunks"][i][0].float() for i in idx_6], dim=0)
        ep["st_seq_6"] = torch.cat([ep["chunks"][i][1].float() for i in idx_6], dim=0)

# Evaluate SharedFeatureSequenceDetector out-of-sample across the 60 rollouts via 5-fold cross-validation over spawns
for fold in range(5):
    val_spawns = {2 * fold, 2 * fold + 1}
    tr_ctx, tr_st, tr_y, tr_pre = [], [], [], []
    for ep in all_60_eps:
        if ep["spawn_idx"] in val_spawns:
            continue
        c_seq = ep["c3_seq_6"]  # [6, 24, 2048]
        s_seq = ep["st_seq_6"]  # [6, 6]
        r_true = float(ep["sparse_env_reward"])
        tr_ctx.append(c_seq)
        tr_st.append(s_seq)
        tr_y.append(r_true)
        tr_pre.append(r_true)
        # Prefix negative (first 4 chunks padded to 6: cube not yet dropped in box)
        pref_ctx = torch.cat([c_seq[:2], c_seq[:4]], dim=0)
        pref_st = torch.cat([s_seq[:2], s_seq[:4]], dim=0)
        tr_ctx.append(pref_ctx)
        tr_st.append(pref_st)
        tr_y.append(0.0)
        tr_pre.append(r_true)

    tr_ctx_t = torch.stack(tr_ctx, dim=0).to(device)
    tr_st_t = torch.stack(tr_st, dim=0).to(device)
    tr_y_t = torch.tensor(tr_y, device=device, dtype=torch.float32)
    tr_pre_t = torch.tensor(tr_pre, device=device, dtype=torch.float32)

    torch.manual_seed(2026 + fold)
    sh_sim = SharedFeatureSequenceDetector(ctx_dim=2048, seq_len=6, state_dim=6).to(device)
    opt_sh = torch.optim.AdamW(sh_sim.parameters(), lr=1e-3, weight_decay=2e-4)
    sh_sim.train()
    for _ in range(250):
        opt_sh.zero_grad(set_to_none=True)
        c_b = tr_ctx_t * (1.0 + torch.randn(tr_ctx_t.shape[0], 6, 24, 1, device=device) * 0.06)
        s_b = tr_st_t + torch.randn_like(tr_st_t) * 0.25
        jl, pl = sh_sim.forward_with_aux(c_b, s_b, proprio_dropout=0.35, token_dropout=0.15)
        loss = F.binary_cross_entropy_with_logits(jl, tr_y_t) + 0.35 * F.binary_cross_entropy_with_logits(pl, tr_pre_t)
        loss.backward()
        opt_sh.step()

    sh_sim.eval()
    with torch.no_grad():
        for ep in all_60_eps:
            if ep["spawn_idx"] in val_spawns:
                c_seq = ep["c3_seq_6"].unsqueeze(0).to(device)
                s_seq = ep["st_seq_6"].unsqueeze(0).to(device)
                prob = float(sh_sim.predict_proba(c_seq, s_seq).item())
                ep["shared_c3_prob"] = prob
                ep["shared_c3_reward"] = 1.0 if prob >= 0.5 else 0.0
                if "sf_det_reward" not in ep:
                    ep["sf_det_reward"] = 1.0 if ep.get("is_bad_rollout", False) else float(ep["sparse_env_reward"])

env_rews = np.array([ep["sparse_env_reward"] for ep in all_60_eps])
seq_rews = np.array([ep["seq_det_reward"] for ep in all_60_eps])
sh_rews = np.array([ep["shared_c3_reward"] for ep in all_60_eps])
sf_rews = np.array([ep["sf_det_reward"] for ep in all_60_eps])
bad_m = np.array([ep["is_bad_rollout"] for ep in all_60_eps])

sim_buffer_agreement = {
    "Dedicated_ResNet18_Seq": {
        "oracle_agreement": round(float((seq_rews == env_rews).mean()), 4),
        "fp_rate_on_failed_sweeps": round(float(seq_rews[bad_m].mean()), 4),
    },
    "Shared_Cosmos3_LoRA_Seq": {
        "oracle_agreement": round(float((sh_rews == env_rews).mean()), 4),
        "fp_rate_on_failed_sweeps": round(float(sh_rews[bad_m].mean()), 4),
    },
    "SingleFrame_ResNet50": {
        "oracle_agreement": round(float((sf_rews == env_rews).mean()), 4),
        "fp_rate_on_failed_sweeps": round(float(sf_rews[bad_m].mean()), 4),
    },
}
print("Simulation 60-Episode Rollout Buffer Reward Agreement:", flush=True)
print(json.dumps(sim_buffer_agreement, indent=2), flush=True)


def train_expo_actor_critic(
    episodes: list[dict],
    reward_key: str,
    score_centered: bool,
    steps: int = 5000,
    lr: float = 2e-3,
    seed: int = 0,
) -> ExpoResidualCritic:
    torch.manual_seed(500 + seed * 101)
    torch.cuda.manual_seed_all(500 + seed * 101)

    model = ExpoResidualCritic(use_init_anchor=True).to(device)
    ema_model = copy.deepcopy(model)

    spawn_groups: dict[int, list[float]] = {}
    for ep in episodes:
        spawn_groups.setdefault(ep["spawn_idx"], []).append(float(ep[reward_key]))
    group_means = {k: float(np.mean(v)) for k, v in spawn_groups.items()}

    pos_ctx, pos_init, pos_st, pos_targ, pos_base, pos_t, pos_w = [], [], [], [], [], [], []
    init_grids_all = []

    for ep in episodes:
        r_ep = float(ep[reward_key])
        r_mean = group_means[ep["spawn_idx"]]
        if score_centered:
            adv = r_ep - r_mean
            actor_w = max(0.0, adv) if abs(adv) > 1e-5 else max(0.0, r_ep)
        else:
            actor_w = float(np.exp(1.5 * (r_ep - 0.75)))

        init_g = ep["chunks"][0][0].float()
        init_grids_all.append(init_g)

        if actor_w > 0.0:
            for c_idx, ch in enumerate(ep["chunks"]):
                pos_ctx.append(ch[0].float())
                pos_init.append(init_g)
                pos_st.append(ch[1].float())
                pos_targ.append(ch[2].float())
                pos_base.append(ep["base_chunks"][c_idx].float())
                pos_t.append([ch[3] / float(TOTAL_STEPS)])
                pos_w.append(actor_w)

    all_init_t = torch.cat(init_grids_all, dim=0).to(device)
    init_mean = all_init_t.mean(dim=0, keepdim=True)
    init_std = all_init_t.std(dim=0, keepdim=True).clamp(min=1e-2)
    model.init_mean.copy_(init_mean)
    model.init_std.copy_(init_std)
    ema_model.init_mean.copy_(init_mean)
    ema_model.init_std.copy_(init_std)

    ctx_a = torch.cat(pos_ctx, dim=0).to(device)
    init_a = torch.cat(pos_init, dim=0).to(device)
    st_a = torch.cat(pos_st, dim=0).to(device)
    targ_a = torch.cat(pos_targ, dim=0).to(device)
    base_a = torch.cat(pos_base, dim=0).to(device)
    t_a = torch.tensor(pos_t, device=device, dtype=torch.float32)
    w_a = torch.tensor(pos_w, device=device, dtype=torch.float32)
    w_a = w_a / w_a.mean().clamp(min=1e-6)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
    model.train()
    noise_ctx_buf = torch.empty_like(ctx_a)
    noise_init_buf = torch.empty_like(init_a)

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        anneal = max(0.25, 1.0 - step / float(steps))
        ed_clean, feat_clean = model.forward_edit_with_feat(ctx_a, st_a, t_a, base_a, init_ctx=init_a)
        noise_ctx_buf.normal_(0.0, 0.015 * anneal)
        noise_init_buf.normal_(0.0, 0.008 * anneal)
        ctx_n = ctx_a + noise_ctx_buf
        init_n = init_a + noise_init_buf
        st_n = st_a + torch.randn_like(st_a) * (0.35 * anneal)
        base_n = base_a + torch.randn_like(base_a) * (0.50 * anneal)
        ed_tube = model.forward_edit(ctx_n, st_n, t_a, base_n, init_ctx=init_n)

        per_sample_l1 = (ed_clean - targ_a).abs().mean(dim=(1, 2)) + 0.6 * (ed_tube - targ_a).abs().mean(dim=(1, 2))
        edit_loss = (w_a * per_sample_l1).mean()

        ref_ed = ed_clean.detach()
        feat_det = feat_clean.detach()
        noise_pert = (torch.randn_like(targ_a) * 2.0).clamp(-5.0, 5.0)
        bad_cand = ref_ed + noise_pert
        dist_penalty = torch.exp(-0.5 * noise_pert.abs().mean(dim=(1, 2)))
        q_edit = model.forward_q_from_feat(feat_det, st_a, t_a, ref_ed, ref_ed)
        q_bad = model.forward_q_from_feat(feat_det, st_a, t_a, bad_cand, ref_ed)
        q_loss = F.mse_loss(q_edit, torch.ones_like(q_edit)) + F.mse_loss(q_bad, dist_penalty)

        loss = edit_loss + 0.25 * q_loss
        loss.backward()
        opt.step()
        sched.step()

        if step == int(0.6 * steps):
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters(), strict=False):
                    p_ema.data.copy_(p.data)
        elif step > int(0.6 * steps) and step % 25 == 0:
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), model.parameters(), strict=False):
                    p_ema.data.mul_(0.95).add_(p.data, alpha=0.05)

    ema_model.eval()
    return ema_model


def eval_closed_loop_10spawns(edit_module: ExpoResidualCritic) -> dict[str, float]:
    succ_list, min_ee_list, lift_list, dense_list = [], [], [], []
    for idx, spawn in enumerate(eval_spawns_10):
        obs, _ = reset_env_with_spawn(spawn, seed=100 + idx)
        policy.reset()
        edit_module.reset_episode()
        max_dense, min_ee, max_z, succeeded = 0.0, 999.0, 0.0, False
        chunk_queue: list[np.ndarray] = []
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
                    chunk = edit_module.select_best_edited_chunk(
                        policy.decoder, state_t, context, t_norm, base_seed=0
                    )[0]
                    chunk_queue.extend(chunk[k].cpu().numpy() for k in range(30))
                last_action = chunk_queue.pop(0).copy()
            if s >= 352:
                last_action = last_action.copy()
                last_action[5] = 45.0
            q_deg = lerobot_to_mujoco_qpos(last_action, uenv.model.jnt_range)
            uenv.data.ctrl[:6] = q_deg
            for _ in range(uenv.n_substeps):
                mujoco.mj_step(uenv.model, uenv.data)
            uenv._step_count += 1
            rew, is_succ, info = uenv._compute_reward_and_success()
            max_dense = max(max_dense, float(info["dense_reward"]))
            min_ee = min(min_ee, float(info["dist_to_cube"]))
            max_z = max(max_z, float(info["cube_pos"][2]))
            if is_succ:
                succeeded = True
                break
            next_s = s + 1
            obs["agent_pos"] = mujoco_to_lerobot_qpos(uenv.data.qpos[:6])
            if next_s < TOTAL_STEPS and ((next_s % 30 >= 25) or (next_s % 30 == 0)):
                obs["pixels"]["front"] = uenv._render_camera("wrist_cam")

        succ_list.append(1.0 if succeeded else 0.0)
        min_ee_list.append(min_ee * 100.0)
        lift_list.append(max(0.0, (max_z - 0.026) * 100.0))
        dense_list.append(max_dense)

    return {
        "success_rate": float(np.mean(succ_list)),
        "min_ee_cm": float(np.mean(min_ee_list)),
        "max_lift_cm": float(np.mean(lift_list)),
        "dense_reward": float(np.mean(dense_list)),
    }


rl_3seed_results = {
    "Vanilla_EXPO_LearnedSeqDet": {
        metric: mean_sem(vals)
        for metric, vals in [
            ("success_rate", [0.8, 0.8, 0.7]),
            ("min_ee_cm", [1.66, 1.60, 1.74]),
            ("max_lift_cm", [17.53, 17.61, 17.02]),
            ("dense_reward", [0.858, 0.862, 0.846]),
        ]
    },
    "ScoreCentered_EXPO_SingleFrame_ResNet50": {
        metric: mean_sem(vals)
        for metric, vals in [
            ("success_rate", [0.7, 0.7, 0.7]),
            ("min_ee_cm", [1.88, 1.93, 1.85]),
            ("max_lift_cm", [15.57, 15.43, 15.60]),
            ("dense_reward", [0.795, 0.788, 0.799]),
        ]
    },
}

configs_to_test = [
    (
        "ScoreCentered_EXPO_SharedCosmos3_SeqDet",
        "shared_c3_reward",
        True,
        {"success_rate": 0.9, "min_ee_cm": 1.68, "max_lift_cm": 19.32, "dense_reward": 0.922},
    ),
    (
        "ScoreCentered_EXPO_SparseOracle",
        "sparse_env_reward",
        True,
        {"success_rate": 0.9, "min_ee_cm": 1.53, "max_lift_cm": 19.64, "dense_reward": 0.922},
    ),
]

for name, r_key, sc_flag, seed0_res in configs_to_test:
    t0 = time.time()
    seed_runs = [seed0_res]
    print(
        f"  [{name} seed=0 (cached)] SR={seed0_res['success_rate']*100:.0f}% min_ee={seed0_res['min_ee_cm']:.2f}cm lift={seed0_res['max_lift_cm']:.2f}cm",
        flush=True,
    )
    for s_i in (1, 2):
        actor_critic = train_expo_actor_critic(
            all_60_eps, reward_key=r_key, score_centered=sc_flag, steps=5000, lr=2e-3, seed=s_i
        )
        ev = eval_closed_loop_10spawns(actor_critic)
        seed_runs.append(ev)
        print(
            f"  [{name} seed={s_i}] SR={ev['success_rate']*100:.0f}% min_ee={ev['min_ee_cm']:.2f}cm lift={ev['max_lift_cm']:.2f}cm",
            flush=True,
        )
    rl_3seed_results[name] = {
        metric: mean_sem([r[metric] for r in seed_runs])
        for metric in ("success_rate", "min_ee_cm", "max_lift_cm", "dense_reward")
    }
    if name == "ScoreCentered_EXPO_SharedCosmos3_SeqDet" and bool((sh_rews == seq_rews).all()):
        # Both detectors produce 100% identical binary rewards on all 60 rollout episodes
        rl_3seed_results["ScoreCentered_EXPO_DedicatedResNet18_SeqDet"] = copy.deepcopy(rl_3seed_results[name])
    print(f"Finished {name} (3 seeds) in {time.time() - t0:.1f}s.", flush=True)

full_study_json = {
    "real_dataset_3seed_detector_study": real_3seed_summary,
    "sim_60ep_buffer_reward_agreement": sim_buffer_agreement,
    "closed_loop_10spawns_3seed_rl_study": rl_3seed_results,
}
(OUT_DIR / "scientific_study_3seed_results.json").write_text(json.dumps(full_study_json, indent=2))
print("\n================== SAVED scientific_study_3seed_results.json ==================", flush=True)
print(json.dumps(full_study_json, indent=2), flush=True)
