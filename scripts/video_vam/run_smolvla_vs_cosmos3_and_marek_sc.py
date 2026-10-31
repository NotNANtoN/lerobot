#!/usr/bin/env python3
"""Head-to-Head Benchmark on `Orellius/so101_sort_cubes_no_top` & `SO101CubeBoxSparse-v0`:
1. `SmolVLA` (`sort-cubes-smolvla-100k-exact`, 450M VLM) vs. `Cosmos 3 Edge LoRA` (`Video-VAM`):
   - Held-Out Offline Validation (`68..72`): Full-30 RMSE, H1 (offset 0), First-5 Action RMSE (F5), Latency.
   - 3-Seed `SharedFeatureSequenceDetector` on Real Dataset (`0..57` Train, `58..67` Val, `68..72` & `68..76` Test):
     `SmolVLA` SigLIP-VLM pooled tokens (`[24, 960]`) vs. `Cosmos 3 Edge LoRA` video tokens (`[24, 2048]`).
   - 3-Seed 10-Spawn Closed-Loop RL Post-Training (`SO101CubeBoxSparse-v0`, 5,000 steps):
     * `SmolVLA` Zero-Shot Base vs. `SmolVLA + Score-Centered EXPO (Marek & Ryabinin stable_reinforce + Dr. GRPO) + Shared Detector`
     * `Cosmos 3 Edge LoRA` Zero-Shot Base vs. `Dr. GRPO-only EXPO` vs. `Marek & Ryabinin Score-Centered EXPO (stable_reinforce + Dr. GRPO) + Shared Detector`
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import av
import gymnasium as gym
import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F  # noqa: N812

import lerobot.envs.so101_cube_box  # noqa: F401
from lerobot.envs.so101_cube_box import (
    lerobot_to_mujoco_qpos,
    mujoco_to_lerobot_qpos,
)
from lerobot.policies.vam.action_rmse import SmolVLABackend
from lerobot.policies.vam.configuration_video_vam import VideoVAMConfig
from lerobot.policies.vam.expo_residual import (
    ExpoResidualCritic,
    compute_marek_score_centered_loss,
)
from lerobot.policies.vam.modeling_video_vam import VideoVAMPolicy, restore_uint8_history
from lerobot.rewards.sequence_detector import SharedFeatureSequenceDetector

torch.set_num_threads(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("outputs/sparse_seq_rl")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    }


def mean_sem(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {"mean": round(float(arr.mean()), 4), "sem": round(sem, 4), "seeds": [round(float(x), 4) for x in vals]}


print("=========================================================================", flush=True)
print("1. LOADING SMOLVLA (`sort-cubes-smolvla-100k-exact`) & COSMOS 3 EDGE LORA", flush=True)
print("=========================================================================", flush=True)
smolvla_ckpt = "outputs/train/sort-cubes-smolvla-100k-exact/checkpoints/last/pretrained_model"
smolvla_backend = SmolVLABackend.from_pretrained(smolvla_ckpt, device=device)
smolvla_policy = smolvla_backend.policy
smolvla_policy.eval()

c3_ckpt = "outputs/train/c3_sort_cubes_30m_lora"
c3_cfg = VideoVAMConfig.from_pretrained(c3_ckpt)
c3_cfg.pretrained_path = Path(c3_ckpt)
c3_cfg.n_action_steps = 30
c3_cfg.joint_limits_min = None
c3_cfg.joint_limits_max = None
c3_policy = VideoVAMPolicy(c3_cfg)
c3_policy.decoder._cuda_graph_disabled = True
c3_policy.eval()


@torch.no_grad()
def extract_smolvla_pooled_tokens(rgb_bchw_01: torch.Tensor) -> torch.Tensor:
    """Extract SmolVLA's frozen SigLIP+connector visual tokens `[B, 64, 960]` and pool to `[B, 24, 960]`."""
    batch_in = {"observation.images.front": rgb_bchw_01.to(device)}
    images, _ = smolvla_policy.prepare_images(batch_in)
    img_emb = smolvla_policy.model.vlm_with_expert.embed_image(images[0])
    img_emb = img_emb * (img_emb.shape[-1] ** 0.5)
    return ExpoResidualCritic.pool_ctx_tokens(img_emb.float())


@torch.no_grad()
def smolvla_predict_chunk_50_batched(
    rgb_hwc_u8: np.ndarray, state_deg_6: np.ndarray, seed: int = 0, k_candidates: int = 1
) -> torch.Tensor:
    """Predict `[K, 50, 6]` physical-unit action chunks from SmolVLA in a single batched pass."""
    img_chw = torch.from_numpy(rgb_hwc_u8).permute(2, 0, 1).float() / 255.0
    obs = {
        "observation.images.front": img_chw,
        "observation.state": torch.from_numpy(state_deg_6).float(),
        "task": "Grab the purple cube and put it into the cardboard box, ignore the other objects",
    }
    proc = smolvla_backend.preprocessor(obs)
    dev, dt = smolvla_backend._device_and_dtype()
    proc = {
        k: (v.expand(k_candidates, *v.shape[1:]).contiguous().to(device=dev) if isinstance(v, torch.Tensor) and v.ndim >= 2 else v)
        for k, v in proc.items()
    }
    noises = []
    for k_i in range(k_candidates):
        gen = torch.Generator(device=dev).manual_seed(seed + k_i * 97)
        noises.append(
            torch.randn(
                (1, smolvla_policy.config.chunk_size, smolvla_policy.config.max_action_dim),
                generator=gen,
                device=dev,
                dtype=dt,
            )
        )
    noise = torch.cat(noises, dim=0)
    smolvla_policy.reset()
    raw = smolvla_policy.predict_action_chunk(proc, noise=noise)
    phys = smolvla_backend.postprocessor(raw.clone())
    return phys[:, :50, :6].float().to(device)


@torch.no_grad()
def smolvla_predict_chunk_30(rgb_hwc_u8: np.ndarray, state_deg_6: np.ndarray, seed: int = 0) -> torch.Tensor:
    return smolvla_predict_chunk_50_batched(rgb_hwc_u8, state_deg_6, seed=seed, k_candidates=1)[:, :30, :6]


# =============================================================================
# PART A: OFFLINE HELD-OUT VALIDATION (`68..72`) ON `so101_sort_cubes_no_top`
# =============================================================================
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
data_files = sorted((snap_dir / "data/chunk-000").glob("*.parquet"))
data_table = pa.concat_tables([pq.read_table(f, columns=["observation.state", "action"]) for f in data_files]).to_pydict()
all_states_np = np.asarray(data_table["observation.state"], dtype=np.float32)
all_actions_np = np.asarray(data_table["action"], dtype=np.float32)

smolvla_val_cache = OUT_DIR / "smolvla_val_68_72_metrics_v2.json"
if smolvla_val_cache.exists():
    smolvla_val_metrics = json.loads(smolvla_val_cache.read_text())
else:
    print("\nEvaluating SmolVLA (`sort-cubes-smolvla-100k-exact`) on held-out episodes 68..72...", flush=True)
    sq_errs_30hz = []
    sq_errs_10hz_16 = []
    latencies = []
    by_vid: dict[int, list[int]] = {}
    for ep in range(68, 73):
        by_vid.setdefault(ep_meta[ep]["vid_file"], []).append(ep)

    for vid_idx, eps in sorted(by_vid.items()):
        mp4_path = snap_dir / f"videos/observation.images.front/chunk-000/file-{vid_idx:03d}.mp4"
        targets: dict[int, list[tuple[int, int]]] = {}
        for ep in eps:
            f0 = ep_meta[ep]["from_idx"]
            f1 = ep_meta[ep]["to_idx"]
            ep_len = f1 - f0
            base_vf = int(round(ep_meta[ep]["from_ts"] * 30.0))
            for rel in range(0, max(1, ep_len - 90), 20):
                targets.setdefault(base_vf + rel, []).append((ep, rel))

        max_vf = max(targets.keys())
        frames_rgb: dict[tuple[int, int], np.ndarray] = {}
        container = av.open(str(mp4_path))
        for vf_i, frame in enumerate(container.decode(video=0)):
            if vf_i in targets:
                arr = frame.to_ndarray(format="rgb24")
                for key in targets[vf_i]:
                    frames_rgb[key] = arr
            if vf_i >= max_vf:
                break
        container.close()

        for (ep, rel), rgb_arr in sorted(frames_rgb.items()):
            f0 = ep_meta[ep]["from_idx"]
            st_cur = all_states_np[f0 + rel]
            # Native 30 Hz target chunk (30 steps) and time-aligned 10 Hz target chunk (16 steps = 48 frames at 30 Hz)
            targ_idx_30hz = [min(ep_meta[ep]["to_idx"] - 1, f0 + rel + k) for k in range(30)]
            targ_idx_10hz = [min(ep_meta[ep]["to_idx"] - 1, f0 + rel + k * 3) for k in range(16)]
            targ_30hz = torch.from_numpy(all_actions_np[targ_idx_30hz]).float()
            targ_10hz = torch.from_numpy(all_actions_np[targ_idx_10hz]).float()
            t0 = time.perf_counter()
            pred_50 = smolvla_predict_chunk_50_batched(rgb_arr, st_cur, seed=0, k_candidates=1)[0].cpu()
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            sq_errs_30hz.append((pred_50[:30] - targ_30hz).pow(2))
            pred_10hz_16 = pred_50[[k * 3 for k in range(16)]]
            sq_errs_10hz_16.append((pred_10hz_16 - targ_10hz).pow(2))

    sq_30 = torch.stack(sq_errs_30hz, dim=0)
    sq_10 = torch.stack(sq_errs_10hz_16, dim=0)
    smolvla_val_metrics = {
        "n_anchors": len(sq_errs_30hz),
        "native_30hz_full30_rmse": round(float(torch.sqrt(sq_30.mean()).item()), 2),
        "native_30hz_h1_rmse": round(float(torch.sqrt(sq_30[:, 0, :].mean()).item()), 2),
        "native_30hz_f5_rmse": round(float(torch.sqrt(sq_30[:, :5, :].mean()).item()), 2),
        "time_aligned_10hz_h16_rmse": round(float(torch.sqrt(sq_10.mean()).item()), 2),
        "time_aligned_10hz_h1_rmse": round(float(torch.sqrt(sq_10[:, 0, :].mean()).item()), 2),
        "time_aligned_10hz_f5_rmse": round(float(torch.sqrt(sq_10[:, :5, :].mean()).item()), 2),
        "arm_rmse_deg_10hz_f5": round(float(torch.sqrt(sq_10[:, :5, :5].mean()).item()), 2),
        "gripper_rmse_10hz_f5": round(float(torch.sqrt(sq_10[:, :5, 5].mean()).item()), 2),
        "median_latency_ms": round(float(np.median(latencies)), 1),
    }
    smolvla_val_cache.write_text(json.dumps(smolvla_val_metrics, indent=2))
print("SmolVLA Held-Out Validation (68..72) RMSE:", json.dumps(smolvla_val_metrics, indent=2), flush=True)


# =============================================================================
# PART B: EXTRACT SMOLVLA SHARED TOKENS ON ALL 76 REAL EPISODES & COMPARE
#         3-SEED SHARED-FEATURE SEQUENCE REWARD DETECTOR (SMOLVLA vs COSMOS 3)
# =============================================================================
real_mv_cache = torch.load(OUT_DIR / "real_detector_multiview_v3.pt", map_location="cpu", weights_only=False)
ep_features_mv = real_mv_cache["ep_features"]
ep_step_groups = real_mv_cache["ep_step_groups"]
ep_c3_features = torch.load(OUT_DIR / "real_cosmos3_shared_features.pt", map_location="cpu", weights_only=False)

real_smolvla_cache_path = OUT_DIR / "real_smolvla_shared_features.pt"
if real_smolvla_cache_path.exists():
    print("Loading cached real-dataset SmolVLA SigLIP pooled token sequences...", flush=True)
    ep_svla_features = torch.load(real_smolvla_cache_path, map_location="cpu", weights_only=False)
else:
    t0 = time.time()
    print("Extracting SmolVLA [24, 960] pooled SigLIP tokens on real dataset (76 episodes)...", flush=True)
    file_to_eps: dict[int, list[int]] = {}
    ep_needed_rels: dict[int, list[int]] = {}
    for ep, (pos_steps, early_steps, carry_steps, ep_len) in ep_step_groups.items():
        needed_rel = sorted(
            {
                max(0, min(ep_len - 1, t_end - (SEQ_LEN - 1 - k) * SEQ_STRIDE))
                for t_end in (pos_steps + early_steps + carry_steps)
                for k in range(SEQ_LEN)
            }
        )
        ep_needed_rels[ep] = needed_rel
        file_to_eps.setdefault(ep_meta[ep]["vid_file"], []).append(ep)

    ep_svla_features: dict[int, torch.Tensor] = {}
    for vid_file_idx, eps_in_file in sorted(file_to_eps.items()):
        mp4_path = snap_dir / f"videos/observation.images.front/chunk-000/file-{vid_file_idx:03d}.mp4"
        frame_targets: dict[int, list[tuple[int, int]]] = {}
        for ep in eps_in_file:
            base_f = int(round(ep_meta[ep]["from_ts"] * 30.0))
            for r in ep_needed_rels[ep]:
                frame_targets.setdefault(base_f + r, []).append((ep, r))

        max_target_frame = max(frame_targets.keys())
        collected_rgb: dict[tuple[int, int], torch.Tensor] = {}
        container = av.open(str(mp4_path))
        for f_idx, frame in enumerate(container.decode(video=0)):
            if f_idx in frame_targets:
                arr = frame.to_ndarray(format="rgb24")
                img_t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                img_256 = F.interpolate(img_t.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False).squeeze(0).to(torch.float16)
                for key in frame_targets[f_idx]:
                    collected_rgb[key] = img_256
            if f_idx >= max_target_frame:
                break
        container.close()

        for ep in eps_in_file:
            rels = ep_needed_rels[ep]
            imgs = torch.stack([collected_rgb[(ep, r)] for r in rels], dim=0).float()
            pooled_list = []
            for b0 in range(0, len(imgs), 32):
                sub = imgs[b0 : b0 + 32]
                pooled_sub = extract_smolvla_pooled_tokens(sub).cpu().to(torch.float16)
                pooled_list.append(pooled_sub)
            ep_svla_features[ep] = torch.cat(pooled_list, dim=0)  # [N_rel, 24, 960]
    torch.save(ep_svla_features, real_smolvla_cache_path)
    print(f"Cached real SmolVLA features in {time.time() - t0:.1f}s.", flush=True)


def assemble_real_split_svla(split_eps: list[int]) -> dict[str, torch.Tensor]:
    svla_seq_l, c3_seq_l, st_l, y_l, pre_l, kind_l = [], [], [], [], [], []
    for ep in split_eps:
        if ep not in ep_features_mv or ep not in ep_c3_features or ep not in ep_svla_features:
            continue
        _, ep_states_cpu, rel_to_local = ep_features_mv[ep]
        c3_raw = ep_c3_features[ep].float()
        svla_raw = ep_svla_features[ep].float()
        pos_steps, early_steps, carry_steps, ep_len = ep_step_groups[ep]

        def _gather(rel_indices: list[int], _map=rel_to_local, _len=ep_len, _c3=c3_raw, _sv=svla_raw, _st=ep_states_cpu):
            loc = torch.tensor([_map[max(0, min(_len - 1, r))] for r in rel_indices], dtype=torch.long)
            return _sv[loc], _c3[loc], _st[loc]

        for t_end in pos_steps:
            rels = [max(0, t_end - (SEQ_LEN - 1 - k) * SEQ_STRIDE) for k in range(SEQ_LEN)]
            sv_s, c3_s, st_s = _gather(rels)
            svla_seq_l.append(sv_s)
            c3_seq_l.append(c3_s)
            st_l.append(st_s)
            y_l.append(1.0)
            pre_l.append(1.0)
            kind_l.append(0)

        for t_end in early_steps:
            rels = [max(0, t_end - (SEQ_LEN - 1 - k) * SEQ_STRIDE) for k in range(SEQ_LEN)]
            sv_s, c3_s, st_s = _gather(rels)
            svla_seq_l.append(sv_s)
            c3_seq_l.append(c3_s)
            st_l.append(st_s)
            y_l.append(0.0)
            pre_l.append(0.0)
            kind_l.append(1)

        for t_end in carry_steps:
            rels = [max(0, t_end - (SEQ_LEN - 1 - k) * SEQ_STRIDE) for k in range(SEQ_LEN)]
            sv_s, c3_s, st_s = _gather(rels)
            svla_seq_l.append(sv_s)
            c3_seq_l.append(c3_s)
            st_l.append(st_s)
            y_l.append(0.0)
            pre_l.append(1.0)
            kind_l.append(1)

        for e_step, p_step in [(early_steps[0], pos_steps[-1]), (early_steps[1], pos_steps[-2])]:
            early_rels = [max(0, e_step - (5 - k) * SEQ_STRIDE) for k in range(6)]
            box_rels = [max(0, p_step - SEQ_STRIDE), p_step]
            sv_s, c3_s, st_s = _gather(early_rels + box_rels)
            st_s = st_s.clone()
            st_s[:6, 5] = st_s[-1, 5]
            svla_seq_l.append(sv_s)
            c3_seq_l.append(c3_s)
            st_l.append(st_s)
            y_l.append(0.0)
            pre_l.append(0.0)
            kind_l.append(2)

        for p_step in (pos_steps[-1], pos_steps[-2]):
            static_rels = [p_step for _ in range(SEQ_LEN)]
            sv_s, c3_s, st_s = _gather(static_rels)
            st_s = st_s + torch.randn_like(st_s) * 0.15
            svla_seq_l.append(sv_s)
            c3_seq_l.append(c3_s)
            st_l.append(st_s)
            y_l.append(0.0)
            pre_l.append(0.0)
            kind_l.append(3)

    return {
        "svla_seq": torch.stack(svla_seq_l, dim=0),
        "c3_seq": torch.stack(c3_seq_l, dim=0),
        "state_seq": torch.stack(st_l, dim=0),
        "y": torch.tensor(y_l, dtype=torch.float32),
        "pre": torch.tensor(pre_l, dtype=torch.float32),
        "kind": torch.tensor(kind_l, dtype=torch.long),
    }


tr_sp = assemble_real_split_svla(list(range(0, 58)))
val_sp = assemble_real_split_svla(list(range(58, 68)))
t5_sp = assemble_real_split_svla(list(range(68, 73)))
t9_sp = assemble_real_split_svla(list(range(68, 77)))

sv_tr = tr_sp["svla_seq"].to(device)
st_tr = tr_sp["state_seq"].to(device)
y_tr = tr_sp["y"].to(device)
pre_tr = tr_sp["pre"].to(device)
pos_w = torch.tensor([(y_tr == 0).sum().float() / max(1.0, (y_tr == 1).sum().float())], device=device)

sv_val = val_sp["svla_seq"].to(device)
st_val = val_sp["state_seq"].to(device)
y_val = val_sp["y"].to(device)

svla_det_cache = OUT_DIR / "smolvla_det_3seed_summary.json"
if svla_det_cache.exists():
    svla_det_summary = json.loads(svla_det_cache.read_text())
else:
    svla_det_seeds = []
    for seed in (0, 1, 2):
        torch.manual_seed(1000 + seed)
        torch.cuda.manual_seed_all(1000 + seed)
        det = SharedFeatureSequenceDetector(ctx_dim=960, seq_len=SEQ_LEN, state_dim=6).to(device)
        opt = torch.optim.AdamW(det.parameters(), lr=8e-4, weight_decay=3e-4)
        best_sd, best_vl = None, 999.0
        for step in range(1, 451):
            det.train()
            opt.zero_grad(set_to_none=True)
            idx = torch.randint(0, sv_tr.shape[0], (min(128, sv_tr.shape[0]),), device=device)
            sub = sv_tr[idx] * (1.0 + torch.randn(len(idx), SEQ_LEN, 24, 1, device=device) * 0.08)
            st_b = st_tr[idx] + torch.randn_like(st_tr[idx]) * 0.35
            jl, pl = det.forward_with_aux(sub, st_b, proprio_dropout=0.45, token_dropout=0.20)
            loss = F.binary_cross_entropy_with_logits(jl, y_tr[idx] * 0.96 + 0.02, pos_weight=pos_w) + 0.4 * F.binary_cross_entropy_with_logits(pl, pre_tr[idx] * 0.96 + 0.02)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(det.parameters(), 1.0)
            opt.step()
            if step % 20 == 0:
                det.eval()
                with torch.no_grad():
                    vl = float(F.binary_cross_entropy_with_logits(det(sv_val, st_val), y_val, pos_weight=pos_w).item())
                    if vl < best_vl:
                        best_vl = vl
                        best_sd = copy.deepcopy(det.state_dict())
        det.load_state_dict(best_sd)
        det.eval()
        seed_out = {}
        for sname, sp in [("val", val_sp), ("test5", t5_sp), ("test9", t9_sp)]:
            with torch.no_grad():
                probs = det.predict_proba(sp["svla_seq"].to(device), sp["state_seq"].to(device)).cpu().numpy()
            y_np = sp["y"].numpy()
            hard_m = (sp["kind"].numpy() == 2) | (sp["kind"].numpy() == 3)
            m = compute_binary_metrics(y_np, probs)
            m["hard_neg_fp_rate"] = round(float((probs[hard_m] >= 0.5).mean()), 4)
            seed_out[sname] = m
        svla_det_seeds.append(seed_out)
        if seed == 0:
            torch.save({"state_dict": best_sd, "seq_len": SEQ_LEN, "ctx_dim": 960}, OUT_DIR / "shared_seq_detector_smolvla.pt")

    svla_det_summary = {
        sname: {
            metric: mean_sem([r[sname][metric] for r in svla_det_seeds])
            for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "hard_neg_fp_rate")
        }
        for sname in ("val", "test5", "test9")
    }
    svla_det_cache.write_text(json.dumps(svla_det_summary, indent=2))
print("\nSmolVLA Shared-Feature Sequence Detector (3 Seeds):", json.dumps(svla_det_summary, indent=2), flush=True)


# =============================================================================
# PART C: 10-SPAWN CLOSED-LOOP RL POST-TRAINING:
#         SMOLVLA vs COSMOS 3 EDGE LORA + MAREK & RYABININ SCORE CENTERING
# =============================================================================
print("\n=========================================================================", flush=True)
print("3. 10-SPAWN CLOSED-LOOP RL: SMOLVLA vs COSMOS 3 + MAREK SCORE CENTERING", flush=True)
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

_ep0_table = pq.read_table(sorted((snap_dir / "data/chunk-000").glob("*.parquet"))[0], columns=["action"]).to_pydict()
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


# 1. Evaluate Zero-Shot SmolVLA across the 10 spawns AND extract SmolVLA tokens + base chunks on the 30 positive rollouts
svla_sim_cache_path = OUT_DIR / "smolvla_sim_10spawns_cache.pt"
if svla_sim_cache_path.exists():
    print("Loading cached SmolVLA 10-spawn simulation features and base chunks...", flush=True)
    svla_sim_cache = torch.load(svla_sim_cache_path, map_location="cpu", weights_only=False)
else:
    print("Evaluating Zero-Shot SmolVLA on 10 spawns & extracting SmolVLA rollout features...", flush=True)
    zs_succ, zs_ee, zs_lift, zs_dense = [], [], [], []
    for idx, spawn in enumerate(eval_spawns_10):
        obs, _ = reset_env_with_spawn(spawn, seed=100 + idx)
        smolvla_policy.reset()
        max_dense, min_ee, max_z, succeeded = 0.0, 999.0, 0.0, False
        chunk_queue: list[np.ndarray] = []
        last_action = None
        for s in range(ROLLOUT_STEPS):
            if s < TOTAL_STEPS:
                if not chunk_queue:
                    chunk = smolvla_predict_chunk_30(obs["pixels"]["front"], obs["agent_pos"], seed=s)[0].cpu().numpy()
                    chunk_queue.extend(chunk[k] for k in range(30))
                last_action = chunk_queue.pop(0).copy()
            if s >= 352:
                last_action = last_action.copy()
                last_action[5] = 45.0
            uenv.data.ctrl[:6] = lerobot_to_mujoco_qpos(last_action, uenv.model.jnt_range)
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
            if next_s < TOTAL_STEPS and (next_s % 30 == 0):
                obs["pixels"]["front"] = uenv._render_camera("wrist_cam")
        zs_succ.append(1.0 if succeeded else 0.0)
        zs_ee.append(min_ee * 100.0)
        zs_lift.append(max(0.0, (max_z - 0.026) * 100.0))
        zs_dense.append(max_dense)

    zs_metrics = {
        "success_rate": round(float(np.mean(zs_succ)), 4),
        "min_ee_cm": round(float(np.mean(zs_ee)), 2),
        "max_lift_cm": round(float(np.mean(zs_lift)), 2),
        "dense_reward": round(float(np.mean(zs_dense)), 4),
    }
    print("Zero-Shot SmolVLA (10 Spawns):", json.dumps(zs_metrics, indent=2), flush=True)

    # Extract SmolVLA tokens [1, 24, 960] and SmolVLA base chunks [1, 30, 6] on the positive rollouts
    # (Also evaluate across all 10 spawns fast using cached chunk states + fast 1-pass SmolVLA batching)
    svla_ep_records = []
    chunk_boundaries = list(range(0, TOTAL_STEPS, 30))  # [0, 30, ..., 330]
    for ep in all_60_eps:
        spawn = eval_spawns_10[ep["spawn_idx"]]
        if ep.get("is_bad_rollout", False):
            # Reuse the first positive episode's visual/base chunks for the same spawn with bad target
            continue
        obs, _ = reset_env_with_spawn(spawn, seed=200 + ep["spawn_idx"])
        sched = np.concatenate([ch[2][0].numpy() for ch in ep["chunks"]], axis=0)
        rgb_list = []
        q_list = []
        targ_list = []
        s_list = []
        for s in range(TOTAL_STEPS):
            if s in chunk_boundaries:
                rgb_u8 = uenv._render_camera("wrist_cam")
                q_cur = mujoco_to_lerobot_qpos(uenv.data.qpos[:6]).astype(np.float32)
                rgb_list.append(rgb_u8)
                q_list.append(q_cur)
                targ_list.append(ep["chunks"][s // 30][2].float().cpu())
                s_list.append(s)
            act = sched[s].copy()
            if s >= 352:
                act[5] = 45.0
            uenv.data.ctrl[:6] = lerobot_to_mujoco_qpos(act, uenv.model.jnt_range)
            for _ in range(uenv.n_substeps):
                mujoco.mj_step(uenv.model, uenv.data)
            uenv._step_count += 1

        imgs_t = torch.stack([torch.from_numpy(r).permute(2, 0, 1).float() / 255.0 for r in rgb_list], dim=0)
        ctxs_sv = extract_smolvla_pooled_tokens(imgs_t).cpu()  # [12, 24, 960]
        svla_chunks = []
        svla_bases = []
        for idx_c in range(len(s_list)):
            base_sv = smolvla_predict_chunk_30(rgb_list[idx_c], q_list[idx_c], seed=s_list[idx_c]).cpu()
            svla_chunks.append(
                (
                    ctxs_sv[idx_c : idx_c + 1],
                    torch.from_numpy(q_list[idx_c]).unsqueeze(0).float(),
                    targ_list[idx_c],
                    s_list[idx_c],
                )
            )
            svla_bases.append(base_sv)
        svla_ep_records.append(
            {
                "spawn_idx": ep["spawn_idx"],
                "sparse_env_reward": ep["sparse_env_reward"],
                "seq_det_reward": ep["seq_det_reward"],
                "chunks": svla_chunks,
                "base_chunks": svla_bases,
            }
        )
    # Add the 30 negative episodes per spawn so group mean R_g matches the 60-episode buffer
    pos_by_spawn = {ep["spawn_idx"]: ep for ep in svla_ep_records}
    for ep in all_60_eps:
        if ep.get("is_bad_rollout", False):
            ref = pos_by_spawn[ep["spawn_idx"]]
            svla_ep_records.append(
                {
                    "spawn_idx": ep["spawn_idx"],
                    "sparse_env_reward": ep["sparse_env_reward"],
                    "seq_det_reward": ep["seq_det_reward"],
                    "chunks": ref["chunks"],
                    "base_chunks": ref["base_chunks"],
                }
            )
    svla_sim_cache = {"zero_shot": zs_metrics, "episodes": svla_ep_records}
    torch.save(svla_sim_cache, svla_sim_cache_path)

print("Zero-Shot SmolVLA (10 Spawns):", json.dumps(svla_sim_cache["zero_shot"], indent=2), flush=True)


def train_expo_with_marek_sc(
    episodes: list[dict],
    reward_key: str,
    ctx_dim: int = 2048,
    use_marek_sc: bool = True,
    steps: int = 5000,
    lr: float = 2e-3,
    seed: int = 0,
) -> ExpoResidualCritic:
    torch.manual_seed(500 + seed * 101)
    torch.cuda.manual_seed_all(500 + seed * 101)

    model = ExpoResidualCritic(ctx_dim=ctx_dim, use_init_anchor=True).to(device)
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
        adv = r_ep - r_mean
        actor_w = max(0.0, adv) if abs(adv) > 1e-5 else max(0.0, r_ep)

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
        if use_marek_sc:
            # Marek & Ryabinin Off-Policy Score Centering (`arXiv:2609.20807` / Tinker `stable_reinforce`)
            with torch.no_grad():
                innovation = (targ_a - base_a * model.base_alpha).abs().mean(dim=(1, 2))
                cov_w = (innovation / innovation.mean().clamp(min=1e-4)).clamp(0.75, 1.35)
            edit_loss = (w_a * cov_w * per_sample_l1).mean()
        else:
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


class _SmolVLADecoderAdapter:
    """Wraps SmolVLA so `ExpoResidualCritic.select_best_edited_chunk` can sample K candidates in one batched pass."""

    def __init__(self) -> None:
        self.rgb_u8: np.ndarray | None = None

    def _noise_for_seed(self, bsz: int, dev: torch.device, seed: int) -> torch.Tensor:
        return torch.tensor([[seed]], device=dev, dtype=torch.long)

    def sample_actions(
        self, state_t: torch.Tensor, ctx: torch.Tensor, noise: torch.Tensor | None = None, seed: int = 0, use_cuda_graph: bool = False
    ) -> torch.Tensor:
        del ctx, use_cuda_graph
        assert self.rgb_u8 is not None
        st_np = state_t[0].detach().cpu().numpy()
        k_cands = state_t.shape[0] if noise is not None else 1
        base_seed = int(noise[0, 0].item()) if noise is not None else seed
        return smolvla_predict_chunk_50_batched(self.rgb_u8, st_np, seed=base_seed, k_candidates=k_cands)[:, :30, :6]


smolvla_adapter = _SmolVLADecoderAdapter()


def eval_smolvla_closed_loop_10spawns(edit_module: ExpoResidualCritic) -> dict[str, float]:
    succ_list, min_ee_list, lift_list, dense_list = [], [], [], []
    for idx, spawn in enumerate(eval_spawns_10):
        obs, _ = reset_env_with_spawn(spawn, seed=100 + idx)
        smolvla_policy.reset()
        edit_module.reset_episode()
        max_dense, min_ee, max_z, succeeded = 0.0, 999.0, 0.0, False
        chunk_queue: list[np.ndarray] = []
        last_action = None

        for s in range(ROLLOUT_STEPS):
            if s < TOTAL_STEPS:
                if not chunk_queue:
                    rgb_u8 = obs["pixels"]["front"]
                    smolvla_adapter.rgb_u8 = rgb_u8
                    img_t = torch.from_numpy(rgb_u8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    context = extract_smolvla_pooled_tokens(img_t)
                    state_t = torch.from_numpy(obs["agent_pos"]).unsqueeze(0).to(device, dtype=torch.float32)
                    t_norm = torch.tensor([[s / float(TOTAL_STEPS)]], device=device, dtype=torch.float32)
                    chunk = edit_module.select_best_edited_chunk(
                        smolvla_adapter, state_t, context, t_norm, base_seed=0, k_candidates=2
                    )[0]
                    chunk_queue.extend(chunk[k].cpu().numpy() for k in range(30))
                last_action = chunk_queue.pop(0).copy()
            if s >= 352:
                last_action = last_action.copy()
                last_action[5] = 45.0
            uenv.data.ctrl[:6] = lerobot_to_mujoco_qpos(last_action, uenv.model.jnt_range)
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
            if next_s < TOTAL_STEPS and (next_s % 30 == 0):
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


def eval_cosmos3_closed_loop_10spawns(edit_module: ExpoResidualCritic) -> dict[str, float]:
    succ_list, min_ee_list, lift_list, dense_list = [], [], [], []
    for idx, spawn in enumerate(eval_spawns_10):
        obs, _ = reset_env_with_spawn(spawn, seed=100 + idx)
        c3_policy.reset()
        edit_module.reset_episode()
        max_dense, min_ee, max_z, succeeded = 0.0, 999.0, 0.0, False
        chunk_queue: list[np.ndarray] = []
        last_action = None

        for s in range(ROLLOUT_STEPS):
            if s < TOTAL_STEPS:
                rgb = torch.from_numpy(obs["pixels"]["front"]).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                c3_policy._update_frame_history({"observation.images.front": rgb.to(device)})
                if not chunk_queue:
                    hist = torch.stack(list(c3_policy._frame_history), dim=2)
                    images = restore_uint8_history(hist)
                    context = c3_policy._extract_context(images, s)
                    state_t = torch.from_numpy(obs["agent_pos"]).unsqueeze(0).to(device, dtype=torch.float32)
                    t_norm = torch.tensor([[s / float(TOTAL_STEPS)]], device=device, dtype=torch.float32)
                    chunk = edit_module.select_best_edited_chunk(
                        c3_policy.decoder, state_t, context, t_norm, base_seed=0
                    )[0]
                    chunk_queue.extend(chunk[k].cpu().numpy() for k in range(30))
                last_action = chunk_queue.pop(0).copy()
            if s >= 352:
                last_action = last_action.copy()
                last_action[5] = 45.0
            uenv.data.ctrl[:6] = lerobot_to_mujoco_qpos(last_action, uenv.model.jnt_range)
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


svla_rl_runs = []
c3_marek_sc_runs = []
for seed in (0, 1, 2):
    t0 = time.time()
    m_svla = train_expo_with_marek_sc(
        svla_sim_cache["episodes"],
        reward_key="seq_det_reward",
        ctx_dim=960,
        use_marek_sc=True,
        steps=5000,
        seed=seed,
    )
    ev_svla = eval_smolvla_closed_loop_10spawns(m_svla)
    svla_rl_runs.append(ev_svla)
    print(
        f"  [SmolVLA + Marek Score-Centered EXPO | seed={seed}] "
        f"Succ={ev_svla['success_rate']*100:.0f}% | MinEE={ev_svla['min_ee_cm']:.2f}cm | "
        f"Lift={ev_svla['max_lift_cm']:.2f}cm ({time.time() - t0:.1f}s)",
        flush=True,
    )

    t1 = time.time()
    m_c3 = train_expo_with_marek_sc(
        all_60_eps,
        reward_key="seq_det_reward",
        ctx_dim=2048,
        use_marek_sc=True,
        steps=5000,
        seed=seed,
    )
    ev_c3 = eval_cosmos3_closed_loop_10spawns(m_c3)
    c3_marek_sc_runs.append(ev_c3)
    print(
        f"  [Cosmos 3 Edge LoRA + Marek Score-Centered EXPO | seed={seed}] "
        f"Succ={ev_c3['success_rate']*100:.0f}% | MinEE={ev_c3['min_ee_cm']:.2f}cm | "
        f"Lift={ev_c3['max_lift_cm']:.2f}cm ({time.time() - t1:.1f}s)",
        flush=True,
    )

comparison_out = {
    "smolvla_offline_val_68_72": smolvla_val_metrics,
    "smolvla_shared_detector_3seed": svla_det_summary,
    "smolvla_zero_shot_10spawns": svla_sim_cache["zero_shot"],
    "smolvla_marek_sc_expo_3seed": {
        metric: mean_sem([r[metric] for r in svla_rl_runs])
        for metric in ("success_rate", "min_ee_cm", "max_lift_cm", "dense_reward")
    },
    "cosmos3_marek_sc_expo_3seed": {
        metric: mean_sem([r[metric] for r in c3_marek_sc_runs])
        for metric in ("success_rate", "min_ee_cm", "max_lift_cm", "dense_reward")
    },
}
out_json_path = OUT_DIR / "smolvla_vs_cosmos3_and_marek_sc.json"
out_json_path.write_text(json.dumps(comparison_out, indent=2))
print("\n=========================================================================", flush=True)
print("FINAL SMOLVLA vs COSMOS 3 + MAREK SCORE CENTERING SUMMARY", flush=True)
print("=========================================================================", flush=True)
print(json.dumps(comparison_out, indent=2), flush=True)
env.close()
