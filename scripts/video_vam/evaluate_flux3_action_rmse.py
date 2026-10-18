"""Offline action RMSE of a FLUX 3 Action (flux3) policy on a wrist-only SO-101 dataset.

Anchors follow the sort-cubes leaderboard convention: held-out episodes, frames 4, 24, 44, ...
Predictions are absolute joint commands (checkpoint postprocessor integrates the deltas) scored on
the first 30 steps in stored units (5 joints in degrees + gripper 0-100), global masked aggregation.

Offline history: 8 image/state observations (clamped at the episode start, as in synchronous
deployment) and the 8 recorded commands before the anchor as ``observation.command_history``.

Memory on a 24 GB GPU: the policy is loaded on CPU, the task embedding is computed once with the
text encoder briefly on the GPU, then only the DiT and VAE stay on the GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.flux3 import Flux3Policy
from lerobot.policies.flux3.processor_flux3 import (
    ActionHistoryUnnormalizerProcessorStep,
    ActionTargetNormalizerProcessorStep,
    ObservationHistoryNormalizerProcessorStep,
    make_flux3_pre_post_processors_from_pretrained,
)

HORIZON = 30
ANCHOR_START = 4
SCENE = "observation.images.scene"


def parse_range(text: str) -> list[int]:
    lo, hi = (int(v) for v in text.split(":"))
    return list(range(lo, hi))


def load_table(root: Path) -> pd.DataFrame:
    files = sorted((root / "data").rglob("*.parquet"))
    cols = ["index", "episode_index", "frame_index", "action", "observation.state"]
    return pd.concat([pq.read_table(p, columns=cols).to_pandas() for p in files], ignore_index=True)


def train_quantiles(table: pd.DataFrame, episodes: list[int]) -> dict[str, torch.Tensor]:
    """q01/q99 of absolute states and consecutive command deltas (gripper absolute), train episodes only.

    Mirrors ``examples/flux3/export_base.py::compute_statistics`` (first delta of an episode is zero).
    """
    deltas, states = [], []
    for ep in episodes:
        rows = table[table.episode_index == ep].sort_values("frame_index")
        cmd = np.stack(rows["action"].to_numpy()).astype(np.float64)
        d = np.zeros_like(cmd)
        d[1:] = cmd[1:] - cmd[:-1]
        d[:, -1] = cmd[:, -1]
        deltas.append(d)
        states.append(np.stack(rows["observation.state"].to_numpy()).astype(np.float64))
    out = {}
    for stream, rows in (("action", np.concatenate(deltas)), ("state", np.concatenate(states))):
        out[f"{stream}.q01"] = torch.tensor(np.percentile(rows, 1, axis=0), dtype=torch.float32)
        out[f"{stream}.q99"] = torch.tensor(np.percentile(rows, 99, axis=0), dtype=torch.float32)
    return out


def recalibrate(pre, post, quantiles: dict[str, torch.Tensor], mode: str) -> None:
    """Swap checkpoint quantiles for dataset ones: ``state`` only, or ``all`` (state + action deltas)."""
    for step in [*pre.steps, *post.steps]:
        if not isinstance(
            step,
            ObservationHistoryNormalizerProcessorStep
            | ActionTargetNormalizerProcessorStep
            | ActionHistoryUnnormalizerProcessorStep,
        ):
            continue
        state = step.state_dict()
        for key in state:
            stream = key.split(".")[0]
            if mode == "all" or stream == "state":
                state[key] = quantiles[key]
        step.load_state_dict(state)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-path", default="black-forest-labs/flux-3-action-so101")
    p.add_argument("--dataset-repo-id", default="Orellius/so101_sort_cubes_no_top")
    p.add_argument("--dataset-revision", required=True)
    p.add_argument("--camera-key", default="observation.images.front", help="dataset wrist camera key")
    p.add_argument("--train-episodes", default="0:68")
    p.add_argument("--eval-episodes", default="68:77")
    p.add_argument("--stride", type=int, default=20)
    p.add_argument(
        "--scene-mode",
        choices=["duplicate", "black", "gray"],
        default="duplicate",
        help="what to feed the missing scene camera (W1 duplicate wrist / W2 blank)",
    )
    p.add_argument("--recalibrate", choices=["none", "state", "all"], default="none")
    p.add_argument("--task", default=None, help="override the dataset task string")
    p.add_argument("--num-inference-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--guidance-scale-action", type=float, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    # ---------------- policy (CPU load, then selective GPU placement)
    t0 = time.time()
    adapter = (Path(args.policy_path) / "adapter_config.json").is_file()
    base_path = args.policy_path
    if adapter:  # LoRA checkpoint: complete base + adapter (processors are saved with the adapter)
        from peft import PeftConfig, PeftModel

        base_path = PeftConfig.from_pretrained(args.policy_path).base_model_name_or_path
    config = PreTrainedConfig.from_pretrained(args.policy_path)
    config.device = "cpu"
    config.use_peft = False
    config.text_encoder_on_cpu = True
    if args.num_inference_steps is not None:
        config.num_inference_steps = args.num_inference_steps
    if args.guidance_scale is not None:
        config.guidance_scale = args.guidance_scale
    if args.guidance_scale_action is not None:
        config.guidance_scale_action = args.guidance_scale_action
    policy = Flux3Policy.from_pretrained(base_path, config=config).eval()
    if adapter:
        policy = PeftModel.from_pretrained(policy, args.policy_path).get_base_model().eval()
    pre, post = make_flux3_pre_post_processors_from_pretrained(
        policy.config,
        args.policy_path,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    load_s = time.time() - t0
    n_obs, chunk = policy.config.n_obs_steps, policy.config.chunk_size

    # ---------------- dataset
    probe = LeRobotDataset(
        args.dataset_repo_id, revision=args.dataset_revision, episodes=[0], download_videos=False
    )
    root = Path(probe.root)
    if probe.meta.fps != policy.config.fps:
        raise ValueError(f"dataset fps {probe.meta.fps} != policy fps {policy.config.fps}")
    table = load_table(root)
    train_eps, eval_eps = parse_range(args.train_episodes), parse_range(args.eval_episodes)
    if set(train_eps) & set(eval_eps):
        raise ValueError("train/eval episode overlap")
    if args.recalibrate != "none":
        recalibrate(pre, post, train_quantiles(table, train_eps), args.recalibrate)

    fps = probe.meta.fps
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        revision=args.dataset_revision,
        episodes=eval_eps,
        download_videos=False,
        delta_timestamps={
            args.camera_key: [i / fps for i in range(1 - n_obs, 1)],
            "observation.state": [i / fps for i in range(1 - n_obs, 1)],
            "action": [i / fps for i in range(-n_obs, HORIZON)],
        },
    )
    raw = dataset.reader.hf_dataset
    ep_col, fr_col = raw["episode_index"], raw["frame_index"]
    anchors = [
        i
        for i in range(len(raw))
        if int(fr_col[i]) >= ANCHOR_START and (int(fr_col[i]) - ANCHOR_START) % args.stride == 0
    ]
    if args.max_samples:
        anchors = anchors[: args.max_samples]

    # ---------------- DiT + VAE on the GPU; Qwen stays on CPU (DiT + VAE + Qwen do not fit 24 GB).
    # _context runs the encoder on its own device and caches the embedding on the GPU.
    task = args.task or dataset[anchors[0]]["task"]
    gib = 2**30
    policy.dit.to(device)
    dit_gib = torch.cuda.memory_allocated() / gib
    policy.frozen.video_vae.module.to(device)
    vae_gib = torch.cuda.memory_allocated() / gib - dit_gib
    t_text = time.time()
    policy._context(task, device)
    policy._context("", device)
    text_s = time.time() - t_text
    print(f"resident GPU: DiT {dit_gib:.2f} GiB, VAE {vae_gib:.2f} GiB; CPU text encode {text_s:.1f}s")
    torch.cuda.reset_peak_memory_stats()

    preds, targets, pads, states, last_commands, ids, latencies = [], [], [], [], [], [], []
    for n, i in enumerate(anchors):
        item = dataset[i]
        wrist = item[args.camera_key]  # (T, C, H, W) float [0, 1]
        if args.scene_mode == "duplicate":
            scene = wrist
        else:
            scene = torch.full_like(wrist, 0.0 if args.scene_mode == "black" else 0.5)
        actions = item["action"]  # (n_obs + HORIZON, 6)
        obs = {
            **{key: (scene if key == SCENE else wrist)[None] for key in policy.config.camera_order},
            "observation.state": item["observation.state"][None],
            "observation.command_history": actions[None, :n_obs],
            "task": [task],
        }
        pre.reset()
        post.reset()
        policy.reset()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            prepared = pre(obs)
            normalized = policy.predict_action_chunk(prepared)
            absolute = post(normalized)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t)
        if absolute.shape[1] != chunk:
            raise ValueError(f"unexpected chunk shape {tuple(absolute.shape)}")
        preds.append(absolute[0, :HORIZON].float().cpu())
        targets.append(actions[n_obs:].float())
        pads.append(item["action_is_pad"][n_obs:].bool())
        states.append(item["observation.state"][-1].float())
        last_commands.append(actions[n_obs - 1].float())
        ids.append(f"episode-{int(ep_col[i]):04d}-frame-{int(fr_col[i]):06d}")
        if n % 20 == 0:
            print(f"[{n + 1}/{len(anchors)}] {ids[-1]} {latencies[-1] * 1000:.0f} ms", flush=True)

    pred, tgt, pad = torch.stack(preds).double(), torch.stack(targets).double(), torch.stack(pads)
    valid = (~pad).unsqueeze(-1).double()

    def metrics(prediction: torch.Tensor) -> dict:
        sq = (prediction - tgt).square() * valid
        n_tok = valid.sum()
        per_step = [
            math.sqrt(float(sq[:, h].sum() / (valid[:, h].sum() * 6)))
            for h in range(HORIZON)
            if valid[:, h].sum() > 0
        ]
        return {
            "full30": math.sqrt(float(sq.sum() / (n_tok * 6))),
            "h1": per_step[0],
            "f5_mean": sum(per_step[:5]) / len(per_step[:5]),
            "per_joint": [math.sqrt(float(v)) for v in sq.sum((0, 1)) / n_tok],
            "arm_deg": math.sqrt(float(sq[..., :5].sum() / (n_tok * 5))),
            "gripper": math.sqrt(float(sq[..., 5].sum() / n_tok)),
        }

    state_repeat = torch.stack(states).double()[:, None].expand_as(tgt)

    # F3A conditions on the recorded command history, so hold-last-command is its fair trivial baseline.
    command_repeat = torch.stack(last_commands).double()[:, None].expand_as(tgt)
    result = {
        "policy_path": args.policy_path,
        "base_path": base_path,
        "camera_keys": policy.config.camera_order,
        "dataset": f"{args.dataset_repo_id}@{args.dataset_revision}",
        "eval_episodes": eval_eps,
        "anchors": {"start": ANCHOR_START, "stride": args.stride, "count": len(anchors)},
        "scene_mode": args.scene_mode,
        "recalibrate": args.recalibrate,
        "task": task,
        "sampling": {
            "num_inference_steps": policy.config.num_inference_steps,
            "guidance_scale": policy.config.guidance_scale,
            "guidance_scale_action": policy.config.guidance_scale_action,
            "seed": policy.config.inference_seed,
        },
        "flux3": metrics(pred),
        "state_repeat": metrics(state_repeat),
        "command_repeat": metrics(command_repeat),
        "latency_ms": {
            "median": 1000 * float(np.median(latencies[1:] or latencies)),
            "first": 1000 * latencies[0],
        },
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / gib,
        "resident_gib": {"dit": dit_gib, "vae": vae_gib},
        "cpu_text_encode_s": text_s,
        "load_s": load_s,
        "sample_ids": ids,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    torch.save(
        {"pred": pred.float(), "target": tgt.float(), "pad": pad, "ids": ids}, args.output.with_suffix(".pt")
    )
    summary = {
        k: result[k] for k in ("scene_mode", "recalibrate", "latency_ms", "peak_gpu_gib", "resident_gib")
    }
    summary["flux3"] = {k: round(v, 2) for k, v in result["flux3"].items() if k != "per_joint"}
    for name in ("state_repeat", "command_repeat"):
        summary[name] = {k: round(result[name][k], 2) for k in ("full30", "h1", "f5_mean")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
