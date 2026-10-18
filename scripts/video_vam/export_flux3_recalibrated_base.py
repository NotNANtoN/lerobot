"""Copy a complete flux3 policy package and replace its normalization quantiles with dataset ones.

The SO-101 checkpoint's quantiles encode its own robots' calibration (e.g. shoulder_lift state
85..184 vs -88..51 on Orellius/so101_sort_cubes_no_top), so states of another calibration are
clipped at +-6. docs/source/flux3.mdx: "To use different statistics, prepare a separate base with
matching processors before training." This does exactly that without touching the weights.

Statistics come from the training episodes only (state: absolute; action: consecutive command
deltas with the absolute gripper), matching examples/flux3/export_base.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file, save_file

from lerobot.datasets.lerobot_dataset import LeRobotDataset

STATE_FILES = {
    "policy_preprocessor_step_3_flux3_observation_history_normalizer.safetensors": ("state", "action"),
    "policy_preprocessor_step_4_flux3_action_target_normalizer.safetensors": ("action",),
    "policy_postprocessor_step_0_flux3_action_history_unnormalizer.safetensors": ("action",),
}


def parse_range(text: str) -> list[int]:
    lo, hi = (int(v) for v in text.split(":"))
    return list(range(lo, hi))


def quantiles(root: Path, episodes: list[int], absolute_dims: list[int]) -> dict[str, dict[str, list[float]]]:
    cols = ["episode_index", "frame_index", "action", "observation.state"]
    table = pd.concat(
        [pq.read_table(p, columns=cols).to_pandas() for p in sorted((root / "data").rglob("*.parquet"))]
    )
    deltas, states = [], []
    for ep in episodes:
        rows = table[table.episode_index == ep].sort_values("frame_index")
        if rows.empty:
            raise ValueError(f"episode {ep} missing")
        cmd = np.stack(rows["action"].to_numpy()).astype(np.float64)
        d = np.zeros_like(cmd)
        d[1:] = cmd[1:] - cmd[:-1]
        d[:, absolute_dims] = cmd[:, absolute_dims]
        deltas.append(d)
        states.append(np.stack(rows["observation.state"].to_numpy()).astype(np.float64))
    return {
        stream: {
            "q01": np.percentile(values, 1, axis=0).tolist(),
            "q99": np.percentile(values, 99, axis=0).tolist(),
        }
        for stream, values in (("action", np.concatenate(deltas)), ("state", np.concatenate(states)))
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-path", required=True, help="local directory of a complete flux3 package")
    p.add_argument("--dataset-repo-id", required=True)
    p.add_argument("--dataset-revision", required=True)
    p.add_argument("--train-episodes", required=True, help="half-open range, e.g. 0:68")
    p.add_argument("--streams", choices=["state", "all"], default="all")
    p.add_argument("--fps", type=float, default=None, help="override control fps (e.g. 10 for cube v1)")
    p.add_argument(
        "--video-position-fps",
        type=float,
        default=None,
        help="override video position fps (default with --fps: keep the checkpoint's video/control ratio)",
    )
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--n-action-steps", type=int, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    src = Path(args.policy_path)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads((src / "config.json").read_text())
    ds = LeRobotDataset(
        args.dataset_repo_id, revision=args.dataset_revision, episodes=[0], download_videos=False
    )
    episodes = parse_range(args.train_episodes)
    new = quantiles(Path(ds.root), episodes, config["delta_absolute_dims"])
    replace = ("state", "action") if args.streams == "all" else ("state",)

    # Contract overrides touch config.json and the processor configs (history steps check chunk_size).
    overrides: dict[str, float | int] = {}
    if args.fps is not None:
        overrides["fps"] = args.fps
        ratio = config["video_position_fps"] / config["fps"] if config.get("video_position_fps") else None
        vfps = args.video_position_fps or (args.fps * ratio if ratio else None)
        if vfps is not None:
            overrides["video_position_fps"] = vfps
    if args.chunk_size is not None:
        overrides["chunk_size"] = args.chunk_size
    if args.n_action_steps is not None:
        overrides["n_action_steps"] = args.n_action_steps
    rewritten = {"config.json", "policy_preprocessor.json"} if overrides else set()

    # Unchanged files (incl. the 14 GB trunk) become absolute symlinks into the source package.
    args.output.mkdir(parents=True)
    for path in src.iterdir():
        if (
            path.is_file()
            and path.name not in STATE_FILES
            and path.name not in rewritten
            and path.name != "dataset_statistics.json"
        ):
            (args.output / path.name).symlink_to(path.resolve())
    if overrides:
        if "n_action_steps" not in overrides and config["n_action_steps"] > overrides.get(
            "chunk_size", 10**9
        ):
            overrides["n_action_steps"] = overrides["chunk_size"]
        config.update(overrides)
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        pre = json.loads((src / "policy_preprocessor.json").read_text())
        for step in pre["steps"]:
            if "chunk_size" in step.get("config", {}) and "chunk_size" in overrides:
                step["config"]["chunk_size"] = overrides["chunk_size"]
        (args.output / "policy_preprocessor.json").write_text(json.dumps(pre, indent=2) + "\n")
    for name, streams in STATE_FILES.items():
        state = load_file(src / name)
        for stream in streams:
            if stream in replace:
                for q in ("q01", "q99"):
                    old = state[f"{stream}.{q}"]
                    state[f"{stream}.{q}"] = torch.tensor(new[stream][q], dtype=old.dtype)
        save_file(state, args.output / name)
    stats = json.loads((src / "dataset_statistics.json").read_text())
    for stream in replace:
        stats[stream].update(new[stream])
    (args.output / "dataset_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    (args.output / "normalization_source.json").write_text(
        json.dumps(
            {
                "base": str(src),
                "dataset": f"{args.dataset_repo_id}@{args.dataset_revision}",
                "training_episodes": episodes,
                "replaced_streams": list(replace),
                "config_overrides": overrides,
                "quantiles": new,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"output": str(args.output), "replaced": list(replace), "overrides": overrides, **new}))


if __name__ == "__main__":
    main()
