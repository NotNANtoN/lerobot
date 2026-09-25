# Plan: FLUX 3 Action (F3A) on our SO-101 with wrist-only camera data

**Status:** planning, 2026-09-25. Upstream LeRobot merged into this fork on 2026-09-25 (`lerobot.policies.flux3`, extra `flux3`). Nothing run yet.
**Why:** strongest open policy that has already been fine-tuned on SO-101; natural centrepiece for the PEFT blog ([plan](./plan_peft_blog.md)) and the new baseline for "a good feature set" ([status §0](./video_vam_status.md#0-goals-stated-by-anton-2026-09-24)).

## 1. Key facts (from BFL report, model card, docs, `docs/source/flux3.mdx`)

| Item                             | Value                                                                                                                                                                                                                                                                                                                                       |
| :------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Model                            | 7B world-action model (joint video + action prediction), FLUX 3 backbone; Qwen3-VL-4B text encoder (~8.3 GiB) + video VAE (needs NATTEN)                                                                                                                                                                                                    |
| Hub                              | `black-forest-labs/flux-3-action-base` (trunk + encoders), `-so101` (trained on `lerobot/community_dataset_v3` SO-101 episodes), `-droid`. License: FLUX Kommunity (non-commercial).                                                                                                                                                        |
| SO-101 checkpoint contract       | cameras `observation.images.scene` (left) + `observation.images.wrist` (right), each 256×256 → 512×256 canvas (`camera_layout="side_by_side"`); **30 Hz**; 6-D joint-delta actions + absolute gripper; 8 observation timesteps, 2 visual snapshots, past-command conditioning; predict 42, execute 32 (~1.07 s); 4 Euler steps, guidance 3. |
| LoRA recipe                      | `examples/flux3/lora.json` (also bundled as `lora.json`): rank/alpha 32, BF16, gradient checkpointing, adapter LR 1e-4 / heads 5e-4, EMA 0.999, batch 2 × accum 4, `steps=10000` microsteps = 2,500 updates, last 20 % of episodes per task held out. Heads train fully; trunk via LoRA.                                                    |
| Rollout                          | synchronous only (`lerobot-rollout --strategy.type=base --inference.type=sync`); **no RTC / async**. Postprocessor integrates deltas; reset processors per episode.                                                                                                                                                                         |
| Reported results                 | RoboLab 38–42 % (SOTA open); Franka blind third-party 28/30; SO-101 demos with ~200 teleop episodes per task, OOD object/container/camera generalization shown.                                                                                                                                                                             |
| Memory                           | BF16 **inference ~32 GB** reported (incl. encoders). Qwen can be moved to CPU after loading (`policy.frozen.text_encoder.cpu()`, −8.3 GiB, not peak during load). FP8 serving ~1.4× faster on local GPUs. LoRA training memory not published.                                                                                               |
| Validation status (upstream doc) | CPU tests only on this exact revision; "still needs GPU and robot validation".                                                                                                                                                                                                                                                              |

## 2. Our constraint: wrist camera only

Our SO-101 datasets (cube-out-of-box v1/v2 at 10 fps; sort-cubes at 30 fps) have **only a wrist camera, no scene camera**. The checkpoint was trained with scene + wrist. Options, cheapest first:

| #   | Approach                                                                      | How                                                                                                                                                | Pros                                                                  | Cons                                                                                                                               |
| :-- | :---------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------- |
| W1  | **Wrist duplicated into both slots**                                          | `--rename_map` maps our wrist key to `scene` _and_ duplicate stream to `wrist` (needs a small processor step or dataset view), keep `side_by_side` | No architecture change; LoRA + heads adapt                            | Distribution shift on the left half; model may rely on scene view for global position                                              |
| W2  | **Blank scene slot**                                                          | feed a constant (black / mean) image as `scene`                                                                                                    | Honest "missing camera" signal                                        | Same shift; model trained never seeing blanks                                                                                      |
| W3  | **`camera_layout="single"`, wrist only, `canvas_hw` e.g. 256×256 or 512×512** | new config + processors; trunk LoRA + heads                                                                                                        | Clean single-camera setup, supported layout in the integration        | Changes packing/canvas vs. checkpoint → larger adaptation, loses more of the SO-101 prior; must package config+processors properly |
| W4  | **Add a fixed scene camera** and record 50–200 episodes                       | hardware + teleop                                                                                                                                  | Matches pretrained contract; best expected quality; best for the blog | Needs new data                                                                                                                     |

Plan: try **W1 → W3** on sort-cubes (30 Hz matches) as a feasibility study; do **W4** in parallel if a camera is available. Compare against the old Orellius SmolVLA policy on the robot.

Frame-rate note: cube-out-of-box is 10 fps; F3A-SO101 expects 30 Hz actions. Use sort-cubes first; for cube-out-of-box either resample (no new information) or record at 30 Hz.

## 3. 4090 feasibility plan (measure, don't assume)

1. **Inference smoke test** with the released SO-101 checkpoint on recorded frames: BF16, text encoder moved to CPU after load; record peak memory and latency. If OOM during load, load on CPU and move trunk only.
2. If still too large: FP8 serving path from the flux-action repo; or cache the fixed task embedding and drop Qwen entirely.
3. **LoRA training smoke test**: `lora.json` with `--batch_size=1`, accumulation 8, 4 optimizer updates, log `peak_mem`. Fallbacks: cached text embeddings, smaller canvas (W3), 8-bit optimizer for heads, gradient checkpointing already on.
4. Record every number in the leaderboard / PEFT plan — "does a 7B WAM LoRA fit a 24 GB consumer GPU, and how" is itself blog content.

## 4. Environment on abakus

```bash
uv sync --extra flux3 --extra peft --extra training   # plus existing extras
uv pip install natten==0.21.6+torch2110cu128 --find-links https://whl.natten.org   # match torch/CUDA
hf auth login   # model access
```

## 5. Experiments (proposed, not started)

| ID  | Experiment                                                                                   | Output                                                      |
| :-- | :------------------------------------------------------------------------------------------- | :---------------------------------------------------------- |
| F0  | Inference smoke test (memory/latency) on 4090                                                | numbers for §3                                              |
| F1  | Zero-shot F3A-SO101 on sort-cubes frames (W1) — offline action error on held-out eps 68–76   | does the prior transfer at all                              |
| F2  | LoRA W1 on sort-cubes train eps 0–67                                                         | held-out RMSE (same metric as leaderboard §3), robot trials |
| F3  | LoRA W3 (single-camera layout)                                                               | compare with F2                                             |
| F4  | (if camera added) LoRA on scene+wrist recordings                                             | best-case reference                                         |
| F5  | Robot comparison: F3A-LoRA vs retrained SmolVLA vs Cosmos 3 Edge BS64 vs old Orellius policy | first real "feature set" answer + R2 data point             |

## 6. Open questions

- Nemo / HF: what did the "PEFT on SO-101" testing cover, which GPU, any wrist-only experience? (message sent 2026-09-25)
- Does the checkpoint rely heavily on the scene view (probe: zero-shot error with scene slot blanked vs duplicated)?
- Delta-action integration + synchronous execution vs. our RTC harness: extend `run_mac_vam_rpc.py` or use `lerobot-rollout` directly?
