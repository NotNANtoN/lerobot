# Plan: FLUX 3 Action (F3A) on our SO-101 with wrist-only camera data

**Status:** F0, F1, F2 (LoRA W1) done on sort-cubes and cube v1 (§7, 2026-09-26) — 2026-09-25. Results in [§7](#7-results-2026-09-25). Upstream LeRobot merged into this fork on 2026-09-25 (`lerobot.policies.flux3`, extra `flux3`).
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

## 7. Results (2026-09-25)

### Environment on abakus (what actually worked)

- Repos were **public** on 2026-09-25 when downloaded to `abakus`, and switched to **gated (`gated: auto`)** on 2026-10-14 (`0713a7af`). `abakus` already has the complete snapshots cached (`c9e13b2a…`, 13.9 GB `model.safetensors`, and base `62878e29…`, `video_vae.safetensors` + `text_encoder/`, ~11 GB). `hf download … --exclude assets/*` fails on a missing `assets/so101.urdf`; use `snapshot_download(allow_patterns=…)`.
- Offline loading: pass the **local snapshot path** as `--policy-path` (downloads at a pinned revision have no `refs/main`, so `HF_HUB_OFFLINE=1` + repo id fails).
- `uv sync --locked` currently **fails** building `transformer-engine-torch==2.19.0` from sdist (no nvcc). The existing venv (torch 2.11.0+cu128, TE 2.18, peft 0.21, transformers 5.14.1) + `natten==0.21.6+torch2110cu128` works, but only with `source scripts/video_vam/cosmos_cuda_env.sh` (TE otherwise fails to find `libcudart` because both cu12 and cu13 TE wheels are installed; peft imports TE).

### F0 — memory / latency on the 4090 (BF16, eager, 4 Euler steps, guidance 3 → 2 passes/step)

| Component                 | GPU                                                                                    |
| :------------------------ | :------------------------------------------------------------------------------------- |
| DiT (image/audio streams filtered) | 12.94 GiB                                                                     |
| Video VAE                 | 2.34 GiB                                                                               |
| Qwen3-VL-4B text encoder  | ~8.3 GiB → **does not fit together** (OOM at 23.4 GiB). Kept on CPU: 60 s one-off per caption, then cached |
| Peak during inference     | **15.8 GiB**                                                                           |
| Latency per 42-action chunk | **~1.26 s** median vs 1.07 s of motion per 32 executed actions at 30 Hz → RTF ≈ 1.18, i.e. slightly slower than real time with synchronous execution (no FP8 / compile / guidance-off tried yet; guidance 3 doubles the DiT passes) |

Added `Flux3Config.text_encoder_on_cpu` (keeps Qwen on CPU through `policy.to(...)`), needed for training on 24 GB.

### F1 — zero-shot on sort-cubes held-out eps 68–76 (405 anchors, stride 20, offset 4)

Metric as leaderboard §3 (Full-30 / H1 / F5 mean, mixed units). `command_repeat` = hold the last recorded command (F3A sees the command history, so this is its fair trivial baseline).

| Variant                                         | Full-30 |   H1 |   F5 | Notes                                                      |
| :---------------------------------------------- | ------: | ---: | ---: | :--------------------------------------------------------- |
| W1 duplicate wrist, checkpoint quantiles        |   23.71 | 1.18 | 3.53 | states of our calibration are clipped (shoulder_lift ckpt 85..184 vs ours −98..80) |
| W1 duplicate wrist, **state quantiles from train eps** |   19.24 | 0.51 | 2.14 | ≈ command-repeat on Full-30, better short-horizon          |
| W2 black scene, state quantiles                 |   19.33 | 0.47 | 2.03 | same as W1 → zero-shot the model barely uses the (fake) scene slot |
| W1, state **and** action-delta quantiles        |   19.31 | 0.50 | 2.13 | action-delta ranges already similar (±4 vs ±3.5–5)         |
| command_repeat                                  |   19.69 | 1.40 | 4.11 |                                                            |
| state_repeat                                    |   22.38 |    — |    — |                                                            |

Reading: zero-shot the prior gives sensible short-horizon continuation but no task knowledge (different calibration, task, and camera). Fine-tuning is required; for reference Cosmos 3 Edge BS8 reaches 15.45 and SmolVLA ~17.3 on the same split.

### F2 — LoRA on a 24 GB 4090 (feasible, measured)

Preset `f3a_sort_cubes_lora_w1` = BFL `lora.json` unchanged (r/alpha 32, BF16, grad checkpointing, batch 2 × accum 4, 10k microsteps = 2.5k updates, EMA 0.999) + recalibrated base + `--policy.text_encoder_on_cpu=true` + wrist duplicated via `--policy.camera_keys` + `--rename_map`.

| Item                          | Value                                                        |
| :---------------------------- | :----------------------------------------------------------- |
| Trainable params              | 137 M of 7.08 B (LoRA on q/k/v/attn_out/mlp_in/mlp_out + full action heads) |
| GPU memory (torch allocated)  | 20–21 GB; nvidia-smi 23.8–24.0 GB of 24.5 → **fits, ~no headroom** |
| Speed                         | 1.87 s / microstep (batch 2) → 10k microsteps ≈ 5.1 h        |
| Checkpoint                    | adapter + heads + processors ≈ 1.1 GB each (raw and EMA)     |
| Offline eval of adapter       | evaluator loads base + `PeftModel` (peak 16.4 GiB)           |
| Wall-clock                    | 5 h 15 min for 10k microsteps (0.43 epochs of the 46.8k train frames) |

**Held-out results (eps 68–76, 405 anchors, same evaluator as F1):**

| Checkpoint (updates)        | Full-30 |   H1 |   F5 | arm ° | gripper |
| :-------------------------- | ------: | ---: | ---: | ----: | ------: |
| zero-shot (recalibrated)    |   19.24 | 0.51 | 2.14 | 20.75 |    8.31 |
| EMA @ 625 (2.5k microsteps) |   16.57 | 0.52 | 1.90 | 17.83 |    7.59 |
| EMA @ 1250                  |   14.82 | 0.48 | 1.65 | 15.92 |    7.18 |
| **EMA @ 1875**              | **14.20** | 0.47 | 1.56 | 15.21 |    7.33 |
| EMA @ 2500 (final)          |   14.54 | 0.45 | 1.55 | 15.59 |    7.35 |
| raw @ 2500 (no EMA)         |   14.73 | 0.43 | 1.56 | 15.82 |    7.11 |

- Better than every previous sort-cubes number on Full-30 (Cosmos 3 Edge BS 64: 14.81, BS 8: 15.45, SmolVLA ~17.3), with **wrist-only input, BFL's recipe unchanged and 5 h on one 4090**. Caveat: single seed; differences < ~0.5 are within noise (see status pitfalls), so "on par with the best Cosmos run" is the honest reading, H1/F5 are not comparable (F3A sees the command history).
- Plateau after ~1.2k updates; EMA helps slightly at the end (14.54 vs 14.73).
- Inference with the **unmerged** adapter: 1.84 s / chunk (vs 1.26 s base) → merge the adapter for deployment (P4 in the PEFT plan).
- Next: robot trial (synchronous `lerobot-rollout`, recalibrated base + adapter, wrist key mapped to both slots), W3 single-camera layout, seed repeat.

### F2b — cube-out-of-box v1 (10 Hz), Protocol 1.0 (2026-09-26)

Preset `f3a_cube_v1_lora_w1`; base `outputs/bases/flux3-so101-cubev1-10hz` = SO-101 checkpoint with cube train-episode quantiles and a **10 Hz contract** (`fps 10`, video positions 8 fps = same 30:24 ratio, chunk 32 = 3.2 s, execute 10; `export_flux3_recalibrated_base.py --fps 10 --chunk-size 32 --n-action-steps 10`). Same LoRA recipe (2.5k updates, 4 h 10 min at 1.47 s/microstep). 88 anchors on val eps 32–39.

| Checkpoint              | Full-30 |   H1 |   F5 | per joint (pan, lift, elbow, wrist flex, roll, grip) |
| :---------------------- | ------: | ---: | ---: | :--------------------------------------------------- |
| zero-shot               |   30.40 | 0.82 | 3.46 | 58.4, 29.0, 20.0, 24.8, 12.9, 10.3                   |
| EMA @ 625               |   26.39 | 0.80 | 3.27 | 36.3, 42.6, 17.6, 21.2, 13.0, 11.0                   |
| EMA @ 1250              |   19.96 | 0.81 | 3.32 |                                                      |
| EMA @ 1875              |   17.37 | 0.85 | 3.43 |                                                      |
| **EMA @ 2500**          | **15.74** | 0.89 | 3.44 | 22.1, 14.7, 12.2, 17.4, 12.5, 13.1                  |
| command_repeat          |   18.19 | 1.53 | 4.46 |                                                      |

Reading: on cube v1 F3A is **clearly behind** the video-feature policies (Cosmos 3 Edge online 13.73–13.82, Cosmos-2B 13.06–13.65, SmolVLA 1 h 14.83) and still improving when the recipe ends — the 10 Hz retiming moves it away from its 30 Hz pretraining, there are only 32 episodes (~3.6k frames vs 46.8k on sort-cubes), and shoulder_pan dominates the error (22°). A longer schedule (e.g. 5k updates) is the obvious next run; resampling cube to 30 Hz would not add information.

Artifacts: `outputs/flux3_eval/f1_*.json|.pt` on abakus; evaluator `scripts/video_vam/evaluate_flux3_action_rmse.py`; recalibrated base `outputs/bases/flux3-so101-sortcubes-recal` (`scripts/video_vam/export_flux3_recalibrated_base.py`, weights symlinked).

## 6. Open questions

- Nemo / HF: what did the "PEFT on SO-101" testing cover, which GPU, any wrist-only experience? (message sent 2026-09-25)
- Does the checkpoint rely heavily on the scene view (probe: zero-shot error with scene slot blanked vs duplicated)?
- Delta-action integration + synchronous execution vs. our RTC harness: extend `run_mac_vam_rpc.py` or use `lerobot-rollout` directly?
