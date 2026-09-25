# Plan: PEFT × LeRobot blog post (with Nemo, HF PEFT)

**Status:** planning, 2026-09-24. No experiments launched yet.
**Owner:** Anton. **Co-author:** Nemo (PEFT maintainer).
**Goal:** a Hugging Face–style blog post showing how to use 🤗 PEFT in LeRobot, with robot-learning examples that go beyond the existing `docs/source/peft_training.mdx` (SmolVLA LoRA on LIBERO).

## 1. Honest starting point

- LeRobot already integrates PEFT: `lerobot-train --peft.method_type=LORA --peft.r … --peft.lora_alpha …` (`TrainPipelineConfig.peft`, `pretrained.py`, `factory.py`, `lerobot_eval`, `rollout/context.py`).
- **The Video-VAM code does not use PEFT.** Five hand-written LoRA stacks: `policies/vam/cosmos_lora.py`, `cosmos3_lora.py`, `cosmos7b_lora.py`, `cosmos14b_lora.py`, `flux2_klein_lora.py` (+ `base/lora.py`). The alpha-mismatch bug found on 2026-09-24 (trained α=16, loaded α=32) is exactly the class of bug PEFT's `adapter_config.json` prevents.
- Therefore the post is also a **migration**: showing Nemo our custom code as a "PEFT example" would not land. Porting to PEFT is the first deliverable.

## 1b. FLUX 3 Action (released 2026-09-22) — likely the better centrepiece

- Open 7B world-action model; LeRobot-native (`lerobot.policies.flux3`, extra `flux3`, merged into this fork on 2026-09-25). Hub: `black-forest-labs/flux-3-action-{base,so101,droid}`. BFL thanks HF/LeRobot for "testing PEFT on SO-101" — ask Nemo what already exists.
- SO-101 recipe: `lerobot-train --config_path=lora.json` (rank/alpha 32, BF16, gradient checkpointing, batch 2 × accum 4, 10k microsteps = 2.5k updates; heads trained fully). ~200 demos per task in their examples.
- **Contract mismatch for us:** checkpoint expects **two cameras** (`scene` left + `wrist` right, 256×256 each → 512×256 canvas) at **30 Hz**, joint-delta actions + absolute gripper, predict 42 / execute 32. Our sort-cubes data has **only a wrist camera** (no scene camera). Options: (a) record a few episodes with an added scene camera; (b) feed the wrist view as the only camera (duplicate or blank the other half) and LoRA-adapt — degrades the pretrained prior, needs testing; (c) full fine-tune of heads + larger LoRA. (a) is cleanest for a blog post.
- **4090 feasibility (unverified):** BFL reports ~32 GB for BF16 _inference_ incl. Qwen3-VL-4B text encoder and VAE. Needs cached prompt embedding / text-encoder offload and probably FP8 to fit 24 GB. LoRA training memory not published. This measurement is itself useful blog content.

## 2. Candidate storylines (pick one primary)

| #   | Story                                                                                                                                                                                 | Why it is good                                                                                                                                                 | Risk / cost                                                                                                                                                                        |
| :-- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A   | **"Adapt a video world model to your robot with PEFT, then use it as a fast policy backbone."** Cosmos 3 Edge (3.4B) LoRA on robot videos → frozen feature tap → small action expert. | Novel for LeRobot users; showcases LoRA on a non-LLM DiT, `merge_and_unload()` for latency, adapter hot-swap. Already have working recipes and robot rollouts. | Needs PEFT to handle the diffusers `Cosmos3OmniTransformer` target modules (should: `target_modules` regex over `nn.Linear`).                                                      |
| B   | **"Make a slow video model fast with a LoRA student" (T=16→T=2 distillation).** Same weights, the LoRA _is_ the speed-up.                                                             | The most original contribution; strong "why PEFT" angle (distillation adapter vs full fine-tune).                                                              | Needs clean re-runs + seeds; Cosmos 3 Edge already fast (~80 ms), so motivation must be framed as a general technique. See [`plan_t2_distillation.md`](./plan_t2_distillation.md). |
| C   | **"PEFT the policy itself":** SmolVLA / action expert LoRA vs full expert fine-tune vs frozen, on a task that works on the robot (sort-cubes).                                        | Directly useful to every LeRobot user; uses the existing `--peft` CLI; cheap.                                                                                  | Less exciting alone; best as a section.                                                                                                                                            |
| D   | **Adapter zoo:** one backbone, per-task / per-dataset adapters swapped at deployment (v1 vs v2 cube, sort-cubes).                                                                     | Natural PEFT feature (multi-adapter); relevant to fleets.                                                                                                      | Needs ≥2 tasks with trained adapters.                                                                                                                                              |

**Recommendation:** primary **A** with **C** as the "start here" section, and **B** as a teaser / follow-up post (it is the part that becomes your own research post). Discuss with Nemo which angle they prefer.

## 3. Post outline (draft)

1. Why PEFT for robots: small data, big pretrained models, 24 GB GPU, many tasks.
2. Quick start: LoRA on the SmolVLA action expert with `lerobot-train --peft.*` on sort-cubes; params / memory / time / offline RMSE / robot success vs full expert fine-tune.
3. Beyond VLAs: LoRA on a video world model (Cosmos 3 Edge) with PEFT on robot videos; what the adapter changes (video prediction before/after — qualitative only).
4. Using the adapted model as a feature extractor for a policy; `merge_and_unload()` and compiled latency.
5. (Teaser) LoRA as a speed-up: distilling a long-context video forward into a 2-frame forward.
6. Results table + one robot clip; practical tips (targets, rank, alpha scaling, merging, saving with `save_pretrained`, loading with `PeftModel.from_pretrained`).

## 4. Engineering work (in order)

1. **PEFT port of the Cosmos 3 Edge video-LoRA** (`train_cosmos3_edge_video_lora.py`): replace `inject_cosmos3_lora` with `peft.LoraConfig(target_modules=…)` + `get_peft_model`; save via `save_pretrained`. Keep the old loader only to read existing adapters (or write a one-off converter to PEFT format).
2. **PEFT loading in the feature extractor** (`cosmos3_features.py`, `train_smolexpert.py --backbone-lora-weights` → accept a PEFT adapter dir; merge for inference).
3. **Equivalence test:** old adapter vs converted PEFT adapter produce identical features (bitwise or ≤1e-3) on a fixed batch.
4. **Cosmos-2B (T=2 distillation) port** only if storyline B is included.
5. Presets: `lora_cosmos3_edge_peft`, `smolvla_sort_cubes_peft_lora`.
6. Retire `cosmos7b_lora.py`, `cosmos14b_lora.py`, `flux2_klein_lora.py` or move them behind the PEFT path.

## 5. Experiments needed (not started)

| ID  | Experiment                                                                                                          | Metric                                                             | Purpose      |
| :-- | :------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------- | :----------- |
| P1  | SmolVLA sort-cubes: frozen-VLM expert-only (baseline recipe) vs `--peft` LoRA on expert (r 16/64) vs full fine-tune | params, peak mem, wall-clock, held-out RMSE (68–76), robot success | Section 2    |
| P2  | Cosmos 3 Edge video-LoRA with PEFT on sort-cubes videos                                                             | video val loss; equivalence to old adapter                         | Sections 3–4 |
| P3  | SmolExpert on PEFT-adapted Cosmos 3 Edge features vs base features (sort-cubes)                                     | held-out RMSE, robot success                                       | Section 4    |
| P4  | Latency: adapter-attached vs merged, eager vs compiled                                                              | ms / chunk on 4090                                                 | Section 4    |
| P5  | (optional) distillation LoRA                                                                                        | see distillation plan                                              | Section 5    |

Use **sort-cubes** as the demo task: it has a policy that works on the robot and a held-out split (68–76). All runs: 3 seeds for the headline comparison, pinned dataset revision.

## 6. Existing assets we can reuse

- Cosmos 3 Edge video-LoRA + SmolExpert on sort-cubes: 15.45 (BS 8), 14.81 (BS 64) held-out RMSE; runs on abakus (`outputs/train/sort-cubes-cosmos3-edge-*`).
- SmolVLA sort-cubes runs (baseline 25k, 100k-exact recipe, 200k compiled) on abakus.
- Cosmos 3 Edge video prediction renders (`outputs/evaluation/cosmos3_edge_*`), RPC rollout harness, training-efficiency plot scripts (`docs/blog/training_efficiency/`).

## 7. Open questions for Nemo

(Casual German message sent 2026-09-25: asked whether he was involved in the F3A × LeRobot SO-101 PEFT testing and proposed a simpler first post — LoRA on F3A or SmolVLA on SO-101 — instead of the distillation story.)

- Preferred angle (A/B/C/D)? Hugging Face blog vs PEFT docs example vs LeRobot docs?
- Any PEFT methods they want showcased beyond LoRA (e.g. DoRA, LoRA+ , VeRA, adapter merging)?
- Timeline / review process; can it include a Space or a Hub collection of adapters?
