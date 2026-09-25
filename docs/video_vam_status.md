# Video-VAM status

**Last updated:** 2026-09-24. Supersedes `archive/video_vam_roadmap.md`, `archive/video_vam_technical_report.md`, `archive/EXPERIMENTS_OVERVIEW.md` and `archive/video_vam_execution_plan.md`.
Numbers here are summaries; the [leaderboard](./video_vam_leaderboard.md) is authoritative.

## 0. Goals (stated by Anton, 2026-09-24)

1. **Publish.** (a) Joint blog post with Nemo (Hugging Face PEFT maintainer) on using PEFT in LeRobot; the video-LoRA / fast-feature work is a candidate example. (b) Own blog posts / Twitter threads (existing drafts not yet satisfactory).
2. **A good baseline on a good feature set:** a reliable policy + a defensible choice of visual features to build on.
3. **Later:** post-training; learn how to train with features in little time and how to train a real robot safely without simulation.

Context: a SmolVLA policy on sort-cubes worked well on the robot (Nov 2025, `Orellius/so101_sort_cubes_no_top_smolvla_base_100k`). On cube-out-of-box no policy has worked yet, even with more data; the cause is unknown. Note: that working policy was _not_ a PEFT or full-VLM fine-tune — it was the standard SmolVLA recipe (frozen vision, action expert trained), BS 32, 200k-step schedule, image transforms, 30 fps data.

Plans: [PEFT blog](./plan_peft_blog.md) · [T=2 distillation](./plan_t2_distillation.md) · [research direction, venues, BFL side goal](./plan_research_direction.md).

## 1. One-paragraph summary

Frozen or video-LoRA-adapted video DiT features (Cosmos-Predict2-2B, Cosmos 3 Edge, LTX-2.5; Cosmos 7B/14B and FLUX.2 klein to a lesser degree) feed a SmolVLA-style flow-matching action expert ("SmolExpert"). On the cube-out-of-box v1 benchmark the best offline numbers (~13.1 mixed RMSE) beat a 1-hour SmolVLA (14.83) by roughly 1.5 units, but every comparison is **single-seed** on 88 anchors from 8 episodes, and none of the policies achieved a reliable physical grasp. Cosmos 3 Edge (600 tokens, ~80–92 ms extraction) is the most practical backbone. On 2026-09-24 three pipeline bugs were found that invalidate several post-audit online results (see §4).

## 2. Datasets

| Name               | Repo @ revision                                     | Episodes / frames                                               | Status                                                                                                                                                                                                                                                              |
| :----------------- | :-------------------------------------------------- | :-------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **v1** (canonical) | `hubnemo/cube_out_of_box_dataset` @ `243370c3…`     | 40 / 6,536 @ 10 fps                                             | Protocol 1.0: train 0–31, val 32–39 (88 anchors, stride 20)                                                                                                                                                                                                         |
| **v2 / Scale-100** | `Orellius/cube_out_of_box_v2` @ `5d0325cc…`         | metadata says 100 / 12,163; files contain 140 eps / 15,998 rows | **Integrity unresolved.** The 09-08 quarantine was never formally lifted, but all Scale-100 and Phase 6 runs used this snapshot (train 0–31 + 40–89, Eval-1 32–39, Eval-2 90–99). Treat Scale-100 numbers as provisional until the extra 40 episodes are explained. |
| **sort-cubes**     | `Orellius/so101_sort_cubes_no_top` @ _not recorded_ | 77 eps @ 30 fps                                                 | Held out 68–76. Different task and dataset: **not comparable** to cube numbers. Runs before 09-24 did not pin a revision; the trainer now requires `--dataset-revision`.                                                                                            |

Dataset shift inside v1: early vs late episodes differ strongly in joint posture and pace (two recording sessions/operators or a recalibration; the exact cause and changepoint, ~ep 20 vs ~ep 24, are **not established**). Val 32–39 lies entirely in the late regime, so v1 validation measures cross-session generalization.

## 3. Current best results (single seed, see leaderboard for full list)

| Benchmark                                   | Best VAM                                                                                                         | SmolVLA                                    | Trivial            |
| :------------------------------------------ | :--------------------------------------------------------------------------------------------------------------- | :----------------------------------------- | :----------------- |
| v1 Protocol 1.0 (32→8 eps)                  | 13.06 Cosmos-2B video-LoRA T=16 pool2 (~1.2 s); 13.15 T=2 distilled (~204 ms); 14.26 Cosmos 3 Edge base (~92 ms) | 14.83 (1 h)                                | state-repeat 18.86 |
| Scale-100 Eval-1 / Eval-2 (provisional, §2) | 13.49 / 17.20 Cosmos 3 Edge video-LoRA                                                                           | — (v2 SmolVLA not scored on this protocol) | —                  |
| sort-cubes held-out 68–76                   | 15.45 Cosmos 3 Edge (BS 8, local artifact); 14.81 BS 64 (**no local artifact; unverified**)                      | 17.27 @50k                                 | —                  |

Caveats that apply to every row: single seed; no confidence interval; 0.1–0.8 differences are likely within noise; full-30 RMSE is dominated by far-horizon error that RTC never executes; the T=2 undistilled and distilled heads were re-trained on 09-09 (13.65 / 13.48 — the gap shrank from 0.59 to 0.16); only the teacher-reference head is still missing.

## 4. Open correctness issues (found 2026-09-24, fixed in code, runs not yet redone)

| #   | Bug                                                                                                                                                                                                                                                                                                                                                                                                         | Affected results                                                                                                                                                    | Fix                                                                                                                                             |
| :-- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------- |
| B1  | Online FLUX.2 path in `train_smolexpert.py` still used bilinear 32×32 **pseudo-latents** (the defect retracted on 09-07). `Flux2KleinExtractor.encode_latents` silently returned RGB when no VAE was configured.                                                                                                                                                                                            | Phase 6 FLUX.2 online-aug (18.42 / 21.59) **invalid**.                                                                                                              | Online FLUX now requires `--backbone-vae` and uses the native VAE with the cache builder's conditioning; `encode_latents` raises without a VAE. |
| B2  | Cosmos-2B step-6000 video-LoRA was trained with **alpha 16** but online runs injected it with **alpha 32**, doubling the adapter delta. The offline cache builder read alpha 16 from the sidecar, so online training features ≠ offline eval features. `v1-cosmos2b-t2-online-unaugmented` (14.52) **invalid**. (Phase 6 Cosmos-2B online-aug used α 16 / σ 10 consistently with its cache — not affected.) | Rank/alpha are read from the adapter metadata/sidecar; a conflicting CLI value is an error; `load_lora_state_dict` rejects wrappers that disagree with the sidecar. |
| B3  | Online non-Cosmos-3 runs were scored on offline caches with no identity check (HEAD also hard-coded `high_noise_sigma=10` online vs caches at 80, and auto-discovered caches from other runs).                                                                                                                                                                                                              | Latent risk for any online run; confirmed only for the B2 run.                                                                                                      | `validate_online_eval_cache_identity` checks LoRA hash/rank/alpha, sigma, layer and augmentation; cross-run cache auto-discovery removed.       |
| B4  | Retracted leaky scripts (`train_smolexpert_on_{cosmos7b,cosmos14b,flux2_klein}.py`, `extract_*`, 7B/FLUX rollout renderers) were still runnable. The 14B "rollout" renderer used a single-step heuristic update, not a sampler.                                                                                                                                                                             | 7B/FLUX PSNR numbers and the 14B base-vs-QLoRA rollout comparison (identical 18.28 dB / 0.7803) **invalid**.                                                        | Scripts deleted (recoverable from git history).                                                                                                 |
| B5  | Uncommitted change silently lowered Euler steps from 10 to 3 for every checkpoint loaded by `VideoVAMPolicy` / `SmolExpertActionDecoder`.                                                                                                                                                                                                                                                                   | Robot rollouts after 09-17 may have used 3 steps without it being recorded.                                                                                         | Default back to checkpoint value (10); explicit `VideoVAMConfig.action_euler_steps` and `rpc_server.py --euler-steps`.                          |

Still open (not fixed by code changes):

- Cosmos 3 Edge gen/und tower mismatch, latent normalization, FPS conditioning (audit §2) — current extractor routes vision through `gen_seq`; results produced before that fix (e.g. 14.26 base) predate it.
- Cosmos-1.0 EDM vs rectified-flow objective mismatch in 7B/14B LoRA trainers (audit §3.3).
- Scale-100 FLUX.2 caches (built 09-07) record no VAE hash, so real-VAE extraction cannot be confirmed; FLUX Scale-100 numbers are `UNVERIFIED`. Rebuild with preset `cache_flux2_klein_scale100` if FLUX matters.
- v2 dataset integrity (§2).

## 5. What was done wrong (process lessons)

1. **No noise floor.** Decisions were made on single-seed deltas smaller than plausible seed variance.
2. **Offline metric ≠ robot success.** Best offline model still never grasped; grasp depends on the first steps and gripper timing, which full-30 RMSE barely weights.
3. **Unmatched baselines.** VAM arms get video-LoRA, more compute and augmentation; SmolVLA got 1 hour and no augmentation. "Video features beat SigLIP" is not isolated.
4. **Breadth before depth.** Six backbones, each with bespoke extractor/LoRA/scripts (≈150 scripts before the 09-24 cleanup). This is where the leak, pseudo-latent and alpha bugs lived.
5. **Train/eval feature identity not enforced** for online runs (B2, B3).
6. **Status drift in docs.** Several "no jobs running" / "quarantined" statements stayed live while runs continued. Status now lives only in this file.

## 6. Next steps (in order)

1. **Re-run invalidated arms** with the fixed code: `c2b_t2_v1_unaug` (and an augmented variant) and, if FLUX is still of interest, online FLUX with `--backbone-vae`.
2. **Seed variance:** 3 seeds each of `c3_v1_unaug` and SmolVLA on v1. Report mean ± std; re-state which leaderboard gaps are real.
3. **Backbone-swap control:** identical SmolExpert head, recipe and augmentation on SigLIP (SmolVLA vision) features. This is the cleanest test of the central question.
4. **Narrow scope** to Cosmos 3 Edge vs SmolVLA; freeze other backbones.
5. **Physical success rate as the primary metric** (fixed protocol, e.g. 20 trials, fixed cube positions); report first-5 and gripper RMSE as offline proxies.
6. **Grasp failure:** recovery / near-grasp demonstrations, gripper-close timing, possibly a wrist camera — before more backbone variants.
7. Resolve v2 integrity; pin sort-cubes revision.

Deferred research bets (from the old roadmap): compact multi-frame student backbones (<50 ms), action-conditioned latent prediction, temporal memory, 1–2 step action sampler distillation, RL once rewards/recovery data exist.

## 7. Backbone taxonomy (reference)

| Backbone            | Params         | Blocks / hidden             | Tap used                               | Tokens to policy                  | Notes                                                                         |
| :------------------ | :------------- | :-------------------------- | :------------------------------------- | :-------------------------------- | :---------------------------------------------------------------------------- |
| Cosmos-Predict2-2B  | 2.0B           | 28 / 2048                   | layer 20                               | 4,800 (T=16 pool2) or 2,400 (T=2) | VAE 8× spatial, 4× temporal; bidirectional DiT; ~1.2 s (T=16) / ~204 ms (T=2) |
| Cosmos 3 Edge       | 3.37B measured | 28 / 2048, MoT (und/gen)    | layer 20, clean observed vision tokens | 600 (2×15×20)                     | Wan 2.2 VAE; ~80–92 ms                                                        |
| LTX-2.5 (distilled) | 22B            | 48 / 4096                   | block 34                               | 2,400 or 640                      | 32× spatial VAE; ~1.2 s with CPU offload                                      |
| Cosmos-1.0 7B       | 7B             | 28 / 4096                   | layers 14+20                           | 64                                | EDM objective                                                                 |
| Cosmos-1.0 14B      | 14.25B         | 36 / 5120                   | layers 18+30                           | pooled                            | FP8 + block streaming on 24 GB                                                |
| FLUX.2 klein        | 9B             | 8 double + 48 single / 6144 | double/single junction                 | 256                               | image model with multi-reference history                                      |

Hardware: single RTX 4090 (24 GB) on `abakus`; robot SO-101 driven from a Mac over SSH RPC ([rollout](./video_vam_rollout.md)).
