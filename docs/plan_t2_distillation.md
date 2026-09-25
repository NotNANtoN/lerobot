# Plan: T=2 manifold distillation (Anton's contribution track)

**Status:** planning, 2026-09-24. No experiments launched.
**Why this track:** it is the most original idea in the project and the basis for Anton's own blog post (current drafts not satisfactory) and possibly a workshop paper. Accepted up front: it is messier than the PEFT story, the Cosmos 3 Edge result partly undercuts the "we need it for speed" motivation, and it needs more careful testing to count as a contribution.

## 1. The idea in one paragraph

A video diffusion model (Cosmos-Predict2-2B) conditioned on past frames builds its best action-relevant representation when it runs the full video window (T=16 latent frames: 2 observed + 14 noisy future slots, ~1.2 s). We only need the features of the **observed** frames at one intermediate layer (layer 20). Running only those observed frames (T=2, ~204 ms) is ~6× faster but the features are worse, because the observed tokens can no longer read back from the future slots. **Train a LoRA on the T=2 forward so that its layer-20 observed-token features match the T=16 teacher's layer-20 observed-token features.** The LoRA is the speed-up: same weights, a 2-frame forward that "imagines" the future context it no longer computes.

## 2. What exists

| Item                                                                       | Status                                                                                                                                               | Where                                                                                          |
| :------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- |
| Teacher: T=16 video-LoRA Cosmos-2B, pool2 SmolExpert                       | 13.06 (single seed, not re-scored at rev 1.1)                                                                                                        | HF `cube-out-of-box-cosmos-pool2-smolexpert`                                                   |
| Teacher `cond_frames` reference (policy on teacher's first 2 latent slots) | 13.08 (weights lost)                                                                                                                                 | diary 09-02                                                                                    |
| Student T=2 undistilled                                                    | 13.74 (original), **13.646** re-trained 09-09                                                                                                        | abakus `outputs/train/cosmos2b-t2-undistilled-smolexpert`                                      |
| Student T=2 direct-distilled LoRA + SmolExpert                             | 13.15 (original), **13.484** re-trained 09-09                                                                                                        | abakus `outputs/train/cosmos2b-t2-direct-distilled-lora`, `…/cosmos2b-t2-distilled-smolexpert` |
| Distillation trainer                                                       | working                                                                                                                                              | `scripts/video_vam/train_cosmos_t2_distillation.py`                                            |
| Failed variants (useful for the story)                                     | trunk backprop collapse (14.5–14.7), T=4 noise slots (17.9), attention scaling (13.74), linear readout head (14.16 / 14.85), 4-frame lag bug (14.08) | diary 08-29 → 09-02                                                                            |
| Evaluation                                                                 | Protocol 1.0 cube v1, 88 anchors, single seed                                                                                                        | leaderboard §1                                                                                 |

**Important observation:** the 09-09 re-trains shrink the gap. Undistilled 13.646 → distilled 13.484 (Δ 0.16), versus the original Δ 0.59. This is exactly why seeds are needed before claiming anything.

## 3. Hypotheses to test

- **H1 (core):** distillation recovers a meaningful part of the T=16 → T=2 gap in _downstream policy quality_, beyond seed noise.
- **H2 (mechanism):** the gain comes from recovering information the T=2 forward loses (the "future read-back"), not merely from extra LoRA fine-tuning on robot videos. Control: same LoRA capacity/steps trained with a non-teacher objective (e.g. video flow loss at T=2, or distillation to the T=2 model itself = identity target).
- **H3 (generality):** the technique transfers to another backbone (Cosmos 3 Edge: short vs longer generation window; or LTX) and/or another task (sort-cubes).
- **H4 (cost):** student reaches teacher quality at ≥5× lower latency; report the full Pareto curve (latency vs RMSE vs robot success).

## 4. Experiments (proposed, not started)

| ID  | Experiment                                                                                  | Seeds           | Cost estimate      | Answers                        |
| :-- | :------------------------------------------------------------------------------------------ | :-------------- | :----------------- | :----------------------------- |
| D1  | Re-score teacher, undistilled, distilled heads with sampling rev 1.1 on the same 88 anchors | —               | < 1 h              | clean baseline numbers         |
| D2  | Train SmolExpert on undistilled vs distilled features, 3 seeds each (features fixed)        | 3+3             | ~6 × 1–2 h         | H1 at head level               |
| D3  | Re-run distillation LoRA with 2 more seeds, re-extract, train heads                         | 2               | ~2 × (3 h + 1.5 h) | H1 incl. distillation variance |
| D4  | Control: LoRA with same rank/steps, non-teacher objective                                   | 2               | ~2 × 4 h           | H2                             |
| D5  | Teacher reference head (`cond_frames`) retrain, 3 seeds                                     | 3               | ~3 × 1.5 h         | upper reference                |
| D6  | Latency: T=16 vs T=2 vs T=2+merged LoRA, eager/compiled                                     | —               | < 1 h              | H4                             |
| D7  | Transfer: same recipe on sort-cubes (and/or Cosmos 3 Edge window)                           | 1–3             | 1–2 days           | H3                             |
| D8  | Robot: undistilled vs distilled on the task that works on the robot                         | 10+ trials each | robot time         | does it matter physically      |

Minimum viable set for a good blog post: **D1, D2, D3, D6** (+ D4 if time). For a workshop paper: add **D4, D5, D7, D8**.

## 5. How to present it (blog, then possibly paper)

- Framing: _"Your video model already knows the future — distil that into a 2-frame forward."_ Show the negative results (trunk collapse, readout head, lag bug) as part of the story; they make the final recipe credible.
- Key figure: latency vs policy error, points for T=16 teacher, T=2, T=2+distilled, Cosmos 3 Edge, SmolVLA; error bars from seeds.
- Second figure: per-horizon error (H1 / first-5 / full-30) — distillation's reported gain was largest early in the chunk, which is what RTC actually executes.
- Honest limitations: single dataset/task unless D7 is done; Cosmos 3 Edge is fast natively; offline RMSE ≠ robot success.

## 6. Relationship to the PEFT post

The distillation LoRA is a natural "advanced" section or follow-up of the PEFT post (LoRA as a speed-up, not just adaptation). Port `train_cosmos_t2_distillation.py` to PEFT only if it goes into that post.

## 7. Risks

- Gap may vanish under seeds (the 09-09 re-train already halved it).
- Teacher itself is single-seed; "89 % gap closure" must not be quoted without CIs.
- The v1 benchmark has a session/operator shift between train and val; results may be dominated by that. D7 mitigates.
