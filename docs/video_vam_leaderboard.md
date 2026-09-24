# Video-VAM leaderboard (single source of numbers)

**Append-only.** Never delete or edit a historical number; add a validity tag and a note. Every new row needs: date, dataset@revision, split, seed(s), run dir, validity.
Metric: mixed-unit aggregate RMSE over a 30-step chunk (5 arm joints in degrees + gripper in [0, 100]); **H1** = offset 0, **F5** = mean of offsets 0–4. Contract: [`video_vam_action_rmse_protocol.md`](./video_vam_action_rmse_protocol.md). W&B: [video-vam-world2action](https://wandb.ai/hubnemo-hugging-face/video-vam-world2action).

**Validity tags**

- `VALID` — Protocol-conformant, leakage-free, features match training.
- `VALID*` — conformant but weights lost / not re-scored with sampling rev 1.1, or pre-dates a later extractor fix; keep as reference only.
- `PROVISIONAL` — conformant code, but dataset integrity unresolved (v2 snapshot, see [status §2](./video_vam_status.md#2-datasets)).
- `UNVERIFIED` — reported in the diary without a local artifact to check.
- `INVALID` — bug affects the number (reason given). `NON-COMPARABLE` — different setup.

**All rows are single-seed** unless stated. Differences below ~0.5 should be treated as noise until seed variance is measured.

---

## 1. Cube-out-of-box v1, Protocol 1.0

`hubnemo/cube_out_of_box_dataset@243370c3`, train eps 0–31, val eps 32–39 (88 anchors, stride 20).

| Date  | Model                                         | Features / tokens                |   Full-30 |    H1 |    F5 | Latency | Validity / note                                                                                                                                         |
| :---- | :-------------------------------------------- | :------------------------------- | --------: | ----: | ----: | :------ | :------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 08-28 | Cosmos-2B video-LoRA + SmolExpert             | T=16, L20, pool2, 4,800          | **13.06** |  4.76 |  6.19 | ~1.2 s  | `VALID*` weights on HF (`cube-out-of-box-cosmos-pool2-smolexpert`); not re-scored at rev 1.1                                                            |
| 08-28 | Cosmos-2B teacher `cond_frames` reference     | T=16 first 2 latent slots, 2,400 |     13.08 |  4.97 |  6.40 | —       | `VALID*` empirical reference, **not a ceiling**; weights lost                                                                                           |
| 08-29 | Cosmos-2B T=2 direct-distilled student        | T=2, L20, 2,400                  |     13.15 |  4.58 |  6.04 | ~204 ms | `VALID*` weights lost; "89.4 % gap closure" is within seed noise until re-run                                                                           |
| 08-24 | Cosmos-2B frozen pretrained                   | T=16, L20, pool2                 |     13.65 |     — |     — | ~1.2 s  | `VALID*`                                                                                                                                                |
| 08-28 | Cosmos-2B T=2 undistilled                     | T=2, L20, 2,400                  |     13.74 |  4.78 |  6.55 | ~204 ms | `VALID*` weights lost                                                                                                                                   |
| 08-25 | Cosmos-2B prefix-VAE                          | T=16 prefix VAE, pool2           |     13.81 |     — |     — | —       | `VALID*`                                                                                                                                                |
| 09-12 | Cosmos 3 Edge video-LoRA, online + legacy aug | 600, L20                         |     13.82 |  4.64 |  6.26 | ~80 ms  | `VALID` (Eval-2 on v2: 24.66, see §2)                                                                                                                   |
| 08-26 | LTX-2.5 22B                                   | block 34 unpooled, 2,400         |     13.84 |  4.55 |  6.24 | ~1.2 s  | `VALID*` weights on HF                                                                                                                                  |
| 08-26 | LTX-2.5 22B                                   | block 34 pool2, 640              |     14.03 |  4.50 |  6.28 | ~1.2 s  | `VALID*` weights on HF                                                                                                                                  |
| 09-04 | Cosmos 3 Edge base (zero-shot)                | 600, L20                         |     14.26 |  4.67 |  6.34 | ~92 ms  | `VALID*` pre-dates the und/gen routing fix (audit §2.1)                                                                                                 |
| 08-20 | Cosmos-2B World2Action decoder                | L20, pool2                       |     14.51 |     — |     — | —       | `VALID*`                                                                                                                                                |
| 09-2x | Cosmos-2B T=2 video-LoRA, online, no aug      | 2,400, L20, σ 80                 |     14.52 |  5.15 |  6.92 | ~204 ms | `INVALID` B2/B3: LoRA injected at alpha 32 (trained 16); eval cache built at alpha 16. Run `outputs/train/v1-cosmos2b-t2-online-unaugmented-smolexpert` |
| 08-24 | SmolVLA 450M, 1 h (29.2k steps)               | SigLIP, 64                       |     14.83 |  4.85 |  6.64 | ~120 ms | `VALID` baseline (no augmentation, 1 h budget — not matched to VAM arms)                                                                                |
| 09-06 | Cosmos 7B base, re-benchmark stage 1          | L14+20, 64                       |     15.98 |  5.58 |  7.77 | —       | `VALID*` real VAE, but unstandardized intermediate cache; retired pending clean run                                                                     |
| 08-19 | state-repeat                                  | —                                |     18.86 |  0.89 |  4.45 | —       | trivial baseline                                                                                                                                        |
| 08-19 | mean action                                   | —                                |     30.15 | 12.40 | 18.20 | —       | trivial baseline                                                                                                                                        |

## 2. Scale-100 (cube v2)

`Orellius/cube_out_of_box_v2@5d0325cc`, train eps 0–31 + 40–89. **Eval-1** = v1 val eps 32–39 (88 anchors), **Eval-2** = eps 90–99 (51 anchors). All `PROVISIONAL` because of the v2 metadata/row mismatch.

| Date  | Model                                                 | Train data       | Eval-1 (H1)      | Eval-2 (H1)      | Validity / note                                                      |
| :---- | :---------------------------------------------------- | :--------------- | :--------------- | :--------------- | :------------------------------------------------------------------- |
| 09-11 | Cosmos 3 Edge video-LoRA + SmolExpert (offline cache) | v2 0–31, 40–89   | **13.49** (3.62) | **17.20** (4.92) | `PROVISIONAL`                                                        |
| 09-12 | Cosmos 3 Edge video-LoRA, online + legacy aug         | v2 0–31, 40–89   | 15.48 (4.60)     | 17.23 (5.08)     | `PROVISIONAL`                                                        |
| 09-12 | Cosmos 3 Edge video-LoRA, online + legacy aug         | **v1** 0–31 only | 13.82 (4.64)     | 24.66 (10.00)    | `VALID` for Eval-1; Eval-2 is cross-dataset                          |
| 09-11 | Cosmos 14B base, L18+30, FP8 streaming                | v2               | 14.02 (4.57)     | 19.41 (6.08)     | `PROVISIONAL`                                                        |
| 09-11 | FLUX.2 klein base, junction tap                       | v2               | 15.75 (5.18)     | 20.45 (6.56)     | `PROVISIONAL`; verify cache `vae_sha256` (B1 follow-up)              |
| 09-11 | FLUX.2 klein video-LoRA                               | v2               | 15.89 (5.07)     | 21.02 (6.77)     | `PROVISIONAL`; verify cache `vae_sha256`                             |
| 09-13 | Cosmos-2B T=2 video-LoRA, online + aug                | v2               | 15.57 (4.84)     | 18.71 (6.32)     | `INVALID` B2/B3 (alpha 32 vs 16; eval cache from different features) |
| 09-13 | FLUX.2 klein base, online + aug                       | v2               | 18.42 (10.93)    | 21.59 (10.74)    | `INVALID` B1 (bilinear pseudo-latents in online path)                |
| 09-11 | Cosmos 14B video-LoRA (FP8 QLoRA)                     | v2               | —                | —                | never ran (was "QUEUED")                                             |

## 3. Sort-cubes (different task — not comparable to §1/§2)

`Orellius/so101_sort_cubes_no_top@<unpinned>`, train eps 0–67, held-out 68–76. All runs before 09-24 did not record a dataset revision.

| Date  | Model                                                                | Recipe                    |                 Full-30 |    H1 |    F5 | Validity / note                                                                                                              |
| :---- | :------------------------------------------------------------------- | :------------------------ | ----------------------: | ----: | ----: | :--------------------------------------------------------------------------------------------------------------------------- |
| 09-2x | Cosmos 3 Edge video-LoRA, online, joint actions                      | BS 8 (4×2), 3 Euler steps |                   15.45 |  4.46 |  6.11 | `VALID` (revision unpinned). Run `outputs/train/sort-cubes-cosmos3-edge-lora-online-unaugmented`                             |
| 09-2x | Cosmos 3 Edge, online, effective BS 64                               | 8×8, LR 1e-3              |                   14.81 |  4.31 |  5.79 | `UNVERIFIED` no local artifact (`sort-cubes-cosmos3-edge-optimized-bs64` on abakus?)                                         |
| 09-2x | Cosmos 3 Edge, online, 5D Cartesian (xyz+pitch+roll), single-step IK | BS 8                      |                   42.74 | 32.58 | 41.67 | `VALID` (revision unpinned); first iteration, yaw drift. Local run `outputs/train/sort-cubes-cosmos3-edge-lora-cartesian-ik` |
| 09-2x | Cosmos 3 Edge, online, 7D Cartesian (xyz+rotvec), iterative IK       | BS 8                      |         24.58 (55.8 mm) |     — |     — | `UNVERIFIED` no local artifact (overwrote the same run dir on abakus?). Still far worse than joint space                     |
| 09-2x | SmolVLA, fresh                                                       | lerobot-train             | 17.27 @50k, 17.73 @155k |     — |     — | `UNVERIFIED` locally                                                                                                         |
| —     | `Orellius/..._smolvla_base_100k` (HF)                                | random 8 % frame holdout  |                   6.515 |     — |     — | `NON-COMPARABLE` trained on the held-out episodes                                                                            |

## 4. Video prediction (PSNR/SSIM) — diagnostic only

PSNR/SSIM reward blurry, conservative predictions and were shown to disagree with task plausibility (diary 08-18). Not a backbone-quality metric.

| Model                                              | Frames | PSNR / SSIM           | Validity                                                                                        |
| :------------------------------------------------- | :----- | :-------------------- | :---------------------------------------------------------------------------------------------- |
| Cosmos 14B base / QLoRA (ep 32, 161 frames)        | 161    | 18.28 / 0.7803 (both) | `INVALID` B4: renderer used a one-step heuristic latent update; base and LoRA outputs identical |
| Cosmos 7B (17 frames)                              | 17     | 28.21 / 0.877         | `INVALID` pseudo-latent renderer                                                                |
| FLUX.2 klein (17 frames)                           | 17     | 12.56 / 0.491         | `INVALID` pseudo-latent renderer                                                                |
| Cosmos-2B, 5-frame conditioning vs static baseline | 16     | 14.55 vs 24.11 (ep 0) | `VALID` diagnostic (diary 08-18)                                                                |

## 5. Retraction ledger (pre-audit, 2026-09-05/06)

| Date  | Run                                                                    | Reported | Reason                                                                |
| :---- | :--------------------------------------------------------------------- | :------- | :-------------------------------------------------------------------- |
| 09-05 | Cosmos 7B base (L14+20)                                                | 9.46     | `INVALID` frame-level `random_split` leak + bilinear pseudo-latents   |
| 09-05 | Cosmos 7B video-LoRA                                                   | 9.75     | `INVALID` same                                                        |
| 09-05 | Cosmos 14B base (L18+30)                                               | 9.97     | `INVALID` frame-level leak; ignored `action_is_pad`                   |
| 09-05 | FLUX.2 video-LoRA                                                      | 10.33    | `INVALID` leak + pseudo-latents                                       |
| 09-05 | FLUX.2 base junction                                                   | 11.45    | `INVALID` leak + pseudo-latents                                       |
| 09-06 | Cosmos 14B base "converged"                                            | 10.99    | `INVALID` leak                                                        |
| 09-06 | Cosmos 14B QLoRA                                                       | 11.08    | `INVALID` leak                                                        |
| 08-20 | Early unadapted Cosmos-2B                                              | 26.12    | `NON-COMPARABLE` uncalibrated early config; features not reproducible |
| 09-08 | `grand_evaluation_summary.json` from `run_full_autonomous_pipeline.sh` | —        | `INVALID` hard-coded placeholder numbers                              |
