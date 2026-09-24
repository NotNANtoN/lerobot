# Video-VAM Correctness & Validity Audit

**Status:** Authoritative Source of Truth for Bugs, Validity, and Correctness
**Date:** 2026-09-07, second audit 2026-09-24 (§6)
**Repository:** `lerobot-video-vam` (`/home/anton/lerobot-video-vam`)
**Scope:** Deep Architectural Audit across Cosmos 2B/3/7B/14B, FLUX.2 [klein], LTX-2.5, Feature Extractors, Training Loops, and Evaluation Protocols

---

## 1. Executive Summary & Purpose

This document serves as the **definitive source of truth regarding implementation correctness, bugs, data leakage, architectural mismatches, and evaluation integrity** in the `lerobot-video-vam` project.

Prior empirical results reported in earlier iterations of the codebase contained severe methodological flaws, data leakage, pseudo-latent mocks, and schedule mismatches that invalidated several headline metrics (including the sub-11° RMSE figures previously attributed to Cosmos 7B, 14B, and FLUX.2).

This audit catalogs every identified failure mode, details the root causes, outlines the required structural invariants, and documents the verification requirements currently being implemented.

---

## 2. Cosmos 3 Edge: Architectural & Implementation Mismatches

### 2.1 Native Generation Tower vs. Custom Understanding Extraction Mismatch

- **The Defect:** In Cosmos 3 Edge (Dual-Pathway Mixture-of-Transformers), Video-LoRA fine-tuning was performed on the **generation tower (`gen_seq`)** using flow-matching velocity prediction on noisy video clips. However, downstream feature extraction bypassed the generation tower and tapped the **understanding tower (`und_seq`)** at Layer 20.
- **The Impact:** The dual LoRA parameters adapted during generative flow matching do not directly align with the un-noised understanding sequence representation. This mismatch persists despite dual-pathway LoRA injection, introducing an un-calibrated representation shift between pretraining, fine-tuning, and policy distillation.

### 2.2 Missing Per-Channel Latent Normalization & Native Posterior Mode

- **The Defect:** Cosmos 3 uses the Wan 2.2 / Cosmos causal VAE latent space. The extraction pipeline omitted native per-channel latent mean/std normalization and failed to use the deterministic **posterior mode** (`latent = posterior.mode()`). Instead, it relied on un-normalized latent samples or default Gaussian sampling, adding stochastic noise into the conditioning features.
- **Remediation:** Enforce exact per-channel latent scaling factors from the VAE configuration and strictly evaluate at posterior mode without sampling noise.

### 2.3 Video Framerate (FPS) Mismatch: 10 FPS vs. 24 FPS

- **The Defect:** The robot demonstration dataset (`cube_out_of_box_dataset`) was recorded at **10 FPS** (100 ms per step). Base Cosmos models are natively pretrained on **24 FPS** video distributions.
- **The Impact:** Feeding 10 FPS video directly into temporal positional embeddings tuned for 24 FPS causes an implicit $2.4\times$ temporal speed distortion. Without explicit framerate conditioning or temporal stride adjustment, the DiT's learned velocity priors do not match physical manipulator dynamics.

### 2.4 Patchify Packing Order Defect: Native `(p, p, C)` vs. Flawed `(C, p, p)`

- **The Defect:** In legacy extraction scripts, manual latent patchification permuted tensors into channel-first spatial patches `(C, p, p)`.
- **Native Contract:** The official Cosmos 3 transformer expects patchified vision tokens packed in spatial-first order `(p, p, C)` before linear input projection (`proj_in`). Channel transposition corrupted the spatial frequency ordering of tokens entering the transformer blocks.

### 2.5 True Model Dimensions vs. Marketing Labels

- **Measured Config:** Local inspection of `nvidia/Cosmos3-Edge` establishes exact dimensions:
  - Hidden Dimension: **2048**
  - Attention Heads: **16**
  - Transformer Layers: **28**
  - Dense Parameters: **3.37 Billion** (measured), contrasting with the public "4B" marketing label.

---

## 3. Large Model Backbones (7B, 14B, FLUX.2): Production Defects

### 3.1 Pseudo-Latent Mocks: Bypassing the VAE

- **The Defect:** In `extract_cosmos7b_features.py`, `extract_cosmos14b_features.py`, and `extract_flux2_klein_features.py`, video frames were not encoded through the backbones' causal 3D VAEs. Instead, RGB frames were downsampled via bilinear interpolation (`F.interpolate(..., size=(32, 32))`), flattened, and zero-padded to 16 channels.
- **The Impact:** The DiT transformer blocks received arbitrary bilinear pixel averages rather than structured VAE latents. Downstream policies trained on these features were not evaluating world-model latent dynamics, but rather over-fitting a random projection of downsampled pixels.
- **Rule:** Pseudo-latents and bilinear mocks are **strictly invalid**. All feature extraction must pass through genuine, calibrated VAE encoders.

### 3.2 Frame-Level Data Leakage (The False SOTA Catastrophe)

- **The Defect:** In `train_smolexpert_on_cosmos7b.py`, `train_smolexpert_on_cosmos14b.py`, and `train_smolexpert_on_flux2_klein.py`, the training script used:
  ```python
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.85, 0.15])
  ```
  across sequentially extracted frames (stride 3).
- **The Impact:** Frame $t$ (e.g. frame 100) was placed in the training set while frame $t+3$ (frame 103, 300 ms later in the same trajectory) was placed in the validation set.
- **Scientific Consequence:** The model memorized trajectories and performed simple pose interpolation. The reported validation RMSEs of **9.46° (7B)**, **9.97° (14B)**, and **10.33° (FLUX.2)** are **METHODICALLY INVALID AND RETRACTED**. True generalization under strict Protocol 1.0 (disjoint held-out episodes 32–39) is significantly higher.

### 3.3 Cosmos-1.0 Formulation Mismatch: EDM Diffusion vs. Rectified Flow (RF)

- **The Defect:** Cosmos-1.0 (7B / 14B) models are formulated using **EDM (Elucidated Diffusion Models / Karras VP)** noise schedules, whereas training scripts applied linear Rectified Flow (RF) interpolation equations:
  $$x_t = (1 - t) x_0 + t \epsilon, \quad v_t = \epsilon - x_0$$
- **The Impact:** Training an EDM-pretrained backbone with linear RF flow matching without reconciling the noise conditioning sigma schedule creates severe drift in velocity vector prediction.

---

## 4. Evaluation Rigor & Pipeline Invariants

To guarantee scientific reproducibility, all evaluation code must strictly enforce the following invariants:

### 4.1 Per-Sample Seed Batch Invariance

- Evaluation sampling must be strictly deterministic and independent of batch size or batch ordering.
- Since `sample_id` is a string identifier (e.g. `"ep32_f4820"`), integer addition is invalid. The deterministic seed must be derived via a stable hash:
  $$\text{sample\_seed} = \text{int}(\text{hashlib.sha256}(f"\{\text{evaluation\_seed}\}:\{\text{sample\_id}\}".\text{encode}("utf-8")).\text{hexdigest}()[:8], 16) \pmod{2^{31} - 1}$$
- Batching 1 sample at a time or 16 samples at a time yields bit-identical policy action outputs.

### 4.2 Gradient Accumulation & Optimizer Step Alignment

- Learning rate schedulers, logging, and checkpoint intervals must advance on **actual optimizer steps**, not raw micro-batch forward passes.
- Gradient accumulation steps (`grad_accum_steps`) must properly scale gradients and update the optimizer only every $K$ micro-batches.

### 4.3 Fail-Closed Validation Gates

- **Label & Dimension Checks:** Reject manifests that lack explicit physical unit metadata or have mismatched action dimensions (must be 6-DoF for SO-101).
- **Split Membership Verification:** Enforce disjoint episode sets via `ProtocolSplitGuard`. If any episode in `val_episodes` appears in `train_episodes`, the process must immediately abort (`raise ProtocolViolationError`).
- **Dataset Identity:** Verify the pinned dataset repository and commit revision (`243370c3c08bcbd860133c4a0d658ea7c1d2e77e`).

### 4.4 Checkpoint Resumption Integrity

- Resume operations must restore the exact optimizer state, learning rate scheduler state, step counters, and best validation metric tracked.

---

## 5. Architectural Abstractions vs. Implementation Reality

The introduction of base abstractions in `src/lerobot/policies/vam/base/` (`BaseVAMExtractor`, `BaseLoRALinear`, `ProtocolSplitGuard`) provides the architectural blueprint, but **does not magically guarantee correctness by itself**.

Concrete extractors must be audited and verified:

1. `Cosmos7BExtractor` must instantiate and call `AutoencoderKLCosmos` with correct temporal packing and scaling.
2. `Cosmos14BExtractor` must use native causal VAE encoding with block streaming and valid FP8 quantizers.
3. `Flux2KleinExtractor` must format genuine 4D RoPE coordinate positions and valid latent packing.
4. Abstract contracts must be backed by concrete test suites that execute end-to-end forward passes on synthetic tensors without mocking the VAE contract.

---

## 6. Second audit (2026-09-24): post-audit regressions in the online path

The 09-07 fixes were applied to the offline cache builders, but the online-extraction path added to `train_smolexpert.py` on 09-12/13 reintroduced or bypassed them. Affected numbers are tagged in the [leaderboard](./video_vam_leaderboard.md).

### 6.1 B1 — Pseudo-latents in online FLUX.2

`OnlineFlux2KleinWrapper` resized RGB to 32×32, space-to-depth packed and zero-padded it to 128 channels instead of calling the VAE (same defect as §3.1). `Flux2KleinExtractor.encode_latents` also returned raw RGB when no VAE path was configured. **Fix:** online FLUX requires `--backbone-vae`, encodes with `FluxVAENormalizer` and mirrors the cache builder's history/target slicing and prompt; `encode_latents` raises `Flux2KleinError` without a VAE.

### 6.2 B2 — LoRA alpha mismatch (Cosmos-2B)

The step-6000 Cosmos-2B video-LoRA sidecar records rank 16 / alpha 16. The online path injected wrappers from `--backbone-lora-alpha` (default and scripts: 32.0), i.e. a 2× adapter delta, and `load_lora_state_dict` checked only tensor names/shapes. The offline cache builder read alpha from the sidecar (16). **Fix:** `resolve_backbone_lora_hyperparameters` fills rank/alpha from safetensors metadata or sidecar and rejects conflicting CLI values; `inject_and_load_lora_file` builds wrappers from the sidecar; `load_lora_state_dict` fails if any wrapper disagrees with the sidecar.

### 6.3 B3 — Online-train vs offline-eval feature identity

Online runs evaluated on precomputed caches. Only Cosmos 3 had an identity preflight. At HEAD `f576307f` online Cosmos-2B used `high_noise_sigma=10` while caches used 80, and the trainer auto-discovered caches from other runs (`outputs/features/v2-cosmos2b-t2-undistilled/...`, `/home/anton/.cache/video-vam/*-scale100-cache/...`). **Fix:** `validate_online_eval_cache_identity` (LoRA sha256/rank/alpha, sigma, `state_t`, hidden layer, augmentation) runs for every non-Cosmos-3 online backbone; cross-run auto-discovery removed (only the sibling `eval2/` of the same cache build is auto-selected in offline mode).

### 6.4 B4 — Retracted scripts still executable

The leaky trainers, pseudo-latent extractors and renderers from §3 remained in `scripts/video_vam/`; `render_cosmos14b_rollout.py` additionally produced "rollouts" with a single heuristic latent update (`latents - w(t) * v`), which explains identical base/QLoRA PSNR. **Fix:** deleted (see [`video_vam_scripts.md`](./video_vam_scripts.md)).

### 6.5 B5 — Silent inference default change

An uncommitted edit changed the Euler-step default from 10 to 3 in `SmolExpertActionDecoder` and `VideoVAMPolicy`. **Fix:** defaults restored to the checkpoint value (10); explicit `VideoVAMConfig.action_euler_steps` / `rpc_server.py --euler-steps`.

### 6.6 Other hardening

- Non-canonical datasets in online mode require `--dataset-revision` (previously loaded unpinned `main`); the Scale-100 snapshot `5d0325cc` is pinned explicitly.
- Online mode rejects train/eval episode overlap from `--train-episodes/--val-episodes`.
- Cartesian IK joint limits are set by joint name instead of hard-coded q-indices `[7:12]`.
- `lerobot.policies` no longer requires `diffusers` at import (lazy exports in `policies/vam/__init__.py`).

## 7. Status

See [`video_vam_status.md`](./video_vam_status.md) §4 for which results must be re-run. Remaining unfixed items: Cosmos 3 und/gen tower history (§2.1, pre-fix results tagged), Cosmos-1.0 EDM vs RF objective (§3.3), v2 dataset integrity.
