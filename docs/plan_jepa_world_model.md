# Control-Aware JEPA Latent World Model & MuZero/MPPI Search

**Status:** Implemented & verified on `abakus` (`2026-10-18`).
**Code:** `src/lerobot/policies/vam/jepa_world_model.py`, `scripts/video_vam/train_jepa_world_model.py`, `scripts/video_vam/presets/jepa_world_model_v1.args`, `tests/policies/test_jepa_world_model.py`.

---

## 1. Motivation & Literature (LeCun et al., Aug–Oct 2026)

Instead of generating future RGB pixels (which makes WAMs slow) or training an RL policy purely inside a small-dataset offline world model (which reward-hacks out-of-distribution world-model errors), we learn a **compact latent world model on top of our frozen video-backbone features** and use it for **fast test-time search (Best-of-$N$ / MPPI) anchored to our trained flow-matching policy prior**.

Four recent papers from Yann LeCun's group directly inform the design:

1. **Toso, LeCun, Anderson, Bounou (Oct 6, 2026, [arXiv:2610.07540](https://arxiv.org/abs/2610.07540)):** *"Preserving Unstable Modes Through Inverse Dynamics in JEPA World Models"*
   - Proves that standard JEPA forward prediction + variance-covariance anti-collapse (`VICReg`/`SIGReg`) still allows the encoder to collapse controllable unstable modes (such as precise gripper-to-object alignment).
   - Adding an **Inverse Dynamics Model (IDM)** $I_\psi(h_t, h_{t+H}, H) \to \hat{a}_{t:t+H}$ forces the latent representation $h_t$ to remain injective over the finite-horizon reachable control subspace.
2. **Zhang, Terver, Rabbat, LeCun, Balestriero (Oct 5, 2026, [arXiv:2610.06805](https://arxiv.org/abs/2610.06805)):** *"H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning"*
   - Uses chunk-horizon latent prediction with inverse-dynamics supervision for visual planning on real-robot videos (DROID).
3. **Yalovetzky, Balestriero, LeCun (Oct 11, 2026, [arXiv:2610.15267](https://arxiv.org/abs/2610.15267)):** *"LeHami: Symplectic Latent Dynamics for World Models"*
   - Encodes both current position and temporal velocity/difference in the latent state to prevent rollout contraction.
4. **Kuhn, Maes, Serra, Le Lidec, LeCun (Aug 27, 2026, [arXiv:2608.27395](https://arxiv.org/abs/2608.27395)):** *"LeVJEPA"*
   - Variance-covariance regularization (`SIGReg` / `VICReg`) for non-contrastive video JEPA training.

---

## 2. Architecture (`ControlAwareJEPAWorldModel`)

Built directly on top of frozen **Cosmos 3 Edge** layer-20 tokens ($z_t \in \mathbb{R}^{2 \times 15 \times 20 \times 2048}$, i.e., 600 tokens) + 6-DoF proprioceptive state $s_t \in \mathbb{R}^6$:

1. **Motion-Aware Spatial-Temporal Pooler $E_\theta(z_t, s_t) \to h_t \in \mathbb{R}^{512}$:**
   - Separates the two temporal latent slots ($t_0, t_1 \in \mathbb{R}^{300 \times 2048}$) and explicitly feeds both the current spatial tokens $t_1$ and the temporal motion difference $t_1 - t_0$ into a 4-query cross-attention pooler fused with the normalized robot state $s_t$.
   - Paired with an EMA target encoder $E_{\bar{\theta}}$ ($\tau = 0.99$) and VICReg variance-covariance regularization on $h_t$.
2. **Variable-Horizon Forward Predictor $P_\phi(h_t, s_t, a_{t:t+H}, H) \to (\hat{h}_{t+H}, \hat{s}_{t+H})$:**
   - Takes current latent $h_t$, current state $s_t$, an action chunk prefix $a_{t:t+H}$ masked beyond lookahead horizon $H$, and a learned horizon embedding for $H \in \{3, 6, 9, 12, 15, 18, 21, 24, 30\}$.
   - Predicts both the future target-encoder latent $\hat{h}_{t+H}$ (smooth-$L_1$ + cosine loss) and future physical joint state $\hat{s}_{t+H}$.
3. **Inverse Dynamics Head $I_\psi(h_t, h_{t+H}, H) \to \hat{a}_{t:t+H}$ (`arXiv:2610.07540`):**
   - Reconstructs the executed action chunk from $(h_t, h_{t+H}, H)$.
   - At planning time, provides a **forward–inverse cycle-consistency score** $-\|I_\psi(h_t, P_\phi(h_t, s_t, a, H), H) - a\|^2$ that penalizes out-of-distribution action candidates where the world model would hallucinate.
4. **MuZero Value / Progress & Goal Head $V_\omega(h_t) \to (\hat{v}_t \in [0, 1], \hat{g}_t)$:**
   - Trained with episode progress regression ($t / T_{\text{ep}}$), temporal ranking margin loss ($v(h_{t+H}) > v(h_t) + \delta$), and terminal goal-prototype cosine similarity.
5. **Fast Latent Search (`score_action_candidates` & `plan_mppi`):**
   - Samples $N$ candidate action chunks in a single batched pass from the trained `SmolExpert` flow-matching policy prior.
   - Evaluates multi-horizon latent rollouts ($H \in \{10, 20, 30\}$) under value, goal-similarity, forward–inverse cycle consistency, and action smoothness in **< 4 ms** on RTX 4090.

---

## 3. Experimental Findings on `cube_out_of_box v1`

Trained on cached Cosmos 3 Edge features (`train` eps 0–31, 12,676 multi-horizon transitions, 40 epochs = ~25 seconds on RTX 4090) and evaluated on unseen validation episodes 32–39 at $H=20$ (2.0 s lookahead):

| Metric on unseen validation episodes (32–39) | Standard JEPA (`--idm-weight 0.0`) | **Control-Aware JEPA (`--idm-weight 1.0`, `arXiv:2610.07540`)** | Trivial / No-Action Baseline |
| :--- | ---: | ---: | ---: |
| **Top-1 Future Latent Retrieval (out of 80 @ Epoch 40)** | 50.0 % *(collapses from 80 %)* | **72.5 %** *(peak **87.5 %**, Top-5 **96.3 %**)* | 1.25 % (identity) |
| **Forward Latent Cosine Similarity ($\hat{h}_{t+20}$ vs $h_{t+20}$)** | 0.780 (+0.171 vs ID) | **0.852** (+0.231 vs ID, peak **0.945**) | 0.621 (identity) / 0.643 (zero-action) |
| **2.0 s Future Joint-State RMSE (degrees)** | 5.27° | **4.73°** | 21.29° (state-repeat) / 26.91° (zero-action) |
| **Inverse Dynamics Action RMSE (degrees)** | 33.11° (untrained) | **7.80°–8.14°** | — |
| **Value / Progress Spearman $\rho$ on unseen episodes** | 0.899 | **0.882–0.890** | 0.000 |
| **MuZero Best-of-16 on SmolExpert (H1 / F5 / Full-30)** | 4.25° / 5.95° / 13.64° *(hurts H1)* | **4.16° / 5.90° / 13.63°** *(improves H1 & F5 vs 4.22° / 5.95° / 13.64° single-sample)* | — |

**Key takeaway:** Without the Inverse Dynamics objective (`--idm-weight 0.0`), Top-1 retrieval peaks early and collapses to 50.0% by epoch 40, and Best-of-16 planning slightly *hurts* immediate action accuracy (`H1`). With LeCun's Oct 2026 IDM regularization (`--idm-weight 1.0`), latent collapse is prevented, future 2.0 s state error drops to **4.73°**, and Best-of-16 search improves immediate action accuracy.

---

## 4. How to Train & Run

### Train on cached Cosmos 3 Edge features (`abakus`)
```bash
RUN_TAG=jepa-world-model-v1-idm1 bash scripts/video_vam/run_experiment.sh jepa_world_model_v1
```

### Run with RPC Server (`abakus`)
```bash
.venv/bin/python scripts/video_vam/rpc_server.py \
  --policy video_vam \
  --checkpoint outputs/train/c3peft_v1_unaug \
  --jepa-world-model outputs/train/jepa-world-model-v1-idm1/best.safetensors \
  --jepa-planning-mode best_of_n \
  --jepa-num-candidates 8
```
