# Research direction, publication targets, and side goals

**Status:** brainstorming, 2026-09-24. Nothing here is scheduled.

## 1. Anton's goals (in priority order)

1. **Get started publishing:** the PEFT × LeRobot blog post with Nemo ([plan](./plan_peft_blog.md)).
2. **Fix own blog post** on the T=2 distillation idea ([plan](./plan_t2_distillation.md)); share on Twitter.
3. **A good baseline on a good feature set** — a policy that works on the robot, and a defensible choice of visual features.
4. **Move from "playing around" to research:** find a publishable question; submit to a smaller conference or workshop (to go there and exchange ideas).
5. **Later:** post-training; training with features in little time; training on a real robot safely without simulation.

## 2. Side goal (archived rationale): Black Forest Labs application

Anton applied to Black Forest Labs as a forward-deployed robotics engineer and was rejected immediately. The FLUX.2 klein and LTX-2.5 work was motivated by that application: show that (a) arbitrary video/image generative models (LTX, BFL's own FLUX.2) can be adapted to robot control, and (b) bigger backbones help (Cosmos 2B → 7B → 14B).

Where that work stands:

- LTX-2.5 22B: valid Protocol 1.0 results (13.84 unpooled / 14.03 pool2), ~1.2 s latency with CPU offload. Usable as a "non-NVIDIA backbone works" data point.
- FLUX.2 klein: Scale-100 cache numbers unverified (no VAE provenance), online numbers invalid (pseudo-latents). Would need a clean rebuild.
- Scaling (2B → 7B → 14B): no valid evidence yet; the sub-11 numbers were retracted, and 14B Scale-100 (14.02 / 19.41, provisional) does not beat Cosmos 3 Edge (13.49 / 17.20).

Decision: **paused.** The goal is now research output rather than a portfolio for one company. The code paths stay (LTX tested; FLUX fixed to fail closed) so a "backbone-agnostic adaptation" section or a scaling study can be revived later if a research question needs it.

## 3. Candidate research questions (brainstorm)

Criteria: answerable on one RTX 4090 + one SO-101 in weeks; interesting to a robot-learning workshop; builds on what exists.

| #   | Question                                                                                                                                                                                  | Builds on                           | Novelty                           | Effort              |
| :-- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------- | :-------------------------------- | :------------------ |
| R1  | **Test-time context distillation for video-model policies:** can a LoRA make a short-context video forward recover long-context (imagined future) features?                               | T=2 distillation                    | medium–high                       | medium              |
| R2  | **Which video-model features predict control, and does offline action error predict real success?** Offline RMSE vs robot success across ~6 policies on one task, with seeds.             | existing checkpoints, robot harness | medium (under-studied, practical) | medium (robot time) |
| R3  | **Parameter-efficient adaptation of world models for low-data robot tasks:** PEFT methods (LoRA/DoRA/…) on video backbone vs policy head vs both, data-efficiency curves.                 | PEFT post                           | medium                            | medium              |
| R4  | **Minutes-scale policy training on frozen video features:** how fast can a policy be trained if features are cached (steps vs wall-clock vs success)?                                     | training-efficiency plots, caching  | medium                            | low–medium          |
| R5  | **Safe real-robot post-training without simulation:** value/critic on frozen features, conservative updates, human-gated rollouts (FLUX-mimic reconstruction in `video_vam_rl_notes.md`). | RL notes, rollout harness           | high                              | high                |

| R6  | **One backbone pass, many heads:** shared frozen video features feed a small action expert *and* cheap auxiliary heads (success / stage-done detector, possibly progress or value). The success head sequences sub-tasks (task A done → trigger task B) and doubles as an automatic evaluator and a reward signal for post-training. | Cosmos 3 Edge features, SmolExpert, rollout harness | medium–high (practical, bridges to R5) | low–medium |

Suggested path: **R1 as the first workshop paper** (it is the contribution you already believe in), with **R2** as a strong supporting section (it also answers "is RMSE a useful indicator?", see §4). **R4** is cheap and fits blog posts. **R5** is the longer-term direction (goal 5).

### R6 details: shared-feature success detector and task chaining (Anton's idea, 2026-09-24)

- **Architecture:** one backbone forward per control step → features `[600, 2048]` (Cosmos 3 Edge) → (a) SmolExpert action head, (b) a small classifier head (attention-pool + MLP, <1M params) predicting `p(stage done)`; optionally (c) task-conditioned action heads or one head with a task token. Extra inference cost: ~0 (the backbone dominates).
- **Labels for free:** from existing demos — last *k* frames of each episode = "done", earlier = "not done"; better: a short manual annotation of the grasp/drop moment. Negatives from failed robot rollouts later (hard negatives).
- **Why it matters beyond chaining:** it is (1) an **automatic success metric** for robot evaluation (fewer manual counts, helps R2), (2) a **reward/termination signal** for post-training without simulation (R5), and (3) a clean PEFT story (one frozen backbone + several tiny heads/adapters).
- **Task candidates** (must be much easier than precise grasping, visually distinct stages, reset-able):
  1. **Push-then-push:** push cube A into a marked zone, then push cube B (no grasp; pushing is forgiving).
  2. **Open/close lid or drawer, then touch/press a target** (large, forgiving motions).
  3. **Pick from a big bowl/bin into a big box** with a large compliant object (sponge, soft ball) instead of a small cube — grasp is easier.
  4. **Sort-cubes as a two-stage chain** (cube 1 then cube 2), since a working policy exists there; the success head only has to detect "cube in box".
- **First experiment (cheap, offline):** train the done-head on sort-cubes features; measure AUROC and timing error (frames) on held-out episodes 68–76. Then run it live alongside the working sort-cubes policy as a passive monitor before letting it trigger anything.

## 4. On "offline RMSE vs robot success"

Anton's position: if a policy cannot predict the demonstrated actions it is probably not good, so offline RMSE is a (correlated) indicator even if the robot is the real test. Agreed, with a nuance worth measuring rather than assuming: the correlation seems weak across _tasks_ (sort-cubes held-out SmolVLA ~17 works on the robot, cube v1 ~14.8 does not) and unknown _within_ a task. Plan: for sort-cubes, collect both offline RMSE (held-out 68–76, H1/F5/full-30) and robot success for every policy tested (old Orellius policy, retrained SmolVLA, Cosmos 3 Edge BS 8 / BS 64). That is R2's first data and immediately useful.

## 5. Venues to consider (verify dates before committing)

Robot-learning workshops attached to CoRL, RSS, ICRA and NeurIPS (e.g. world models, foundation models for robotics, robot learning workshops) typically accept 4–8 page workshop papers with deadlines 1–3 months before the event. The Hugging Face blog and LeRobot community channels are the fastest outlets for the blog-level results.

## 6. Current robot-side plan (Anton, in progress)

On sort-cubes with the cubes placed in the cardboard box:

1. Run the old Orellius policy on the robot (success rate).
2. Run the retrained SmolVLA (BS 32 / long schedule; runs `sort-cubes-smolvla-100k-exact`, `…-200k-compiled` on abakus).
3. Compare with held-out offline RMSE.
4. Run the Cosmos 3 Edge policies (`sort-cubes-cosmos3-edge-optimized-bs64`, 14.81; `…-online-unaugmented`, 15.45). If lower RMSE also means better robot performance, that is the first R2 data point and the feature-set baseline for goal 3.

Record each session with the protocol in [`video_vam_rollout.md`](./video_vam_rollout.md#physical-test-protocol-proposed-standard).

## 7. On SigLIP as a comparison

Why SigLIP specifically: it is the vision encoder inside SmolVLA, so "Cosmos 3 Edge features vs SigLIP features under the _same_ action head and recipe" isolates the one variable the project claims matters (video-model features vs image-text features). Without it, "Cosmos 3 beats SmolVLA" confounds features with head size, training budget and augmentation. It is only needed for the research/paper claim (R1/R2/R3), not for getting a working robot.
