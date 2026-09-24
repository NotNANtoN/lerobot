# Video-VAM scripts and how to run experiments

On 2026-09-24 `scripts/video_vam/` was reduced from ~175 files to ~65. The ~80 one-off `run_*.sh` / `*_queue.sh` wrappers were replaced by **one launcher + preset files**. Deleted scripts remain in git history at commit `f576307f` (`git show f576307f:scripts/video_vam/<name>`); untracked ones were copied to `outputs/archive/scripts-untracked-20260924/` (gitignored).

## Launching experiments

```bash
bash scripts/video_vam/run_experiment.sh --list                  # show presets
bash scripts/video_vam/run_experiment.sh c3_v1_unaug             # run one
bash scripts/video_vam/run_experiment.sh c3_v1_unaug --seed 1    # extra args override the preset
RUN_TAG=c3_v1_unaug_s1 bash scripts/video_vam/run_experiment.sh c3_v1_unaug --seed 1
DRY_RUN=1 bash scripts/video_vam/run_experiment.sh c3_sort_cubes # print the resolved command
bash scripts/video_vam/run_queue.sh c3_v1_unaug c3_v1_physics_aug   # sequential queue
tmux new -d -s vam_queue 'bash scripts/video_vam/run_queue.sh a b c' # detached on abakus
```

The launcher sources `cosmos_cuda_env.sh` (if the CUDA wheels exist), acquires `gpu_lock.sh` (when `flock` + `nvidia-smi` exist; `GPU_LOCK=0` to skip), tees output to `outputs/logs/<RUN_TAG>_<timestamp>.log` and prints commit + dirty flag. Outputs go to `outputs/train/${RUN_TAG}` (default `RUN_TAG` = preset name).

### Preset format (`scripts/video_vam/presets/<name>.args`)

One or more CLI tokens per line, `#` comments. Directives:

- `#! trainer: smolexpert` (default, `train_smolexpert.py`), `lerobot-train`, or any `scripts/video_vam/<script>.py`.
- `#! include: <preset>` prepends another preset. Files starting with `_` are shared fragments and hidden from `--list`.
- `${RUN_TAG}` and other env vars are expanded; command substitution is rejected.

| Preset                                                                              | Purpose                                                 |
| :---------------------------------------------------------------------------------- | :------------------------------------------------------ |
| `c3_v1_unaug`, `c3_v1_aug_legacy`, `c3_v1_photometric`, `c3_v1_physics_aug`         | Cosmos 3 Edge online, cube v1, augmentation arms        |
| `c3_scale100_aug`                                                                   | Cosmos 3 Edge online, Scale-100, legacy aug             |
| `c2b_t2_v1_unaug`                                                                   | Cosmos-2B T=2 online (re-run of the B2-invalidated arm) |
| `c3_sort_cubes`, `c3_sort_cubes_bs64`, `c3_sort_cubes_cartesian`                    | sort-cubes arms (**append `--dataset-revision <sha>`**) |
| `smolvla_sort_cubes`, `smolvla_sort_cubes_100k`, `smolvla_sort_cubes_200k_compiled` | native SmolVLA baselines                                |
| `lora_cosmos3_edge_v1`                                                              | train the Cosmos 3 Edge dual-pathway video-LoRA         |
| `cache_cosmos3_edge_v1_{train,val}`                                                 | Cosmos 3 Edge 600-token caches                          |
| `cache_cosmos14b_scale100`, `cache_flux2_klein_scale100`                            | unified cache builder (`build_vam_feature_cache.py`)    |

Note: `--aug-strategy` defaults to `physics` since 09-17; presets that reproduce earlier augmented runs pin `legacy` (+ `--spatial-crop` where the old run cropped).

## Remaining scripts by role

| Role                              | Scripts                                                                                                                                                                                                                                                                                                     |
| :-------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Launch                            | `run_experiment.sh`, `run_queue.sh`, `presets/`                                                                                                                                                                                                                                                             |
| Environment                       | `cosmos_cuda_env.sh`, `gpu_lock.sh`, `test_gpu_lock.sh`                                                                                                                                                                                                                                                     |
| Action-expert training / eval     | `train_smolexpert.py` (offline caches **or** `--online-backbone`; `--eval-only`, `--eval-run-dir` re-evaluation), `evaluate_action_rmse.py`, `evaluate_smolvla_dataset_direct.py`, `post_train_rmse.sh`, `ablate_euler_steps.py`                                                                            |
| Feature caches                    | `build_vam_feature_cache.py` (Cosmos 7B/14B, FLUX.2; Protocol guards, real VAE), `build_cosmos_feature_cache.py` (Cosmos-2B), `extract_cosmos3_edge_pure_vision.py` (Cosmos 3 Edge), `build_ltx_feature_cache.py`, `build_ltx_layer_mix_cache.py`, `subset_cosmos_cache_manifest.py`, `create_vam_split.py` |
| Backbone video-LoRA               | `train_cosmos_video_lora.py` (2B), `train_cosmos3_edge_video_lora.py`, `train_cosmos7b_video_lora.py`, `train_cosmos14b_video_lora.py`, `train_flux2_klein_lora.py`, `train_consolidated_video_lora.py`, `merge_cosmos_lora.py`                                                                             |
| Distillation / research trainers  | `train_cosmos_t2_distillation.py`, `train_cosmos_t2_world_expert.py`, `train_cosmos_lora_cotrain.py`                                                                                                                                                                                                        |
| Legacy World2Action path (tested) | `train_cosmos_world2action.py`, `train_cosmos_world2action_overfit.py`, `evaluate_cosmos_world2action_cache.py`, `train_ltx_world2action.py`, `train_ltx_world2action_tiny.py`, `train_ltx_layer_mix.py`, `train_smolexpert_on_cosmos.py`, and their `run_*` wrappers                                       |
| Prompt embeddings                 | `generate_cosmos_prompt_embedding.py`, `precompute_ltx_prompt.py`, `precompute_fastwam_text.py`                                                                                                                                                                                                             |
| Rollout / RPC                     | `rpc_server.py`, `run_rpc_server.sh`, `rpc_client.py`, `run_mac_vam_rpc.py`, `prepare_video_vam_rollout.py`, `dry_run_rollout.py`, `validate_video_vam_rollout.py`, `temporal_consistency.py`                                                                                                               |
| Video prediction / renders        | `preview_cosmos_video_prediction.py`, `preview_cosmos_t2_we_video_prediction.py`, `render_cosmos3_edge_rollout.py`, `render_cosmos3_edge_full_episode.py`                                                                                                                                                   |
| Benchmarks / smoke tests          | `benchmark_cosmos_extraction.py`, `benchmark_ltx_extraction.py`, `benchmark_compiled_train_step.py`, `smoke_test_cosmos_extractor.py`, `smoke_test_ltx_extractor.py`, `run_smoke_test_cosmos_extractor.sh`                                                                                                  |
| Downloads                         | `download_cosmos2b_backbone.py`, `download_cosmos14b.py` (and `src/lerobot/policies/vam/download_cosmos_checkpoints.py`)                                                                                                                                                                                    |

## What was removed and why

| Category            | Removed                                                                                                                                                                                                                                                                                                                                                                                                                                       | Replacement                                                                                                         |
| :------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------ |
| Retracted / invalid | `train_smolexpert_on_{cosmos7b,cosmos14b,flux2_klein}.py` (frame-level `random_split`), `extract_{cosmos7b,cosmos14b,flux2_klein}_features.py` and `render_{cosmos7b,flux2_klein}_rollout.py` (bilinear pseudo-latents), `render_cosmos14b_rollout.py` (one-step heuristic, not a sampler), `run_full_autonomous_pipeline.sh` (hard-coded numbers), `update_*_leaderboard.py` (scripts writing results into docs), `download_and_eval_c7b.py` | `build_vam_feature_cache.py` + `train_smolexpert.py`; `rollout_harness.py` for real rollouts                        |
| Ad-hoc evaluators   | `eval_v2_*.py`, `eval_both_v1_v2.py`, `run_eval_v2_simple.py`, `run_protocol1_1_dual_eval.py`, `evaluate_ltx_action_rmse.py`, `report_ltx_*.py`, `benchmark_orellius_models.py`, `benchmark_uploaded_smolvla_suite.py`, `generate_comparison_plots.py`                                                                                                                                                                                        | `train_smolexpert.py --eval-only / --eval-run-dir`, `evaluate_action_rmse.py`, `evaluate_smolvla_dataset_direct.py` |
| August probes       | sigma-427 probes, `sigma_discriminability_probe.py`, `state_context_incremental_probe.py`, `check_context_plumbing.py`, `verify_random_anchor.py`, `smoke_random_anchor.sh`, `chain_smoke.sh`, `clip_ab.sh`, `bench_ltx_*`, `benchmark_ltx_layer_mix.py`, `benchmark_distilled_inference.py`, `inspect_cube_dataset.py`, `extract_cosmos3_edge_features.py` (superseded by pure-vision extractor)                                             | results are recorded in the diary                                                                                   |
| Wrappers / queues   | ~80 `run_*.sh`, `*_queue.sh`, `*_pipeline.sh`, `schedule_*.sh`, download retry loops; root `run_smolvla_heldout_0_31*.sh`, `prefetch_smolvla_retry.sh`, `run_night_online_vam.sh`                                                                                                                                                                                                                                                             | `run_experiment.sh` + presets, `run_queue.sh`                                                                       |
| Backups             | `*.pre-rollout-fix-20260908`, `*.pre-tmux-fix-20260908`                                                                                                                                                                                                                                                                                                                                                                                       | git history / `outputs/archive/`                                                                                    |
