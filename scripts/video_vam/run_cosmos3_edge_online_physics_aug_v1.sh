#!/usr/bin/env bash
# ==============================================================================
# COSMOS 3 EDGE: ONLINE VIDEO EXTRACTION WITH PHYSICS-INFORMED AUGMENTATIONS
# (Planckian Illuminant Jitter + Smooth Cast Shadows + Gamma Correction, Seed 42)
# Strictly zero spatial crop (preserves geometry) and zero blur (preserves edges).
# ==============================================================================
set -Eeuo pipefail

REPO_ROOT="/home/anton/lerobot-video-vam"
cd "$REPO_ROOT"

source scripts/video_vam/cosmos_cuda_env.sh
export PYTHONPATH=".:src"
export PATH="/home/anton/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
PYTHON="/home/anton/lerobot-video-vam/.venv/bin/python"

mkdir -p outputs/logs outputs/train

LOG_FILE="outputs/logs/c3_online_physics_aug_v1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

OUTPUT_DIR="outputs/train/v1-cosmos3-edge-lora-online-physics-aug-smolexpert"
LORA_WEIGHTS="outputs/train/cosmos3-edge-video-lora/best_lora.safetensors"
VAL_MANIFEST="outputs/features/cosmos3-edge-lora/val/manifest.json"
EVAL2_MANIFEST="outputs/features/cosmos3-online-aug-eval-fix-20260914/v1-adapter-eval2/manifest.json"

echo "================================================================================"
echo "LAUNCHING COSMOS 3 EDGE ONLINE PHYSICS-AUG V1 TRAINING (SEED 42) AT $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "LoRA weights:     $LORA_WEIGHTS"
echo "Augmentation:     Planckian Jitter + Smooth Cast Shadows + Gamma (Strategy: physics)"
echo "Seed:             42"
echo "Log file:         $LOG_FILE"
echo "================================================================================"

"$PYTHON" scripts/video_vam/train_smolexpert.py \
    --online-backbone cosmos3_edge \
    --backbone-checkpoint /home/anton/.cache/video-vam/cosmos3-edge \
    --backbone-lora-weights "$LORA_WEIGHTS" \
    --backbone-lora-rank 16 \
    --backbone-lora-alpha 32.0 \
    --backbone-layer 20 \
    --dataset-repo-id hubnemo/cube_out_of_box_dataset \
    --train-stride 1 \
    --augment \
    --aug-strategy physics \
    --no-spatial-crop \
    --val-manifest "$VAL_MANIFEST" \
    --eval2-manifest "$EVAL2_MANIFEST" \
    --output-dir "$OUTPUT_DIR" \
    --protocol protocol1 \
    --batch-size 4 \
    --grad-accum-steps 2 \
    --lr 1e-4 \
    --lr-scheduler cosine \
    --warmup-steps 1000 \
    --min-steps 5000 \
    --max-steps 25000 \
    --val-every 500 \
    --patience 20 \
    --seed 42 \
    --overwrite

echo "================================================================================"
echo "COSMOS 3 EDGE ONLINE PHYSICS-AUG V1 TRAINING (SEED 42) COMPLETED AT $(date)"
echo "================================================================================"
