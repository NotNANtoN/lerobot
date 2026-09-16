#!/usr/bin/env bash
# ==============================================================================
# COSMOS 3 EDGE VIDEO-LORA: ONLINE VIDEO EXTRACTION WITH COHERENT AUGMENTATIONS (CLEAN V1 DATASET)
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

LOG_FILE="outputs/logs/c3_online_aug_v1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

OUTPUT_DIR="outputs/train/v1-cosmos3-edge-lora-online-aug-smolexpert"
LORA_WEIGHTS="outputs/train/cosmos3-edge-video-lora/best_lora.safetensors"

echo "================================================================================"
echo "LAUNCHING COSMOS 3 EDGE ONLINE AUGMENTED V1 TRAINING AT $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "LoRA weights:     $LORA_WEIGHTS"
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
    --val-manifest outputs/features/cosmos3-edge-lora/val/manifest.json \
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
echo "COSMOS 3 EDGE ONLINE AUGMENTED V1 TRAINING COMPLETED AT $(date)"
echo "================================================================================"
