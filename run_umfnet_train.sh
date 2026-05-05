#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
ENV_NAME="${ENV_NAME:-mambacpnet}"
RESULT_DIR_NAME="${RESULT_DIR_NAME:-Result_UMFNet}"
SAVE_DIR="$PROJECT_ROOT/Results/$RESULT_DIR_NAME"
GPU_ID="${GPU_ID:-0}"
IMAGE_ROOT_MODE="${IMAGE_ROOT_MODE:-all}"
LOAD_PRE="${UMFNET_PRETRAIN:-}"
TEST_RGB_ROOT="${UMFNET_TEST_RGB_ROOT:-}"
TEST_DEPTH_ROOT="${UMFNET_TEST_DEPTH_ROOT:-}"
TEST_GT_ROOT="${UMFNET_TEST_GT_ROOT:-}"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  source "$HOME/.bashrc"
fi

conda activate "$ENV_NAME"

cd "$PROJECT_ROOT"
mkdir -p "$SAVE_DIR"

export PYTHONUNBUFFERED=1

python -u "$PROJECT_ROOT/UMFNet_train.py" \
  --epoch 100 \
  --lr 5e-5 \
  --batchsize 8 \
  --trainsize 384 \
  --clip 1.0 \
  --lr_sched cosine \
  --min_lr 1e-6 \
  --warmup_lr 1e-6 \
  --warmup_epochs 5 \
  --decay_rate 0.1 \
  --decay_epoch 100 \
  --load_pre "$LOAD_PRE" \
  --gpu_id "$GPU_ID" \
  --image_root "$IMAGE_ROOT_MODE" \
  --test_start_epoch 1 \
  --resume "" \
  --resume_mode none \
  --test_rgb_root "$TEST_RGB_ROOT" \
  --test_depth_root "$TEST_DEPTH_ROOT" \
  --test_gt_root "$TEST_GT_ROOT" \
  --save_path "$SAVE_DIR/"
