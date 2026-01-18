#!/bin/bash
# A/B Test: Background Handling Effect on ESOL
# Compares training with background='none' vs background='mean'
#
# This creates separate output folders and does NOT affect existing checkpoints.

set -e

DATASET="esol"
EPOCHS=100  # Shorter for quick comparison
LR="2e-5"
BATCH_SIZE=32

echo "=============================================="
echo "A/B Test: Background Handling Effect"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS (quick test)"
echo ""

# Create experiment directory
EXP_DIR="experiments/background_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXP_DIR"

echo "Experiment directory: $EXP_DIR"
echo ""

# Test A: No background processing (current default)
echo "=============================================="
echo "[A] Training with background='none' (current default)"
echo "=============================================="
python src/train.py \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --background none \
    --output_dir "$EXP_DIR/none"

# Test B: Mean background replacement
echo ""
echo "=============================================="
echo "[B] Training with background='mean'"
echo "=============================================="
python src/train.py \
    --dataset $DATASET \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --background mean \
    --output_dir "$EXP_DIR/mean"

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "=============================================="
echo ""
echo "Results saved to: $EXP_DIR"
echo ""
echo "Compare results:"
echo "  A (none): $EXP_DIR/none/results.json"
echo "  B (mean): $EXP_DIR/mean/results.json"
