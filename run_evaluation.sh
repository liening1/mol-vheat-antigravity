#!/bin/bash
# Run comprehensive model evaluation experiments
# Usage: ./run_evaluation.sh [dataset]

DATASET=${1:-esol}
MODEL_PATH="checkpoints/${DATASET}_best.pth"
OUTPUT_DIR="periodic reports/evaluation_${DATASET}_$(date +%Y%m%d)"

echo "=============================================="
echo "Mol-vHeat Comprehensive Evaluation"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    # Try alternative path
    MODEL_PATH="logs/esol_ep400_lr2e-5_1231_1005/best_model.pth"
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Error: Model checkpoint not found!"
        echo "Please specify a valid model path."
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Step 1: Running comprehensive evaluation..."
python src/evaluate.py --model "$MODEL_PATH" --dataset "$DATASET" --output "$OUTPUT_DIR"

echo ""
echo "Step 2: Running cross-validation (5-fold)..."
python src/benchmark.py --model "$MODEL_PATH" --dataset "$DATASET" --output "$OUTPUT_DIR" --cv --cv-folds 5

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
