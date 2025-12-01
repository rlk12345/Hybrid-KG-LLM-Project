#!/bin/bash
# Complete evaluation workflow: generate predictions and evaluate

set -e

echo "=========================================="
echo "Model Evaluation on Test Set"
echo "=========================================="
echo ""

MODEL_PATH="outputs/paper-results"
TEST_DATA="data/hybrid/test.jsonl"
PREDICTIONS="outputs/test_predictions.jsonl"
RESULTS="outputs/evaluation_results.json"

# Step 1: Generate predictions
echo "Step 1: Generating predictions from trained model..."
python3 scripts/generate_predictions.py \
  --model_path "$MODEL_PATH" \
  --test_jsonl "$TEST_DATA" \
  --output_jsonl "$PREDICTIONS" \
  --max_new_tokens 20

echo ""

# Step 2: Evaluate predictions
echo "Step 2: Evaluating predictions..."
python3 scripts/eval_multihop_qa.py \
  --gold_jsonl "$TEST_DATA" \
  --pred_jsonl "$PREDICTIONS" > "$RESULTS"

echo ""
echo "=========================================="
echo "Evaluation Results:"
echo "=========================================="
cat "$RESULTS"
echo ""
echo "Results saved to: $RESULTS"

