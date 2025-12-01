#!/bin/bash
# Complete setup for research paper evaluation
# This creates a proper dataset and evaluation framework

set -e

echo "=========================================="
echo "Research Paper Evaluation Setup"
echo "=========================================="
echo ""

# Step 1: Check if we need more data
echo "Step 1: Checking available data..."
SAMPLE_COUNT=$(wc -l < data/sample_triples.jsonl)
echo "  Current sample data: $SAMPLE_COUNT triples"

if [ "$SAMPLE_COUNT" -lt 50 ]; then
    echo ""
    echo "⚠️  WARNING: You only have $SAMPLE_COUNT triples!"
    echo "   For a research paper, you need 100-1000+ triples."
    echo ""
    echo "Options:"
    echo "  1. Generate synthetic data (recommended - quick)"
    echo "  2. Use existing sample data (acknowledge limitations)"
    echo ""
    read -p "Generate synthetic data? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Generating synthetic dataset..."
        python3 scripts/generate_synthetic_data.py --output data/synthetic_triples.jsonl --num_triples 200 --seed 42
        TRIPLES_SOURCE="data/synthetic_triples.jsonl"
        echo "  ✓ Synthetic data created"
    else
        TRIPLES_SOURCE="data/sample_triples.jsonl"
        echo "  Using sample data (will be limited - mention in paper)"
    fi
else
    TRIPLES_SOURCE="data/sample_triples.jsonl"
fi

# Step 2: Create proper dataset with good splits
echo ""
echo "Step 2: Creating dataset with proper train/val/test splits..."
python3 scripts/prepare_hybrid_dataset.py \
  --triples_jsonl "$TRIPLES_SOURCE" \
  --out_dir data/paper_eval \
  --limit 100 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42

TRAIN_COUNT=$(wc -l < data/paper_eval/train.jsonl)
VAL_COUNT=$(wc -l < data/paper_eval/val.jsonl)
TEST_COUNT=$(wc -l < data/paper_eval/test.jsonl)

echo "  ✓ Dataset created:"
echo "    Train: $TRAIN_COUNT samples"
echo "    Val: $VAL_COUNT samples"
echo "    Test: $TEST_COUNT samples"

# Step 3: Retrain model on proper dataset
echo ""
echo "Step 3: Training model on proper dataset..."
echo "  This will take 10-20 minutes..."
python3 << 'PYTHON_SCRIPT'
from src.hybrid_dpo import train_hybrid_dpo

train_hybrid_dpo({
    'data': {
        'train_path': 'data/paper_eval/train.jsonl',
        'eval_path': 'data/paper_eval/val.jsonl'
    },
    'dpo': {
        'output_dir': 'outputs/paper_eval_model',
        'num_train_epochs': 3,
        'per_device_train_batch_size': 1,
        'per_device_eval_batch_size': 1,
        'learning_rate': 5e-6,
        'logging_steps': 1,
        'save_steps': 1000,
        'gradient_accumulation_steps': 2
    }
})
PYTHON_SCRIPT

# Step 4: Generate predictions
echo ""
echo "Step 4: Generating predictions on test set..."
python3 scripts/generate_predictions.py \
  --model_path outputs/paper_eval_model \
  --test_jsonl data/paper_eval/test.jsonl \
  --output_jsonl outputs/paper_eval_predictions.jsonl \
  --max_new_tokens 20

# Step 5: Comprehensive evaluation
echo ""
echo "Step 5: Running comprehensive evaluation..."
python3 scripts/comprehensive_eval.py \
  --gold_jsonl data/paper_eval/test.jsonl \
  --pred_jsonl outputs/paper_eval_predictions.jsonl \
  --output_json outputs/paper_eval_results.json

echo ""
echo "=========================================="
echo "✓ Evaluation setup complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - outputs/paper_eval_results.json (detailed metrics)"
echo "  - outputs/paper_eval_predictions.jsonl (all predictions)"
echo ""
echo "Next steps:"
echo "  1. Review results in outputs/paper_eval_results.json"
echo "  2. Run baseline comparisons (see run_baselines.sh)"
echo "  3. Create visualizations for paper"

