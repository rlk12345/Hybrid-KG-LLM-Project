#!/bin/bash
# Complete test workflow - verifies the entire pipeline works
# This script tests: dataset generation -> training -> prediction -> evaluation

set -e

echo "=========================================="
echo "Complete Test Workflow"
echo "=========================================="
echo ""
echo "This script will test the complete pipeline:"
echo "  1. Generate a small test dataset"
echo "  2. Train a model (GPT-2, safe on any machine)"
echo "  3. Generate predictions"
echo "  4. Evaluate results"
echo ""
echo "Expected time: 5-10 minutes"
echo ""

# Check if we're in the right directory
if [ ! -f "scripts/generate_all_datasets.sh" ]; then
    echo "ERROR: Please run this script from the clean_repo directory"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.10+"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: Virtual environment not activated."
    echo "Activating virtual environment..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "ERROR: Virtual environment not found. Run QUICK_START.sh first."
        exit 1
    fi
fi

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "✓ PYTHONPATH set"
echo ""

# Step 1: Generate test dataset
echo "=========================================="
echo "Step 1: Generating Test Dataset"
echo "=========================================="
echo "Creating a small test dataset (20 samples) for quick testing..."
echo ""

python3 scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/test_workflow \
  --limit 20 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42 \
  --no_images

echo ""
echo "✓ Test dataset created in data/test_workflow/"
echo "  - Train: $(wc -l < data/test_workflow/train.jsonl) samples"
echo "  - Val: $(wc -l < data/test_workflow/val.jsonl) samples"
echo "  - Test: $(wc -l < data/test_workflow/test.jsonl) samples"
echo ""

# Step 2: Train model
echo "=========================================="
echo "Step 2: Training Model"
echo "=========================================="
echo "Training GPT-2 model (small, safe on any machine)..."
echo "This will take 2-5 minutes..."
echo ""

python3 << 'PYTHON_SCRIPT'
from src.hybrid_dpo import train_hybrid_dpo

print("Starting training...")
train_hybrid_dpo({
    'data': {
        'train_path': 'data/test_workflow/train.jsonl',
        'eval_path': 'data/test_workflow/val.jsonl'
    },
    'dpo': {
        'output_dir': 'outputs/test_workflow',
        'num_train_epochs': 1,
        'per_device_train_batch_size': 1,
        'learning_rate': 5e-6,
        'logging_steps': 1,
        'save_steps': 1000
    }
})
print("✓ Training completed!")
PYTHON_SCRIPT

echo ""
echo "✓ Model trained and saved to outputs/test_workflow/"
echo ""

# Step 3: Generate predictions
echo "=========================================="
echo "Step 3: Generating Predictions"
echo "=========================================="
echo "Generating predictions on test set..."
echo ""

python3 scripts/generate_predictions.py \
  --model_path outputs/test_workflow \
  --test_jsonl data/test_workflow/test.jsonl \
  --output_jsonl outputs/test_workflow_predictions.jsonl \
  --max_new_tokens 20

echo ""
echo "✓ Predictions saved to outputs/test_workflow_predictions.jsonl"
echo ""

# Step 4: Evaluate
echo "=========================================="
echo "Step 4: Evaluating Results"
echo "=========================================="
echo "Computing evaluation metrics..."
echo ""

python3 scripts/comprehensive_eval.py \
  --gold_jsonl data/test_workflow/test.jsonl \
  --pred_jsonl outputs/test_workflow_predictions.jsonl \
  --output_json outputs/test_workflow_results.json

echo ""
echo "✓ Evaluation complete!"
echo ""

# Step 5: Show results
echo "=========================================="
echo "Test Results"
echo "=========================================="
echo ""
if [ -f "outputs/test_workflow_results.json" ]; then
    python3 << 'PYTHON_SCRIPT'
import json
with open('outputs/test_workflow_results.json', 'r') as f:
    results = json.load(f)
    
print("Evaluation Metrics:")
print(f"  Accuracy: {results.get('accuracy', 'N/A'):.2%}")
if 'precision' in results:
    print(f"  Precision: {results.get('precision', 'N/A'):.2%}")
    print(f"  Recall: {results.get('recall', 'N/A'):.2%}")
    print(f"  F1 Score: {results.get('f1', 'N/A'):.2%}")
print(f"\nFull results saved to: outputs/test_workflow_results.json")
PYTHON_SCRIPT
else
    echo "Results file not found. Check for errors above."
fi

echo ""
echo "=========================================="
echo "✓ Complete Test Workflow Finished!"
echo "=========================================="
echo ""
echo "What happened:"
echo "  1. ✓ Created test dataset (data/test_workflow/)"
echo "  2. ✓ Trained model (outputs/test_workflow/)"
echo "  3. ✓ Generated predictions (outputs/test_workflow_predictions.jsonl)"
echo "  4. ✓ Evaluated results (outputs/test_workflow_results.json)"
echo ""
echo "Next steps:"
echo "  - Review results in outputs/test_workflow_results.json"
echo "  - Try generating all dataset variants: bash scripts/generate_all_datasets.sh"
echo "  - Train on paper evaluation dataset (see README.md)"
echo "  - Explore the code in src/ to understand the implementation"
echo ""

