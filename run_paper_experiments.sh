#!/bin/bash
# Complete workflow to generate results for your research paper
# Safe to run on MacBook Air - uses GPT-2 (small model)

set -e

echo "=========================================="
echo "Research Paper Results Generation"
echo "=========================================="
echo ""

# Step 1: Verify setup
echo "Step 1: Verifying setup..."
python3 verify_setup.py
echo ""

# Step 2: Check/create dataset
echo "Step 2: Checking dataset..."
if [ ! -f "data/hybrid/train.jsonl" ]; then
    echo "  Creating dataset..."
    python3 scripts/prepare_hybrid_dataset.py \
      --triples_jsonl data/sample_triples.jsonl \
      --out_dir data/hybrid \
      --limit 50 \
      --seed 42
else
    echo "  Dataset already exists ✓"
fi

echo "  Dataset statistics:"
echo "    Train: $(wc -l < data/hybrid/train.jsonl) samples"
echo "    Val: $(wc -l < data/hybrid/val.jsonl) samples"
echo "    Test: $(wc -l < data/hybrid/test.jsonl) samples"
echo ""

# Step 3: Create visualization
echo "Step 3: Creating visualization for paper..."
python3 -c "
from src.kg_visualize import render_kg
from src.kg_data import read_triples_jsonl

triples = read_triples_jsonl('data/sample_triples.jsonl')[:8]
render_kg(triples, 'paper_figure_kg.png')
print('  ✓ Saved to paper_figure_kg.png')
"
echo ""

# Step 4: Run training
echo "Step 4: Running training (GPT-2, safe on laptop)..."
echo "  This will take 10-20 minutes..."
python3 << 'PYTHON_SCRIPT'
from src.hybrid_dpo import train_hybrid_dpo

print("  Starting training...")
train_hybrid_dpo({
    'data': {
        'train_path': 'data/hybrid/train.jsonl',
        'eval_path': 'data/hybrid/val.jsonl'
    },
    'dpo': {
        'output_dir': 'outputs/paper-results',
        'num_train_epochs': 2,
        'per_device_train_batch_size': 1,
        'per_device_eval_batch_size': 1,
        'learning_rate': 5e-6,
        'logging_steps': 1,
        'save_steps': 1000,
        'gradient_accumulation_steps': 4
    }
})
print("  ✓ Training completed!")
PYTHON_SCRIPT
echo ""

# Step 5: Show results
echo "Step 5: Results summary..."
echo "  Training outputs saved to: outputs/paper-results/"
if [ -f "outputs/paper-results/trainer_state.json" ]; then
    echo "  ✓ Training state saved"
fi
if [ -d "outputs/paper-results" ]; then
    echo "  Files created:"
    ls -lh outputs/paper-results/ | tail -5
fi
echo ""

echo "=========================================="
echo "✓ All steps completed successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - data/hybrid/ (dataset with train/val/test)"
echo "  - paper_figure_kg.png (visualization)"
echo "  - outputs/paper-results/ (training results)"
echo ""
echo "You now have results for your research paper!"

