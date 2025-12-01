#!/bin/bash
# Demo script to verify the project works on standard machines
# This runs all non-training components that should work on any laptop

set -e

echo "=========================================="
echo "Hybrid-KG-LLM Project Demo"
echo "=========================================="
echo ""

# Step 1: Verify setup
echo "Step 1: Verifying setup..."
python3 verify_setup.py
echo ""

# Step 2: Prepare dataset
echo "Step 2: Preparing dataset (train/val/test splits)..."
python3 scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/demo \
  --limit 20 \
  --no_images

echo "✓ Dataset created"
echo "  Train samples: $(wc -l < data/demo/train.jsonl)"
echo "  Val samples: $(wc -l < data/demo/val.jsonl)"
echo "  Test samples: $(wc -l < data/demo/test.jsonl)"
echo ""

# Step 3: Create visualization
echo "Step 3: Creating sample visualization..."
python3 -c "
from src.kg_visualize import render_kg
from src.kg_data import read_triples_jsonl

triples = read_triples_jsonl('data/sample_triples.jsonl')[:5]
render_kg(triples, 'demo_visualization.png')
print('✓ Visualization saved to demo_visualization.png')
"
echo ""

# Step 4: Test training with small model (optional, but safe)
echo "Step 4: Testing training with small model (GPT-2, safe on any machine)..."
echo "  This will take a few minutes but won't crash your system..."
python3 -c "
from src.hybrid_dpo import train_hybrid_dpo

train_hybrid_dpo({
    'data': {
        'train_path': 'data/demo/train.jsonl',
        'eval_path': 'data/demo/val.jsonl'
    },
    'dpo': {
        'output_dir': 'outputs/demo',
        'num_train_epochs': 1,
        'per_device_train_batch_size': 1,
        'learning_rate': 5e-6,
        'logging_steps': 1,
        'save_steps': 1000  # Don't save during demo
    }
})
"
echo "✓ Training completed successfully"
echo ""

# Step 5: Summary
echo "=========================================="
echo "Demo completed successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - data/demo/train.jsonl"
echo "  - data/demo/val.jsonl"
echo "  - data/demo/test.jsonl"
echo "  - demo_visualization.png"
echo "  - outputs/demo/ (training outputs)"
echo ""
echo "All components are working correctly!"

