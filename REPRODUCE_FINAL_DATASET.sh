#!/bin/bash
# Simple script to reproduce the final dataset
# This is what was used for the research paper submission

set -e

echo "=========================================="
echo "Reproducing Final Dataset"
echo "=========================================="
echo ""

# Check if source data exists
if [ ! -f "data/sample_triples.jsonl" ]; then
    echo "ERROR: data/sample_triples.jsonl not found!"
    echo "Please ensure the repository is complete."
    exit 1
fi

echo "Creating final dataset (data/paper_eval/)..."
echo "  Source: data/sample_triples.jsonl"
echo "  Output: data/paper_eval/"
echo "  Samples: 200 (140 train, 30 val, 30 test)"
echo ""

python3 scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/paper_eval \
  --limit 200 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42 \
  --no_images

echo ""
echo "=========================================="
echo "✓ Final dataset created successfully!"
echo "=========================================="
echo ""
echo "Dataset location: data/paper_eval/"
echo ""
echo "Verification:"
wc -l data/paper_eval/*.jsonl
echo ""
echo "Expected:"
echo "  140 data/paper_eval/train.jsonl"
echo "   30 data/paper_eval/val.jsonl"
echo "   30 data/paper_eval/test.jsonl"
echo ""



