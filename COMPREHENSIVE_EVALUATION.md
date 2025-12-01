# Comprehensive Evaluation Plan for Research Paper

## Current Problem
- Only 1 test sample (way too small!)
- Need proper train/val/test splits
- Need multiple evaluation metrics
- Need baseline comparisons
- Need statistical significance

## What You Need for a Research Paper

### 1. Proper Dataset Sizes
- **Training set:** 100-1000+ samples
- **Validation set:** 20-100+ samples  
- **Test set:** 20-100+ samples
- **Total:** Enough to show statistical significance

### 2. Multiple Evaluation Metrics
- Accuracy (what you have)
- Precision, Recall, F1
- Per-relation accuracy
- Error analysis

### 3. Baseline Comparisons
- Untrained model (zero-shot)
- Random baseline
- Rule-based baseline
- Your trained model

### 4. Ablation Studies
- With/without SimCSE
- Different hyperparameters
- Different model sizes

### 5. Statistical Analysis
- Confidence intervals
- Significance tests
- Multiple runs with different seeds

## Action Plan

### Step 1: Create Larger Dataset
```bash
# Create dataset with proper splits
python3 scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/paper_dataset \
  --limit 100 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42
```

This gives you:
- ~70 train samples
- ~15 val samples
- ~15 test samples

### Step 2: Enhanced Evaluation Script
Need to create evaluation that computes:
- Overall accuracy
- Per-relation accuracy
- Precision/Recall/F1
- Confusion matrix
- Error examples

### Step 3: Baseline Comparisons
- Zero-shot (no training)
- Random baseline
- Your trained model

### Step 4: Multiple Runs
- Train with different seeds
- Report mean ± std
- Show statistical significance

