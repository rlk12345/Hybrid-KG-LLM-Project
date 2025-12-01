# Research Paper Output Guide

This guide walks you through generating all the outputs needed for your research paper.

## Overview of Outputs You'll Generate

1. **Prepared Datasets**: Train/val/test splits with DPO pairs
2. **Visualizations**: Graph images for qualitative analysis
3. **Training Results**: Model checkpoints and training metrics
4. **Evaluation Metrics**: Link prediction and QA accuracy scores
5. **Ablation Results**: Hyperparameter sweep results (if running ablations)

---

## Step-by-Step Workflow

### Step 1: Prepare Your Dataset

First, let's create a dataset with train/val/test splits. You can use the included sample data or prepare a larger dataset.

#### Option A: Quick Test (Small Dataset - Fast)

```bash
# Prepare a small test dataset (10 samples, no images for speed)
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid_test \
  --limit 10 \
  --no_images

# Check what was created
ls -lh data/hybrid_test/
```

**Expected Output:**
- `train.jsonl` - Training examples (80% of data)
- `val.jsonl` - Validation examples (10% of data)  
- `test.jsonl` - Test examples (10% of data)

#### Option B: Full Dataset with Images (For Paper)

```bash
# Prepare a larger dataset with visualizations
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid \
  --limit 100 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42

# Check output
ls -lh data/hybrid/
ls data/hybrid/images/ | head -5  # Check some images
```

**Expected Output:**
- `train.jsonl` - Training examples
- `val.jsonl` - Validation examples
- `test.jsonl` - Test examples
- `images/` - Directory with graph visualizations (PNG files)

#### Option C: With SimCSE Ranking (Advanced)

```bash
# Prepare dataset with SimCSE-based neighbor selection
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid_simcse \
  --limit 100 \
  --use_sns \
  --entity_texts_jsonl data/entity_texts.jsonl \
  --sns_top_k 5 \
  --sns_threshold 0.8 \
  --seed 42
```

---

### Step 2: Inspect Your Dataset

Let's verify the dataset structure:

```bash
# Check dataset statistics
echo "Train samples:"
wc -l data/hybrid/train.jsonl

echo "Val samples:"
wc -l data/hybrid/val.jsonl

echo "Test samples:"
wc -l data/hybrid/test.jsonl

# Look at a sample entry
head -n 1 data/hybrid/train.jsonl | python3 -m json.tool
```

**What to note for your paper:**
- Dataset size (train/val/test split)
- Data format (prompt, chosen, rejected, image paths)
- Image count if using visualizations

---

### Step 3: Visualize Sample Graphs (For Paper Figures)

```bash
# Render a sample graph for inspection
python3 -c "
from src.kg_visualize import render_kg
from src.kg_data import read_triples_jsonl

# Load and visualize first few triples
triples = read_triples_jsonl('data/sample_triples.jsonl')[:5]
render_kg(triples, 'sample_graph.png')
print('Graph saved to sample_graph.png')
"

# View the graph (macOS)
open sample_graph.png
```

**Output for Paper:**
- Graph visualization images showing KG structure
- Can be included in methodology/experiments section

---

### Step 4: Run Training (DPO Fine-tuning)

**⚠️ CRITICAL WARNING:** Training large models (7B+ parameters) on a MacBook Air will **crash your system**. See `SAFE_TRAINING_GUIDE.md` for alternatives.

**Options:**
1. **Skip training** - Focus on dataset/methodology (RECOMMENDED for laptops)
2. **Use cloud resources** - Google Colab, Kaggle, or university HPC
3. **Use tiny models** - GPT-2 (124M) instead of Mistral-7B (7B)

**Note:** Training requires a GPU and can take hours. For a research paper, you might:
- Use a smaller model for quick experiments
- Run on a subset of data first
- Use pre-trained checkpoints if available
- **Focus on dataset contribution instead of training**

#### ⚠️ DO NOT RUN THIS ON MACBOOK AIR - WILL CRASH SYSTEM

#### Quick Training Test (Small Scale - Use Cloud Instead)

```bash
# Set environment variables for training
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run training with minimal config (for testing)
python3 -c "
from src.hybrid_dpo import train_hybrid_dpo

train_hybrid_dpo({
    'data': {
        'train_path': 'data/hybrid/train.jsonl',
        'eval_path': 'data/hybrid/val.jsonl'
    },
    'dpo': {
        'output_dir': 'outputs/hybrid-dpo-test',
        'num_train_epochs': 1,  # Just 1 epoch for testing
        'per_device_train_batch_size': 2,
        'per_device_eval_batch_size': 2,
        'learning_rate': 5e-6
    }
})
"
```

**Expected Output:**
- `outputs/hybrid-dpo-test/` directory with:
  - Model checkpoints
  - Training logs
  - `trainer_state.json` with metrics

#### Full Training (For Paper Results)

```bash
# Full training configuration
python3 -c "
from src.hybrid_dpo import train_hybrid_dpo

train_hybrid_dpo({
    'model': {
        'base_model_name_or_path': 'mistralai/Mistral-7B-Instruct-v0.2'
    },
    'data': {
        'train_path': 'data/hybrid/train.jsonl',
        'eval_path': 'data/hybrid/val.jsonl'
    },
    'dpo': {
        'output_dir': 'outputs/hybrid-dpo-full',
        'num_train_epochs': 2,
        'per_device_train_batch_size': 4,
        'per_device_eval_batch_size': 4,
        'learning_rate': 5e-6,
        'beta': 0.5,
        'gradient_accumulation_steps': 4
    }
})
"
```

**Outputs for Paper:**
- Training loss curves
- Validation metrics
- Model checkpoints
- Training time/epoch information

---

### Step 5: Run Ablation Studies (Optional but Recommended)

If you want to test different hyperparameters:

```bash
# Run ablation study (tests different thresholds and beta values)
python scripts/ablations/dpo_ablate.py \
  --train_jsonl data/hybrid/train.jsonl \
  --val_jsonl data/hybrid/val.jsonl \
  --thresholds 0.7,0.8,0.9 \
  --betas 0.3,0.5 \
  --seeds 42 \
  --max_samples 50 \
  --dry-run \
  --out_dir experiments/dpo/ablation

# Summarize results
python scripts/ablations/summarize_runs.py experiments/dpo/ablation
```

**Outputs for Paper:**
- Grid of results across hyperparameters
- Plots showing performance vs. hyperparameters
- Summary statistics

---

### Step 6: Evaluation

#### Link Prediction Evaluation

```bash
# First, you need to generate predictions (implement inference)
# For now, this shows the evaluation script structure

# The evaluation expects a predictions file with format:
# {"head": "...", "relation": "...", "tail_rank": 1}

# Example evaluation (once you have predictions):
python scripts/eval_link_prediction.py \
  --triples_jsonl data/hybrid/test.jsonl \
  --predictions_jsonl outputs/predictions.jsonl
```

**Outputs for Paper:**
- MRR (Mean Reciprocal Rank)
- Hits@10 scores
- Can compare different model variants

#### Multi-hop QA Evaluation

```bash
# Similar to link prediction, you need predictions first
# Format: {"prediction": "..."}

# Example evaluation:
python scripts/eval_multihop_qa.py \
  --gold_jsonl data/hybrid/test.jsonl \
  --pred_jsonl outputs/qa_predictions.jsonl
```

**Outputs for Paper:**
- Accuracy scores
- Per-question results (if you extend the script)

---

## What to Include in Your Paper

### 1. Dataset Section
- Dataset statistics (sizes, splits)
- Sample data examples
- Graph visualization examples

### 2. Experimental Setup
- Model architecture
- Training hyperparameters
- Hardware specifications

### 3. Results Section
- Training curves (loss over epochs)
- Evaluation metrics (MRR, Hits@10, Accuracy)
- Ablation study results (if run)
- Comparison with baselines

### 4. Qualitative Analysis
- Sample predictions
- Graph visualizations
- Error analysis

---

## Quick Reference: Key Commands

```bash
# 1. Prepare dataset
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid \
  --limit 100

# 2. Verify dataset
wc -l data/hybrid/*.jsonl

# 3. Train model
python3 -c "from src.hybrid_dpo import train_hybrid_dpo; train_hybrid_dpo({...})"

# 4. Evaluate (after generating predictions)
python scripts/eval_link_prediction.py --triples_jsonl data/hybrid/test.jsonl --predictions_jsonl outputs/preds.jsonl

# 5. Visualize
python3 -c "from src.kg_visualize import render_kg; from src.kg_data import read_triples_jsonl; render_kg(read_triples_jsonl('data/sample_triples.jsonl')[:5], 'fig.png')"
```

---

## Troubleshooting

**Out of Memory:**
- Reduce batch size: `'per_device_train_batch_size': 2`
- Use gradient accumulation: `'gradient_accumulation_steps': 8`
- Reduce dataset size: `--limit 50`

**Training Too Slow:**
- Use smaller model
- Reduce epochs: `'num_train_epochs': 1`
- Use `--no_images` flag when preparing data

**Need More Data:**
- Download PRIMEKG: `python scripts/primekg_download.py --target_dir third_party/PrimeKG`
- Create subset: `python scripts/primekg_subset.py --primekg_dir third_party/PrimeKG --out_dir data/primekg --limit_nodes 50000`

---

## Next Steps

1. ✅ Setup verified - DONE
2. ⬜ Prepare dataset
3. ⬜ Run training (or use pre-trained model)
4. ⬜ Generate predictions
5. ⬜ Evaluate and collect metrics
6. ⬜ Create visualizations for paper
7. ⬜ Write up results

Good luck with your research paper!

