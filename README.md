# Hybrid-KG-LLM-Project

Hybrid multi-hop reasoning over knowledge graphs with LLM alignment and optional vision grounding. This repository combines ideas and utilities from SNS, GITA, and GraphWiz, and adds a tailored hybrid DPO training and data tooling for KG reasoning.

**References:**
- [SNS](https://github.com/ruili33/SNS)
- [GITA](https://github.com/WEIYanbin1999/GITA)
- [GraphWiz](https://github.com/Graph-Reasoner/Graph-Reasoning-LLM)

## Overview

- **Goal**: Train and evaluate an LLM (with optional visual cues) to perform multi-hop reasoning over a KG using a hybrid dataset and preference-alignment (DPO).
- **Data tooling**: Build synthetic and subset datasets from PRIMEKG, render small graph neighborhoods, and create DPO pairs.
- **Training**: A simple entrypoint for DPO fine-tuning on hybrid KG reasoning examples.
- **Evaluation**: Scripts for link prediction and multi-hop QA on generated/test splits.

## High-level Architecture

1. Data is prepared from raw KG triples and entity texts using `scripts/prepare_hybrid_dataset.py`.
2. Optional neighbor ranking with SimCSE (see `src/sns_ranker.py`) to focus candidate paths.
3. DPO training with `src/hybrid_dpo.py` aligns the model on positive vs negative reasoning chains.
4. Evaluation (`scripts/eval_link_prediction.py`, `scripts/eval_multihop_qa.py`) measures performance.
5. Visualization (`src/kg_visualize.py`) renders small subgraphs for qualitative analysis.

## Prerequisites

### System Dependencies

**Required:**
- Python 3.10 or higher
- Git
- Graphviz (for graph visualization)

**Install Graphviz:**

- **macOS**: `brew install graphviz`
- **Ubuntu/Debian**: `sudo apt-get install graphviz`
- **Windows**: Download from [Graphviz website](https://graphviz.org/download/) or use `choco install graphviz`

### Python Environment

Create and activate a virtual environment:

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n hybridkg python=3.10
conda activate hybridkg
```

## Installation

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd Hybrid-KG-LLM-Project
```

### Step 2: Install PyTorch

Install PyTorch matching your system (CPU or CUDA). Visit [PyTorch website](https://pytorch.org/get-started/locally/) for the correct command, or use:

```bash
# CPU only
pip install torch torchvision torchaudio

# CUDA (adjust version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set PYTHONPATH

**Important**: You must set the PYTHONPATH to the project root for imports to work correctly.

```bash
# Linux/macOS - add to ~/.bashrc or ~/.zshrc for persistence
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run in each terminal session:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Windows PowerShell
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD"

# Windows CMD
set PYTHONPATH=%PYTHONPATH%;%CD%
```

**Verify installation:**

```bash
# Quick verification script
python verify_setup.py

# Or manually test imports
python -c "from src.config import HybridConfig; print('Import successful!')"
```

**Note:** The scripts in `scripts/` automatically add the project root to `sys.path`, so they work without setting PYTHONPATH. However, for interactive Python sessions or custom scripts, you should set PYTHONPATH as shown above.

## Quick Start (Testing the Setup)

**Step 1: Verify Installation**

```bash
# Run the verification script
python verify_setup.py
```

**See also:** `SETUP_CHECKLIST.md` for a detailed verification checklist.

This checks:
- All Python imports work correctly
- Required dependencies are installed
- Sample data files are present
- Graphviz is installed

**Step 2: Test Data Preparation**

```bash
# Prepare a small test dataset (uses included sample data)
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid_test \
  --limit 10 \
  --no_images  # Skip image rendering for faster testing

# Verify the output
ls data/hybrid_test/  # Should show train.jsonl, val.jsonl, and test.jsonl
```

If both steps complete without errors, your setup is correct!

## Directory Structure

- `src/`
  - `config.py`: Centralized configuration helper and defaults for paths, model names, and training knobs.
  - `hybrid_dpo.py`: DPO training entrypoints/utilities for hybrid KG reasoning datasets.
  - `kg_data.py`: Lightweight KG data utilities: loading triples/entity texts, sampling neighborhoods, batching.
  - `prompting.py`: Prompt templates and formatting utilities for reasoning over KG facts.
  - `graphwiz_integration.py`: Hooks to generate or verify reasoning paths with GraphWiz-style methods when desired.
  - `sns_ranker.py`: SimCSE-based neighbor ranking to prioritize graph expansions.
  - `kg_visualize.py`: Small utilities to draw subgraphs used in datasets/evals.

- `scripts/`
  - `prepare_hybrid_dataset.py`: Build hybrid datasets (JSONL + rendered images) from triples; can sub-sample for demos.
  - `primekg_download.py`: Download PRIMEKG data (raw); large files are not pushed to the repo.
  - `primekg_subset.py`: Create smaller subsets from PRIMEKG for quick experiments.
  - `train_hybrid_dpo.sh` / `train_hybrid_dpo.ps1`: Convenience launchers for training.
  - `eval_link_prediction.py`: Evaluate link prediction on held-out edges/splits.
  - `eval_multihop_qa.py`: Evaluate multi-hop QA style tasks generated from subgraphs.

- `data/`
  - Tracked example datasets: `hybrid/`, `hybrid_simcse/`, `hybrid_simcse_default/`
  - Each dataset directory contains: `train.jsonl`, `val.jsonl`, `test.jsonl`, and `images/` folder
  - Sample files: `entity_texts.jsonl` and `sample_triples.jsonl` (included for testing)
  - Note: Large raw datasets (`data/primekg_raw/`) are intentionally ignored in Git. Use the download and subset scripts below to reproduce.

- `graphwiz_module/`, `gita_module/`, `sns_module/`, `third_party/`
  - Upstream references and scripts. These are included for reference but the main codebase uses only a subset of utilities.

## Data: Download, Subset, and Prepare

### Option 1: Use Included Sample Data (Quick Start)

The repository includes sample data files for immediate testing:

```bash
# Prepare a demo hybrid dataset (small sample with images)
# Creates train/val/test splits (80/10/10 by default)
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid \
  --limit 50

# Custom split ratios (must sum to 1.0)
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid \
  --limit 50 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42
```

### Option 2: Download PRIMEKG (Full Dataset)

**Step 1: Download PRIMEKG (raw, large; will NOT be committed):**

```bash
python scripts/primekg_download.py --target_dir third_party/PrimeKG
```

This clones the PrimeKG repository. The actual data files will be in the cloned repository.

**Step 2: Create a smaller subset for quick experiments:**

```bash
python scripts/primekg_subset.py --primekg_dir third_party/PrimeKG --out_dir data/primekg --limit_nodes 50000
```

**Step 3: Prepare hybrid dataset:**

```bash
# Creates train/val/test splits (80/10/10 by default)
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/subset_triples.jsonl \
  --out_dir data/hybrid \
  --limit 1000
```

**SimCSE-assisted variants** are controlled via flags (see `--help`) and implemented in `src/sns_ranker.py`:

```bash
python scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid_simcse \
  --limit 50 \
  --use_sns \
  --entity_texts_jsonl data/entity_texts.jsonl \
  --sns_top_k 5 \
  --sns_threshold 0.8
```

## Training (Hybrid DPO)

### Python API

```python
from src.hybrid_dpo import train_hybrid_dpo

# Basic usage
train_hybrid_dpo({
    "data": {
        "train_path": "data/hybrid/train.jsonl",
        "eval_path": "data/hybrid/val.jsonl"
    },
    "dpo": {
        "output_dir": "outputs/hybrid-dpo",
        "num_train_epochs": 2,
        "per_device_train_batch_size": 4,
        "learning_rate": 5e-6
    }
})

# Advanced: override model, SNS settings, etc.
train_hybrid_dpo({
    "model": {
        "base_model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2"
    },
    "data": {
        "train_path": "data/hybrid/train.jsonl",
        "eval_path": "data/hybrid/val.jsonl"
    },
    "sns": {
        "similarity_threshold": 0.8
    },
    "dpo": {
        "output_dir": "outputs/hybrid-dpo",
        "beta": 0.5,
        "num_train_epochs": 2
    }
})
```

### CLI Launcher (Shell Script)

**Linux/macOS:**

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TRAIN_JSONL="data/hybrid/train.jsonl"
export EVAL_JSONL="data/hybrid/val.jsonl"
export OUTPUT_DIR="outputs/hybrid-dpo"
export EPOCHS=2
export BSZ=4
export LR=5e-6

bash scripts/train_hybrid_dpo.sh
```

**Windows PowerShell:**

```powershell
$env:PYTHONPATH = "$env:PYTHONPATH;$PWD"
$env:TRAIN_JSONL = "data/hybrid/train.jsonl"
$env:EVAL_JSONL = "data/hybrid/val.jsonl"
$env:OUTPUT_DIR = "outputs/hybrid-dpo"
$env:EPOCHS = 2
$env:BSZ = 4
$env:LR = 5e-6

pwsh scripts/train_hybrid_dpo.ps1
```

**Artifacts** (checkpoints, logs) are written under `outputs/`.

## Evaluation

### Link Prediction

The link prediction evaluation script expects a predictions file with rank information:

```bash
# First, generate predictions (you need to implement inference separately)
# The predictions file should be JSONL with fields: head, relation, tail_rank

# Then evaluate:
python scripts/eval_link_prediction.py \
  --triples_jsonl data/sample_triples.jsonl \
  --predictions_jsonl outputs/predictions.jsonl
```

**Output:** JSON with `MRR` and `Hits@10` metrics.

### Multi-hop QA

The multi-hop QA evaluation script compares predictions to gold answers:

```bash
# First, generate predictions (you need to implement inference separately)
# The predictions file should be JSONL with field: prediction
# The gold file should be JSONL with fields: prompt, answer

# Evaluate on validation set:
python scripts/eval_multihop_qa.py \
  --gold_jsonl data/hybrid/val.jsonl \
  --pred_jsonl outputs/qa_predictions_val.jsonl

# Evaluate on test set (final evaluation):
python scripts/eval_multihop_qa.py \
  --gold_jsonl data/hybrid/test.jsonl \
  --pred_jsonl outputs/qa_predictions_test.jsonl
```

**Output:** JSON with `Accuracy` metric.

**Note:** These evaluation scripts compute metrics from pre-generated predictions. You'll need to implement model inference separately to generate the prediction files. Use the validation set for hyperparameter tuning and the test set only for final evaluation.

## Visualization

Render small subgraphs for inspection:

```bash
python -c "from src.kg_visualize import render_example; render_example('data/hybrid/train.jsonl', 0)"
```

Or use the visualization utilities programmatically:

```python
from src.kg_visualize import render_kg
from src.kg_data import read_triples_jsonl

triples = read_triples_jsonl("data/sample_triples.jsonl")
# Render first 10 triples as a graph
edges = triples[:10]
render_kg(edges, "output_graph.png")
```

## Configuration

- Start with `src/config.py` for default knobs and path conventions.
- Prompts and formatting live in `src/prompting.py`.
- Neighbor ranking options in `src/sns_ranker.py`.

All configuration can be overridden via the `train_hybrid_dpo()` function's dictionary argument. See the dataclasses in `src/config.py` for available options.

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Make sure PYTHONPATH is set:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Graphviz Errors

**Problem:** `FileNotFoundError: [Errno 2] No such file or directory: 'dot'`

**Solution:** Install Graphviz system package (see Prerequisites section).

### CUDA/GPU Issues

**Problem:** CUDA out of memory or GPU not detected

**Solution:**
- Reduce batch size in training config: `"per_device_train_batch_size": 2`
- Use CPU: Set device to `"cpu"` in SNS config
- Use gradient accumulation: Increase `"gradient_accumulation_steps"`

### Missing Data Files

**Problem:** `FileNotFoundError: data/sample_triples.jsonl`

**Solution:** The sample files should be included in the repository. If missing, check that you cloned the full repository. Alternatively, download PRIMEKG data using the scripts in the Data section.

### Windows Line Endings

**Problem:** Scripts fail on Windows with line ending errors

**Solution:**
```bash
git config core.autocrlf true
```

### Model Download Issues

**Problem:** Hugging Face model download fails

**Solution:**
- Check internet connection
- Set `HF_HOME` environment variable if using custom cache location
- For large models, consider using `huggingface-cli` to download manually

## For Readers Exploring the Code

Recommended order to read:
1. `src/config.py` → how configuration is passed.
2. `src/kg_data.py` → how triples and entity texts are loaded and batched.
3. `src/prompting.py` → how inputs are formatted for the model.
4. `src/hybrid_dpo.py` → main training and DPO setup.
5. `src/graphwiz_integration.py` → optional path-finding hooks.
6. `src/sns_ranker.py` → SimCSE ranking.
7. `src/kg_visualize.py` → diagnostics and plots.

## Large Files and Git LFS

- `data/primekg_raw/` is ignored to avoid exceeding GitHub's 100MB file limit.
- If you need to track large artifacts, consider Git LFS (`https://git-lfs.github.com`).
- Otherwise, use the provided download/subset scripts to reproduce datasets locally.

## Novelty/Contributions

This work extends SNS, GITA, and GraphWiz with an end-to-end, reproducible pipeline for PrimeKG-scale knowledge graphs. Key innovations include:

- **Dual-stage caching system**: 12-character subset IDs, rich cache manifests, and flat run manifests, ensuring deterministic reruns across Roar and local machines.

- **HPC-aware rendering workflow**: Throttles Matplotlib/NetworkX image generation with `--max_images`, `--num_workers`, and guards visual regressions via PNG snapshot tests.

- **Groq-driven DPO ablation tooling**: Configurable threshold/β grids with summary stats/plots, tailored to KG data.

See `experiments/` for ablation results.

## Licenses and Acknowledgements

- Original third-party licenses are retained in `third_party_licenses/`.
- This repo builds upon SNS, GITA, and GraphWiz; please cite and follow their licenses.

## Contact

For issues or questions, please open an issue on the repository.
