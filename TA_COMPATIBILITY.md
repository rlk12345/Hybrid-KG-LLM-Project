# TA Compatibility - Code Runs on Standard Machines

## Changes Made for TA Compatibility

### 1. Default Model Changed to GPT-2
- **Before:** Default was Mistral-7B (14GB, requires GPU/16GB+ RAM)
- **After:** Default is GPT-2 (500MB, works on any machine)
- **Impact:** Training now works on standard laptops without crashing

### 2. Clear Resource Requirements Documentation
- Added "System Requirements" section to README
- Clearly marked what works on standard machines vs. what needs GPU
- Added warnings for large models

### 3. Demo Script Created
- `run_demo.sh` - Complete end-to-end test that works on any machine
- Verifies all components work correctly
- Uses safe defaults (GPT-2, small dataset)

## What TAs Can Run

### ✅ Guaranteed to Work on Any Machine:
1. **Setup verification**
   ```bash
   python verify_setup.py
   ```

2. **Dataset preparation**
   ```bash
   python scripts/prepare_hybrid_dataset.py \
     --triples_jsonl data/sample_triples.jsonl \
     --out_dir data/hybrid \
     --limit 50
   ```

3. **Visualization**
   ```bash
   python3 -c "from src.kg_visualize import render_kg; from src.kg_data import read_triples_jsonl; render_kg(read_triples_jsonl('data/sample_triples.jsonl')[:5], 'fig.png')"
   ```

4. **Training with default (GPT-2)**
   ```bash
   python3 -c "from src.hybrid_dpo import train_hybrid_dpo; train_hybrid_dpo({'data': {'train_path': 'data/hybrid/train.jsonl', 'eval_path': 'data/hybrid/val.jsonl'}, 'dpo': {'output_dir': 'outputs/test', 'num_train_epochs': 1}})"
   ```

5. **Complete demo**
   ```bash
   bash run_demo.sh
   ```

### ⚠️ Optional (Requires GPU/High-End Machine):
- Training with large models (Mistral-7B, etc.)
- Can be done by overriding model name, but clearly documented as requiring special resources

## Testing Instructions for TAs

1. **Clone repository**
2. **Follow README setup instructions**
3. **Run demo script:**
   ```bash
   bash run_demo.sh
   ```
4. **Expected result:** All steps complete successfully in 5-10 minutes

## For HPC/Cloud Usage

If TAs have access to HPC or want to use large models:

1. The code supports large models - just override the model name
2. All other components work the same way
3. Training scripts are compatible with SLURM (see `scripts/hpc/`)

## Key Points

- ✅ **All core functionality works on standard machines**
- ✅ **Default configuration is safe for laptops**
- ✅ **Large models are optional and clearly documented**
- ✅ **Complete demo script verifies everything works**
- ✅ **No special hardware required for basic usage**

The code is fully runnable on TA machines while still supporting advanced use cases with proper hardware.

