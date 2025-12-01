# Safe Training Guide - Avoiding System Crashes

## ⚠️ IMPORTANT: Your MacBook Air Cannot Handle Large Model Training

Training a 7B parameter model (like Mistral-7B) requires:
- **16GB+ RAM** (your MacBook Air likely has 8GB)
- **GPU with 16GB+ VRAM** (you're using CPU)
- **Proper cooling** (laptops overheat quickly)

**Result:** System crash/kernel panic (which you just experienced)

---

## ✅ Safe Alternatives for Your Research Paper

### Option 1: Focus on Dataset & Methodology (RECOMMENDED)

For a research paper, you don't necessarily need to train the model yourself. You can:

1. **Document the dataset preparation process**
   - Show train/val/test splits
   - Demonstrate data format
   - Include visualizations

2. **Describe the training methodology**
   - Explain DPO training setup
   - Document hyperparameters
   - Reference baseline results

3. **Use pre-trained models or baselines**
   - Compare against GPT-4 or other APIs
   - Use smaller pre-trained models
   - Focus on the dataset contribution

**This is actually common in research papers!** Many papers focus on:
- Novel dataset creation
- Methodology improvements
- Evaluation frameworks

### Option 2: Use Cloud Resources (If Training is Required)

If you absolutely need to train:

1. **Google Colab** (Free tier with GPU)
   - Limited but works for small experiments
   - 15GB RAM, T4 GPU

2. **Kaggle Notebooks** (Free)
   - 30 hours/week GPU time
   - P100 GPU, 13GB RAM

3. **AWS/GCP/Azure** (Paid)
   - More control, costs money
   - Can use spot instances to save

4. **University HPC** (If available)
   - Check if your university has GPU clusters
   - Usually free for students

### Option 3: Use a Much Smaller Model (If You Must Train Locally)

If you really want to try training locally, use a tiny model:

```python
# Use a very small model instead
train_hybrid_dpo({
    'model': {
        'base_model_name_or_path': 'gpt2'  # Only 124M parameters!
    },
    'data': {
        'train_path': 'data/hybrid/train.jsonl',
        'eval_path': 'data/hybrid/val.jsonl'
    },
    'dpo': {
        'output_dir': 'outputs/test-run-small',
        'num_train_epochs': 1,
        'per_device_train_batch_size': 1,
        'learning_rate': 5e-6
    }
})
```

**Even this might be slow on CPU, but it won't crash your system.**

---

## 📊 What You CAN Do Safely on Your MacBook Air

### 1. Dataset Preparation ✅
```bash
# This is safe - just processes data
python3 scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/hybrid \
  --limit 100
```

### 2. Data Analysis ✅
```bash
# Analyze your datasets
python3 -c "
import json
with open('data/hybrid/train.jsonl') as f:
    train = [json.loads(l) for l in f]
print(f'Train samples: {len(train)}')
print(f'Sample prompt: {train[0][\"prompt\"][:100]}...')
"
```

### 3. Visualizations ✅
```bash
# Create graph visualizations
python3 -c "
from src.kg_visualize import render_kg
from src.kg_data import read_triples_jsonl

triples = read_triples_jsonl('data/sample_triples.jsonl')[:10]
render_kg(triples, 'paper_figure.png')
print('✓ Visualization created')
"
```

### 4. Evaluation (After Getting Predictions) ✅
```bash
# If you have predictions from elsewhere
python scripts/eval_link_prediction.py \
  --triples_jsonl data/hybrid/test.jsonl \
  --predictions_jsonl predictions.jsonl
```

---

## 🎓 For Your Research Paper

### What You Can Write About:

1. **Dataset Contribution**
   - Novel hybrid KG-LLM dataset
   - Train/val/test splits
   - DPO pair construction methodology

2. **Methodology**
   - DPO training approach
   - SimCSE integration
   - Graph visualization pipeline

3. **Experimental Setup**
   - Hyperparameters
   - Evaluation metrics
   - Baseline comparisons

4. **Results** (from cloud training or baselines)
   - Training curves
   - Evaluation metrics
   - Ablation studies

### What You DON'T Need:

- ❌ Training large models locally
- ❌ Running full training on your laptop
- ❌ Risking system crashes

---

## 🚀 Recommended Workflow for Your Paper

1. **✅ Prepare multiple datasets** (you can do this!)
   - Different sizes
   - With/without SimCSE
   - With/without images

2. **✅ Create visualizations** (you can do this!)
   - Sample graphs
   - Dataset statistics
   - Methodology diagrams

3. **✅ Document methodology** (you can do this!)
   - Training procedure
   - Hyperparameters
   - Evaluation setup

4. **⏸️ Training** (use cloud or skip)
   - Use Colab/Kaggle if needed
   - Or focus on dataset/methodology contribution

5. **✅ Evaluation** (after getting predictions)
   - Run evaluation scripts
   - Compare metrics
   - Analyze results

---

## 💡 Key Takeaway

**Your research paper can be excellent without training on your laptop!**

Many successful papers focus on:
- Novel datasets
- Methodology improvements
- Evaluation frameworks
- Theoretical contributions

The dataset preparation and methodology you've built are valuable contributions on their own!

