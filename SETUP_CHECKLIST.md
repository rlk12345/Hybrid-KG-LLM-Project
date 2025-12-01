# Setup Verification Checklist

Use this checklist to verify the project can run on a fresh machine.

## Pre-flight Checks

- [ ] Python 3.10+ is installed (`python3 --version`)
- [ ] Git is installed (`git --version`)
- [ ] Graphviz is installed (`dot -V`)
- [ ] Virtual environment is created and activated

## Installation Steps

- [ ] Repository cloned successfully
- [ ] PyTorch installed (CPU or CUDA version)
- [ ] All Python dependencies installed (`pip install -r requirements.txt`)
- [ ] PYTHONPATH set (or using scripts that auto-configure it)

## Verification

- [ ] Run `python verify_setup.py` - all checks pass
- [ ] Sample data files exist:
  - [ ] `data/sample_triples.jsonl`
  - [ ] `data/entity_texts.jsonl`

## Quick Test

- [ ] Data preparation works:
  ```bash
  python scripts/prepare_hybrid_dataset.py \
    --triples_jsonl data/sample_triples.jsonl \
    --out_dir data/hybrid_test \
    --limit 10 \
    --no_images
  ```
- [ ] Output files created:
  - [ ] `data/hybrid_test/train.jsonl`
  - [ ] `data/hybrid_test/val.jsonl`

## If All Checks Pass

The project is ready to use! You can now:
1. Prepare larger datasets
2. Run training
3. Evaluate models

## Common Issues

If verification fails, check:
1. **Import errors**: Set PYTHONPATH to project root
2. **Graphviz errors**: Install system Graphviz package
3. **Missing data**: Sample files should be in repository
4. **CUDA issues**: Install correct PyTorch version for your system

See README.md Troubleshooting section for detailed solutions.

