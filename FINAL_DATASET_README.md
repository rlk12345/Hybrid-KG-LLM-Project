# Final Dataset for Submission

## Dataset Location

The final dataset used for the research paper is located in:
```
data/paper_eval/
```

## Dataset Contents

The final dataset consists of:
- **train.jsonl**: 140 training samples
- **val.jsonl**: 30 validation samples  
- **test.jsonl**: 30 test samples
- **images/**: Directory for graph visualizations (empty in final dataset - images were not generated to keep dataset size manageable)

**Total**: 200 samples

## Dataset Format

Each line in the JSONL files is a JSON object with the following structure:
```json
{
  "prompt": "Question: Given the KG snippet, what is the relation between Entity_X and Entity_Y?\nRelevant KG triples:\n- (Entity_X) -[relation]-> (Entity_Y)\nAnswer:",
  "chosen": "relation",
  "rejected": "unknown",
  "image": null
}
```

- **prompt**: The question and relevant KG triples
- **chosen**: The correct relation (what the model should learn to prefer)
- **rejected**: The incorrect relation (always "unknown" in this dataset)
- **image**: Path to graph visualization (null in final dataset)

## How to Reproduce the Final Dataset

If you need to regenerate the final dataset, run:

```bash
python3 scripts/prepare_hybrid_dataset.py \
  --triples_jsonl data/sample_triples.jsonl \
  --out_dir data/paper_eval \
  --limit 200 \
  --train_ratio 0.7 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42 \
  --no_images
```

**Note**: The `--no_images` flag skips image generation to keep the dataset simple and small. The dataset works perfectly without images.

## Verification

To verify the dataset is correct, check the line counts:
```bash
wc -l data/paper_eval/*.jsonl
```

Expected output:
- train.jsonl: 140 lines
- val.jsonl: 30 lines
- test.jsonl: 30 lines

## Dataset Statistics

- **Source**: `data/sample_triples.jsonl` (included in repository)
- **Split**: 70% train, 15% val, 15% test
- **Random seed**: 42 (for reproducibility)
- **Total samples**: 200
- **Images**: Not included (optional feature, not required for evaluation)

## Usage in Paper

This dataset was used for:
1. Training the DPO model (train.jsonl)
2. Validation during training (val.jsonl)
3. Final evaluation (test.jsonl) - results in `outputs/paper_eval_results.json`

The evaluation results show:
- Accuracy: 96.7% (29/30 correct)
- Precision: 92.9%
- Recall: 91.1%
- F1 Score: 91.8%

## For TAs/Reviewers

The final dataset is **already included** in the repository at `data/paper_eval/`. No additional steps are needed to use it. The dataset is self-contained and ready for evaluation.

To run evaluation on the final dataset:
```bash
python3 scripts/comprehensive_eval.py \
  --gold_jsonl data/paper_eval/test.jsonl \
  --pred_jsonl outputs/paper_eval_predictions.jsonl \
  --output_json outputs/paper_eval_results.json
```



