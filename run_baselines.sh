#!/bin/bash
# Run baseline comparisons for research paper

set -e

echo "=========================================="
echo "Baseline Comparisons"
echo "=========================================="
echo ""

TEST_DATA="data/paper_eval/test.jsonl"
OUTPUT_DIR="outputs/baselines"

mkdir -p "$OUTPUT_DIR"

# Baseline 1: Zero-shot (no training, use base GPT-2)
echo "Baseline 1: Zero-shot GPT-2 (no training)..."
python3 << 'PYTHON_SCRIPT'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load test data
with open("data/paper_eval/test.jsonl") as f:
    test_samples = [json.loads(line) for line in f if line.strip()]

predictions = []
for sample in tqdm(test_samples, desc="Zero-shot"):
    prompt = sample.get("prompt", "")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_text = generated[len(prompt):].strip() if prompt in generated else generated.strip()
    
    # Extract relation (same logic as generate_predictions.py)
    import re
    relation_match = re.search(r'-\[([^\]]+)\]->', prompt)
    expected_relation = relation_match.group(1) if relation_match else None
    
    prediction = None
    if expected_relation and expected_relation.lower() in new_text.lower():
        prediction = expected_relation
    else:
        words = new_text.split()
        stop_words = {'is', 'a', 'an', 'the', 'that', 'used', 'to', 'and'}
        content_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
        prediction = content_words[0] if content_words else (words[0] if words else new_text[:20])
    
    predictions.append({"prediction": prediction or "unknown"})

# Save
with open("outputs/baselines/zero_shot_predictions.jsonl", "w") as f:
    for pred in predictions:
        f.write(json.dumps(pred) + "\n")

print("✓ Zero-shot baseline complete")
PYTHON_SCRIPT

# Evaluate zero-shot
python3 scripts/comprehensive_eval.py \
  --gold_jsonl "$TEST_DATA" \
  --pred_jsonl "$OUTPUT_DIR/zero_shot_predictions.jsonl" \
  --output_json "$OUTPUT_DIR/zero_shot_results.json"

# Baseline 2: Random baseline
echo ""
echo "Baseline 2: Random baseline..."
python3 << 'PYTHON_SCRIPT'
import json
import random

# Get all relations from test set
relations = set()
with open("data/paper_eval/test.jsonl") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            rel = obj.get("chosen") or obj.get("answer", "")
            if rel:
                relations.add(rel)

relations = list(relations)

# Generate random predictions
predictions = []
with open("data/paper_eval/test.jsonl") as f:
    for line in f:
        if line.strip():
            predictions.append({"prediction": random.choice(relations)})

with open("outputs/baselines/random_predictions.jsonl", "w") as f:
    for pred in predictions:
        f.write(json.dumps(pred) + "\n")

print("✓ Random baseline complete")
PYTHON_SCRIPT

python3 scripts/comprehensive_eval.py \
  --gold_jsonl "$TEST_DATA" \
  --pred_jsonl "$OUTPUT_DIR/random_predictions.jsonl" \
  --output_json "$OUTPUT_DIR/random_results.json"

# Compare all baselines
echo ""
echo "=========================================="
echo "Baseline Comparison Summary"
echo "=========================================="
echo ""
echo "Zero-shot GPT-2:"
python3 -c "import json; r=json.load(open('outputs/baselines/zero_shot_results.json')); print(f\"  Accuracy: {r['overall']['accuracy']:.4f}\")"
echo ""
echo "Random baseline:"
python3 -c "import json; r=json.load(open('outputs/baselines/random_results.json')); print(f\"  Accuracy: {r['overall']['accuracy']:.4f}\")"
echo ""
echo "Your trained model:"
python3 -c "import json; r=json.load(open('outputs/paper_eval_results.json')); print(f\"  Accuracy: {r['overall']['accuracy']:.4f}\")"
echo ""

