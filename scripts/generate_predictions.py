#!/usr/bin/env python3
"""
Generate predictions from a trained model for evaluation.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def generate_predictions(model_path: str, test_jsonl: str, output_jsonl: str, max_new_tokens: int = 20):
    """
    Load trained model and generate predictions on test set.
    
    Args:
        model_path: Path to trained model directory
        test_jsonl: Path to test JSONL file
        output_jsonl: Path to save predictions
        max_new_tokens: Maximum tokens to generate
    """
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading test data from {test_jsonl}...")
    test_samples = []
    with open(test_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))
    
    print(f"Generating predictions for {len(test_samples)} samples...")
    predictions = []
    
    for sample in tqdm(test_samples, desc="Generating"):
        prompt = sample.get("prompt", "")
        chosen = sample.get("chosen", "")
        
        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode prediction
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (text after the prompt)
        if prompt in generated_text:
            new_text = generated_text[len(prompt):].strip()
        else:
            new_text = generated_text.strip()
        
        # Try to extract relation from the prompt context first
        # Look for pattern: -[relation]->
        import re
        relation_match = re.search(r'-\[([^\]]+)\]->', prompt)
        expected_relation = relation_match.group(1) if relation_match else None
        
        # Try to find relation word in generated text
        prediction = None
        if expected_relation:
            # Check if the relation word appears in generated text
            if expected_relation.lower() in new_text.lower():
                prediction = expected_relation
            # Check for variations (e.g., "treat" vs "treats")
            elif expected_relation.rstrip('s').lower() in new_text.lower():
                prediction = expected_relation
        
        # If not found, try to extract first meaningful word
        if not prediction:
            # Remove common words and take first content word
            words = new_text.split()
            stop_words = {'is', 'a', 'an', 'the', 'that', 'used', 'to', 'and', 'or', 'in', 'on', 'at'}
            content_words = [w for w in words if w.lower() not in stop_words and len(w) > 2]
            if content_words:
                prediction = content_words[0].strip('.,!?;:')
            else:
                prediction = words[0] if words else new_text[:20]
        
        predictions.append({
            "prompt": prompt,
            "prediction": prediction,
            "gold": chosen
        })
    
    # Save predictions
    print(f"Saving predictions to {output_jsonl}...")
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    
    print(f"✓ Generated {len(predictions)} predictions")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions from trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--test_jsonl", type=str, required=True, help="Path to test JSONL file")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Maximum tokens to generate")
    args = parser.parse_args()
    
    generate_predictions(
        model_path=args.model_path,
        test_jsonl=args.test_jsonl,
        output_jsonl=args.output_jsonl,
        max_new_tokens=args.max_new_tokens
    )

