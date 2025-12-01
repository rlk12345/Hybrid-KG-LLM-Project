#!/usr/bin/env python3
"""
Retrain with better logging to get a proper training curve.
"""
from src.hybrid_dpo import train_hybrid_dpo

print("Training with detailed logging (will show training curve)...")
train_hybrid_dpo({
    'data': {
        'train_path': 'data/hybrid/train.jsonl',
        'eval_path': 'data/hybrid/val.jsonl'
    },
    'dpo': {
        'output_dir': 'outputs/paper-results-detailed',
        'num_train_epochs': 2,
        'per_device_train_batch_size': 1,
        'per_device_eval_batch_size': 1,
        'learning_rate': 5e-6,
        'logging_steps': 1,  # Log every step for detailed curve
        'save_steps': 1000,
        'gradient_accumulation_steps': 1  # More steps = more data points
    }
})

