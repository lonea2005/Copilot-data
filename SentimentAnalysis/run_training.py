#!/usr/bin/env python3

"""
Full training script for sentiment analysis
"""

import argparse
import sys
sys.path.append('.')

from main import main

if __name__ == "__main__":
    # You can run this script with custom arguments or use the defaults
    
    # Example: python run_training.py --epochs 5 --batch_size 16 --lr_encoder 2e-5
    
    # If no arguments provided, run with good default settings
    if len(sys.argv) == 1:
        print("Running with default parameters...")
        sys.argv.extend([
            "--model_name", "bert-base-uncased",
            "--epochs", "10",
            "--batch_size", "4", 
            "--max_length", "64",
            "--dropout", "0.1",
            "--lr_encoder", "2e-5",
            "--lr_head", "1e-4", 
            "--warmup_ratio", "0.1"
        ])
    
    # Run the main function
    main()