#!/usr/bin/env python
"""
RF-DETR Training Script
Simplified version for single GPU training
"""
import os
import torch
from utils.config import load_config
from trainers.single_gpu import train_single_gpu

def train():
    """Main training function"""
    print("Loading config...")
    config = load_config()
    
    # Force single GPU mode
    config['distributed']['use_distributed'] = False
    if 'gpu_ids' in config['distributed']:
        config['distributed']['gpu_ids'] = [0]  # Use first visible GPU
    
    print("Starting single GPU training...")
    train_single_gpu(config)
    print("Training completed!")

if __name__ == "__main__":
    train() 