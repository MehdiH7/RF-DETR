#!/usr/bin/env python
"""
RF-DETR Training Analysis Tool

This script analyzes training results and creates visualizations for the metrics
captured during training. It loads the training history from a specified run
directory and generates plots for:
- Training and validation loss curves
- Accuracy metrics over time
- Other model-specific metrics

All plots are saved to a 'plots' subdirectory within the run folder.
"""

import argparse
import json
from pathlib import Path
import glob

from utils.analysis import analyze_training_results


def find_latest_run(base_dir="output"):
    """Find the most recent training run directory"""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory {base_dir} not found")
    
    run_dirs = sorted(glob.glob(str(base_dir / "run_*")))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {base_dir}")
    
    return Path(run_dirs[-1])


def load_training_history(run_dir):
    """Load training history from JSON file"""
    history_path = Path(run_dir) / "training_history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found at {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history


def load_config(run_dir):
    """Load the configuration used for training"""
    config_path = Path(run_dir) / "config_used.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def main():
    parser = argparse.ArgumentParser(description="RF-DETR Training Analysis Tool")
    parser.add_argument('--run_dir', type=str, help="Path to specific run directory")
    parser.add_argument('--output_dir', type=str, default="output", 
                        help="Base output directory containing run folders")
    args = parser.parse_args()
    
    try:
        # Determine which run to analyze
        if args.run_dir:
            run_dir = Path(args.run_dir)
        else:
            run_dir = find_latest_run(args.output_dir)
        
        print(f"Analyzing training results in: {run_dir}")
        
        # Load training history and config
        history = load_training_history(run_dir)
        config = load_config(run_dir)
        
        # Analyze training results
        analyze_training_results(history, config, run_dir)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 