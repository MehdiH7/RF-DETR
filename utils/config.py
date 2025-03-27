"""
Configuration utilities for RF-DETR training
"""
import json
import os

# Default config path that can be overridden
CONFIG_PATH = "config.json"

def load_config(config_path=None):
    """Load configuration from JSON file"""
    # Use global CONFIG_PATH if no path is provided
    if config_path is None:
        config_path = CONFIG_PATH
    
    # Print loading information for debugging
    print(f"Loading config from: {config_path}")
    print(f"File exists: {os.path.exists(config_path)}")
    
    with open(config_path, 'r') as f:
        return json.load(f) 