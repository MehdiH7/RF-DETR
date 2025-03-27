"""
Training utilities for RF-DETR
"""
import torch
import os

def fix_autocast_dtype():
    """Patch torch.cuda.amp.autocast to use float16 instead of bfloat16"""
    if torch.cuda.is_available():
        orig_init = torch.cuda.amp.autocast.__init__
        def patched_init(self, enabled=True, dtype=torch.float16, cache_enabled=None):
            orig_init(self, enabled=enabled, dtype=torch.float16, cache_enabled=cache_enabled)
        torch.cuda.amp.autocast.__init__ = patched_init

def get_training_params(config, run_dir, distributed=False, rank=0, world_size=1, device=None, device_id=None):
    """Create training parameters dictionary from config sections"""
    # Combine all config sections into one dictionary
    training_params = {}
    
    # Add all sections from config
    for section in ["training", "model", "loss", "data", "device"]:
        if section in config:
            training_params.update(config[section])
    
    # Set run-specific model output directory for saving weights
    training_params['output_dir'] = str(run_dir)
    training_params['save_dir'] = str(run_dir / "weights")
    
    # Add distributed parameters if applicable
    if distributed:
        training_params.update({
            'distributed': True,
            'rank': rank,
            'world_size': world_size,
            'device': device,
            'device_id': device_id
        })
        
        # Add sync_bn if it exists in the config
        if "sync_bn" in config.get("distributed", {}):
            training_params["sync_bn"] = config["distributed"]["sync_bn"]
    else:
        training_params.update({
            'distributed': False,
            'device': device,
            'device_id': device_id if device_id is not None else 0 if torch.cuda.is_available() else None
        })
    
    # Rename learning_rate to lr as expected by the model
    if 'learning_rate' in training_params and 'lr' not in training_params:
        training_params['lr'] = training_params.pop('learning_rate')
        
    return training_params 