"""
Distributed training utilities for RF-DETR
"""
import os
import torch
import torch.distributed as dist
from pathlib import Path

def setup_distributed(rank, world_size, config):
    """Setup distributed training environment"""
    os.environ['MASTER_ADDR'] = config['distributed']['master_addr']
    os.environ['MASTER_PORT'] = config['distributed']['master_port']
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def broadcast_run_dir(run_dir, is_main_process, rank, device):
    """Broadcast run directory path from main process to all other processes"""
    run_dir_str = str(run_dir) if run_dir else ""
    
    # Broadcast run_dir from rank 0 to all processes
    if torch.cuda.is_available():
        run_dir_str_tensor = torch.tensor([ord(c) for c in run_dir_str] + [0] * (1024 - len(run_dir_str)), 
                                         dtype=torch.long, device=device)
        dist.broadcast(run_dir_str_tensor, 0)
        
        # Convert back to string on other processes
        if not is_main_process:
            chars = []
            for i in run_dir_str_tensor.cpu().tolist():
                if i == 0:
                    break
                chars.append(chr(i))
            run_dir_str = "".join(chars)
            run_dir = Path(run_dir_str)
    
    return run_dir 