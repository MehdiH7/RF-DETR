"""
Directory and path utilities for RF-DETR training
"""
import glob
from pathlib import Path
from datetime import datetime

def get_run_dir(base_output_dir):
    """Create a unique run directory with incrementing run number"""
    # Create timestamp string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get the base output directory
    base_dir = Path(base_output_dir)
    
    # Find all existing run directories matching the pattern run_*
    existing_runs = glob.glob(str(base_dir / "run_*"))
    
    # Extract run numbers from directory names
    run_numbers = []
    for run_dir in existing_runs:
        try:
            run_num = int(Path(run_dir).name.split("_")[1])
            run_numbers.append(run_num)
        except (ValueError, IndexError):
            continue
    
    # Calculate the next run number
    next_run = 1
    if run_numbers:
        next_run = max(run_numbers) + 1
    
    # Create the new run directory name
    run_dir = base_dir / f"run_{next_run:03d}_{timestamp}"
    
    # Create the directory
    run_dir.mkdir(parents=True, exist_ok=True)
    
    return run_dir 