"""
Distributed training implementation for RF-DETR
"""
import os
import time
import torch
import torch.distributed as dist
from pathlib import Path
from rfdetr import RFDETRBase

from utils import (
    get_run_dir, setup_comet, get_metric_callback, 
    save_training_results, fix_autocast_dtype, 
    get_training_params, setup_distributed, broadcast_run_dir
)

# Import evaluation module
from evaluate_model import run_evaluation

def train_distributed(rank, world_size, config, gpu_ids):
    """Training function for distributed training"""
    # Get the actual GPU ID from the list of selected GPUs
    gpu_id = gpu_ids[rank]
    
    # Setup distributed process group
    setup_distributed(rank, world_size, config)
    
    # Fix autocast to use float16 instead of bfloat16
    fix_autocast_dtype()
    
    # Set device for this process - use the specific GPU ID
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)
    
    # Only print from the main process
    is_main_process = rank == 0
    
    # Create a unique run directory (only on main process)
    run_dir = None
    if is_main_process:
        base_output_dir = Path(config["logging"]["output_dir"])
        base_output_dir.mkdir(exist_ok=True)
        run_dir = get_run_dir(base_output_dir)
    
    # Broadcast run_dir from rank 0 to all processes
    run_dir = broadcast_run_dir(run_dir, is_main_process, rank, device)
    
    if is_main_process:
        print(f"\n{'='*50}")
        print(f"Distributed Training: {world_size} GPUs")
        print(f"Process rank: {rank}")
        print(f"Using GPU ID: {gpu_id}")
        print(f"Device: {device}")
        print(f"Run directory: {run_dir}")
        print(f"{'='*50}\n")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Initialize COMET logging (only on main process)
    experiment = setup_comet(config, is_main_process, run_dir)
    
    if is_main_process:
        print(f"Using dataset from: {os.path.abspath(config['data']['dataset_dir'])}")
        print(f"{'='*50}\n")
        print("Initializing RF-DETR model...")
    
    # Initialize model
    model = RFDETRBase()
    
    # Setup training history and callback
    history = []
    callback = get_metric_callback(history, experiment if is_main_process else None)
    model.callbacks["on_fit_epoch_end"] = [callback]
    
    if is_main_process:
        print("\nStarting training...")
        print(f"Training parameters:")
        for key, value in config["training"].items():
            print(f"- {key}: {value}")
        print(f"{'='*50}\n")
    
    # Create weights directory (only on main process)
    weights_dir = run_dir / "weights"
    if is_main_process:
        weights_dir.mkdir(exist_ok=True)
    
    # Get training parameters
    training_params = get_training_params(
        config, run_dir, 
        distributed=True, 
        rank=rank, 
        world_size=world_size,
        device=device, 
        device_id=gpu_id
    )
    
    # Training
    start_time = time.time()
    model.train(**training_params)
    end_time = time.time()
    
    # Save results (only on main process)
    if is_main_process:
        save_training_results(run_dir, history, config, model, end_time, start_time, experiment)
        
        # Run evaluation on the trained model (only on main process)
        print("\n\nRunning evaluation to generate confusion matrix...\n")
        
        # Get the class names from the config
        num_classes = config["model"]["num_classes"]
        
        # Use the dataset directory
        dataset_dir = config["data"]["dataset_dir"]
        test_dir = os.path.join(dataset_dir, "test")
        
        # Save the model explicitly to make sure we have the latest state
        eval_model_path = run_dir / "checkpoint_for_eval.pth"
        print(f"Saving model state for evaluation to: {eval_model_path}")
        
        # Save the model - handle the case where model doesn't have state_dict
        try:
            # First try standard PyTorch way
            if hasattr(model, 'state_dict'):
                torch.save(model.state_dict(), eval_model_path)
            # Then try saving via model's save method if available
            elif hasattr(model, 'save_model'):
                model.save_model(eval_model_path)
            # Otherwise save the whole model object
            else:
                torch.save(model, eval_model_path)
            print("Model saved, running evaluation...")
        except Exception as e:
            print(f"Error saving model: {e}")
            print("Continuing without evaluation results")
            return  # Skip evaluation if we can't save the model
        
        # Use subprocess to run the standalone evaluation script
        # This ensures we use the exact same evaluation method as the standalone script
        import subprocess
        
        # Construct the command to run the evaluation script
        cmd = [
            "python", "evaluate_model.py",
            "--run_dir", str(run_dir),
            "--test_dir", test_dir,
            "--model_path", str(eval_model_path),
            "--class_names", "player", "ball", "logo",
            "--debug"
        ]
        
        print(f"Running evaluation command: {' '.join(cmd)}")
        
        # Run the evaluation script
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            
            # Log results to Comet if available
            if experiment is not None:
                # Log confusion matrix and metrics to Comet
                confusion_matrix_path = run_dir / "confusion_matrix.png"
                metrics_path = str(confusion_matrix_path).replace('.png', '_metrics.png')
                
                if confusion_matrix_path.exists():
                    experiment.log_image(confusion_matrix_path, name="Confusion Matrix")
                
                if Path(metrics_path).exists():
                    experiment.log_image(metrics_path, name="Evaluation Metrics")
                
                # Extract mAP if possible
                import re
                mAP_match = re.search(r"mAP: (\d+\.\d+)", result.stdout)
                if mAP_match:
                    mAP = float(mAP_match.group(1))
                    experiment.log_metric("test_mAP", mAP)
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation script: {e}")
            print(f"Error output: {e.stderr}")
            print("Continuing without evaluation results")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Clean up distributed group
    dist.destroy_process_group() 