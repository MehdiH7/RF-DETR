"""
Single GPU training implementation for RF-DETR
"""
import os
import time
import torch
from pathlib import Path
from rfdetr import RFDETRBase

from utils import (
    get_run_dir, setup_comet, get_metric_callback, 
    save_training_results, fix_autocast_dtype, 
    get_training_params
)

# Import evaluation module
from evaluate_model import run_evaluation

def train_single_gpu(config):
    """Training function for single GPU or CPU"""
    # Fix autocast to use float16 instead of bfloat16
    fix_autocast_dtype()
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_id = 0
        print(f"Using single GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        device_id = None
        print(f"CUDA not available. Using device: {device}")

    # Create a unique run directory
    base_output_dir = Path(config["logging"]["output_dir"])
    base_output_dir.mkdir(exist_ok=True)
    run_dir = get_run_dir(base_output_dir)
    print(f"Run directory: {run_dir}")

    # Create weights directory
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    # Initialize COMET logging
    experiment = setup_comet(config, run_dir=run_dir)
    print(f"\n{'='*50}")
    print(f"Using dataset from: {os.path.abspath(config['data']['dataset_dir'])}")
    print(f"Device: {device}")
    print(f"{'='*50}\n")
    
    # Initialize model
    print("Initializing RF-DETR model...")
    model = RFDETRBase()
    
    # Setup training history and callback
    history = []
    callback = get_metric_callback(history, experiment)
    model.callbacks["on_fit_epoch_end"] = [callback]
    
    print("\nStarting training...")
    print(f"Training parameters:")
    for key, value in config["training"].items():
        print(f"- {key}: {value}")
    print(f"{'='*50}\n")
    
    # Get training parameters
    training_params = get_training_params(
        config, run_dir, 
        distributed=False, 
        device=device, 
        device_id=device_id
    )
    
    # Training
    start_time = time.time()
    model.train(**training_params)
    end_time = time.time()
    
    # Save results
    save_training_results(run_dir, history, config, model, end_time, start_time, experiment)
    
    # Run evaluation on the trained model
    print("\n\nRunning evaluation to generate confusion matrix...\n")
    
    # Get the class names from the config
    num_classes = config["model"]["num_classes"]
    
    # Use the dataset directory
    dataset_dir = config["data"]["dataset_dir"]
    test_dir = os.path.join(dataset_dir, "test")
    
    # Save the model explicitly to make sure we have the latest state
    eval_model_path = run_dir / "checkpoint_for_eval.pth"
    print(f"Saving model state for evaluation to: {eval_model_path}")
    
    # Save the model state dict directly
    torch.save(model.state_dict(), eval_model_path)
    print("Model saved, running evaluation...")
    
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