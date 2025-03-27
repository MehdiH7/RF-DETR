"""
Logging and experiment tracking utilities for RF-DETR training
"""
import json
from pathlib import Path
from comet_ml import Experiment
import torch
from .analysis import analyze_training_results

def setup_comet(config, is_main_process=True, run_dir=None):
    """Initialize COMET experiment"""
    if not is_main_process:
        return None
        
    try:
        # Initialize the experiment without experiment_name
        experiment = Experiment(
            project_name=config["logging"]["comet"]["project_name"],
            workspace=config["logging"]["comet"]["workspace"],
            api_key=config["logging"]["comet"]["api_key"]
        )
        # Log hyperparameters
        experiment.log_parameters(config)
        if run_dir:
            experiment.log_parameter("run_directory", str(run_dir))
            # Set experiment name after creation
            try:
                run_num = Path(run_dir).name.split("_")[1]
                experiment.set_name(f"Run_{run_num}")
            except (IndexError, AttributeError, Exception):
                pass
        return experiment
    except Exception as e:
        print(f"Warning: Could not initialize Comet ML: {e}")
        print("Training will continue without Comet logging")
        return None

def get_metric_callback(history, experiment=None):
    """Create a callback function for logging training metrics"""
    def callback(data):
        history.append(data)
        # Print and log metrics
        print(f"\nEpoch {len(history)} Metrics:")
        for key, value in data.items():
            # Check type and format accordingly
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
                if experiment is not None:
                    experiment.log_metric(key, value, step=len(history))
            elif isinstance(value, list):
                print(f"{key}: {value}")
                if experiment is not None and len(value) > 0 and isinstance(value[0], (int, float)):
                    # If it's a list of numbers, log the mean
                    experiment.log_metric(key + "_mean", sum(value)/len(value), step=len(history))
            else:
                print(f"{key}: {value}")
    
    return callback

def save_training_results(run_dir, history, config, model, end_time, start_time, experiment=None):
    """Save training results, model, and config"""
    print("\nSaving training history...")
    print(f"Saving results to: {run_dir}")
    
    # Save training history
    with open(run_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=4)
    
    # Save config used for this run
    with open(run_dir / "config_used.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Save final model
    try:
        model_path = run_dir / "weights" / "final_model.pth"
        print(f"Saving final model to: {model_path}")
        if hasattr(model, 'module'):
            torch.save(model.module.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
    except Exception as e:
        print(f"Warning: Could not save final model: {e}")
    
    print(f"\n{'='*50}")
    print(f"Training completed in {end_time - start_time:.2f} seconds!")
    print(f"Training history saved to: {run_dir / 'training_history.json'}")
    print(f"{'='*50}\n")
    
    # Generate analysis plots
    analyze_training_results(history, config, run_dir)
    
    # End COMET experiment
    if experiment is not None:
        experiment.end() 