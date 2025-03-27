import os
import json
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from rfdetr import RFDETRBase
from comet_ml import Experiment
from pathlib import Path
from datetime import datetime
import glob

def load_config(config_path="config.json"):
	with open(config_path, 'r') as f:
		return json.load(f)

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

def setup_distributed(rank, world_size, config):
	"""Setup distributed training environment"""
	os.environ['MASTER_ADDR'] = config['distributed']['master_addr']
	os.environ['MASTER_PORT'] = config['distributed']['master_port']
	dist.init_process_group("nccl", rank=rank, world_size=world_size)

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
	
	# End COMET experiment
	if experiment is not None:
		experiment.end()

def fix_autocast_dtype():
	"""Patch torch.cuda.amp.autocast to use float16 instead of bfloat16"""
	if torch.cuda.is_available():
		orig_init = torch.cuda.amp.autocast.__init__
		def patched_init(self, enabled=True, dtype=torch.float16, cache_enabled=None):
			orig_init(self, enabled=enabled, dtype=torch.float16, cache_enabled=cache_enabled)
		torch.cuda.amp.autocast.__init__ = patched_init

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
		run_dir_str = str(run_dir)
	else:
		run_dir_str = ""
	
	# Broadcast run_dir from rank 0 to all processes
	if torch.cuda.is_available():
		run_dir_str_tensor = torch.tensor([ord(c) for c in run_dir_str] + [0] * (1024 - len(run_dir_str)), dtype=torch.long, device=device)
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
	
	# Clear GPU cache
	torch.cuda.empty_cache()
	
	# Clean up distributed group
	dist.destroy_process_group()

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

def train():
	"""Main training entry point"""
	# Load configuration
	config = load_config()
	
	# Check if distributed training is enabled
	use_distributed = config.get("distributed", {}).get("use_distributed", False)
	
	if use_distributed and torch.cuda.is_available():
		# Get the list of GPUs to use
		gpu_ids = config.get("distributed", {}).get("gpu_ids", list(range(torch.cuda.device_count())))
		
		# Limit to available GPUs
		gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id < torch.cuda.device_count()]
		
		world_size = len(gpu_ids)
		
		if world_size > 1:
			print(f"Found {torch.cuda.device_count()} GPUs. Using {world_size} GPUs: {gpu_ids}")
			
			# Set memory allocation parameters to avoid OOM
			os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
			
			# Spawn multiple processes for distributed training
			mp.spawn(
				train_distributed,
				args=(world_size, config, gpu_ids),
				nprocs=world_size,
				join=True
			)
			return
		else:
			print(f"Found {torch.cuda.device_count()} GPUs, but only {world_size} specified in config. Using single GPU mode.")
	
	# Single GPU or CPU training
	train_single_gpu(config)

if __name__ == '__main__':
	# Set multiprocessing start method
	mp.set_start_method('spawn', force=True)
	train()