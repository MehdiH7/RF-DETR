import os
import json
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rfdetr import RFDETRBase
from tqdm import tqdm
from comet_ml import Experiment
from pathlib import Path

def load_config(config_path="config.json"):
	with open(config_path, 'r') as f:
		return json.load(f)

def setup_comet(config):
	"""Initialize COMET experiment"""
	experiment = Experiment(
		project_name=config["logging"]["comet"]["project_name"],
		workspace=config["logging"]["comet"]["workspace"]
	)
	# Log hyperparameters
	experiment.log_parameters(config)
	return experiment

def setup_device(gpu_id, config):
	"""Setup device and distributed training if available"""
	if torch.cuda.is_available():
		torch.cuda.set_device(gpu_id)
		device = torch.device(f'cuda:{gpu_id}')
		# Initialize distributed process group
		dist.init_process_group(
			backend='nccl',
			init_method='env://',
			world_size=torch.cuda.device_count(),
			rank=gpu_id
		)
	else:
		device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
		print(f"CUDA not available. Using device: {device}")
	
	return device

def train(gpu_id, config):
	"""Main training function"""
	# Setup device
	device = setup_device(gpu_id, config)
	is_main_process = gpu_id == 0 if torch.cuda.is_available() else True

	if is_main_process:
		# Initialize COMET logging
		experiment = setup_comet(config)
		print(f"\n{'='*50}")
		print(f"Using dataset from: {os.path.abspath(config['data']['dataset_dir'])}")
		print(f"Device: {device}")
		if torch.cuda.is_available():
			print(f"GPU Count: {torch.cuda.device_count()}")
			print(f"Current GPU: {gpu_id}")
		print(f"{'='*50}\n")

	# Initialize model
	if is_main_process:
		print("Initializing RF-DETR model...")
	model = RFDETRBase()
	model.to(device)

	# Setup distributed training if using CUDA
	if torch.cuda.is_available():
		model = torch.nn.parallel.DistributedDataParallel(
			model, device_ids=[gpu_id]
		)

	history = []

	def callback2(data):
		history.append(data)
		if is_main_process:
			# Print and log metrics
			print(f"\nEpoch {len(history)} Metrics:")
			for key, value in data.items():
				print(f"{key}: {value:.4f}")
				experiment.log_metric(key, value, step=len(history))

	# Add callback to track training history
	model.callbacks["on_fit_epoch_end"].append(callback2)

	if is_main_process:
		print("\nStarting training...")
		print(f"Training parameters:")
		for key, value in config["training"].items():
			print(f"- {key}: {value}")
		print(f"{'='*50}\n")

	# Training
	start_time = time.time()
	model.train(
		dataset_dir=config["data"]["dataset_dir"],
		epochs=config["training"]["epochs"],
		batch_size=config["training"]["batch_size"],
		lr=config["training"]["learning_rate"]
	)
	end_time = time.time()

	if is_main_process:
		# Save training history
		print("\nSaving training history...")
		output_dir = Path(config["logging"]["output_dir"])
		output_dir.mkdir(exist_ok=True)
		
		with open(output_dir / "training_history.json", "w") as f:
			json.dump(history, f, indent=4)

		print(f"\n{'='*50}")
		print(f"Training completed in {end_time - start_time:.2f} seconds!")
		print(f"Training history saved to: {output_dir / 'training_history.json'}")
		print(f"{'='*50}\n")

		# End COMET experiment
		experiment.end()

def main():
	# Load configuration
	config = load_config()
	
	if torch.cuda.is_available():
		# Use all available GPUs
		world_size = torch.cuda.device_count()
		mp.spawn(train, args=(config,), nprocs=world_size)
	else:
		# Use CPU or MPS
		train(0, config)

if __name__ == '__main__':
	main()