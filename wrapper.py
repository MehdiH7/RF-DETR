# Import comet_ml first to avoid warning
import comet_ml
print("Comet ML imported successfully")

# Import other modules
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Import and run the training script
import train
print("Starting training...")
train.train()
