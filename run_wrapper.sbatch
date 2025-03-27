#!/bin/bash

#SBATCH --job-name=RF-DETR_wrapper
#SBATCH --output=RF-DETR_wrapper-%j.out
#SBATCH --error=RF-DETR_wrapper-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=hgx2q
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --gpu-bind=map_gpu:5
#SBATCH -c 20
#SBATCH --qos=normal

# Load modules
module purge
module load slurm/21.08.8

# Print environment info
echo "==================== ENVIRONMENT INFO ===================="
echo "Hostname: $(hostname)"
echo "Current directory: $(pwd)"
echo "Available GPUs: $(nvidia-smi -L)"
echo "Python version: $(python --version)"
echo "=========================================================="

# Set GPU
export CUDA_VISIBLE_DEVICES=5

# Set environment variables for distributed training
# Even though we're not using distributed training, the library expects these
export MASTER_ADDR="localhost"
export MASTER_PORT="12355"
export WORLD_SIZE="1"
export RANK="0"
export LOCAL_RANK="0"

# Print environment variables
echo "Setting environment variables:"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "LOCAL_RANK=$LOCAL_RANK"

# Activate conda environment
source /home/mehdihou/anaconda3/bin/activate trainenv

# Create a wrapper Python script to make sure comet_ml is imported first
cat > /home/mehdihou/D1/RF-DETR/wrapper.py << EOF
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
EOF

# Run training
echo "Starting single-GPU training with environment variables set..."
cd /home/mehdihou/D1/RF-DETR
python wrapper.py 