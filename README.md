# RF-DETR Training

This repository contains a modular implementation of training code for the RF-DETR model.

## Project Structure

The code has been organized into the following structure:

```
RF-DETR/
├── config.json              # Main configuration file
├── train.py                 # Main training script (entry point)
├── analyze_results.py       # Standalone script for analyzing training results
├── evaluate_model.py        # Script for evaluating model on test data
├── utils/                   # Utility modules
│   ├── __init__.py          # Re-exports key utility functions
│   ├── analysis.py          # Training result analysis and plotting utilities
│   ├── config.py            # Configuration loading utilities
│   ├── directories.py       # Directory management utilities
│   ├── distributed.py       # Distributed training utilities
│   ├── logging.py           # Logging and experiment tracking utilities
│   └── training.py          # Training parameter utilities
└── trainers/                # Training implementation modules
    ├── __init__.py          # Exports training functions
    ├── distributed.py       # Distributed training implementation
    └── single_gpu.py        # Single GPU training implementation
```

## Usage

### Training

To train the model, simply run:

```bash
./train.py
```

The training script will automatically:

1. Load the configuration from `config.json`
2. Determine if distributed training should be used 
3. Create a unique run directory for outputs
4. Set up logging with Comet ML (if configured)
5. Train the model and save checkpoints
6. Generate analysis plots and reports at the end of training

### Evaluation

To evaluate a trained model on the test set and generate a confusion matrix:

```bash
# Evaluate the most recent training run
./evaluate_model.py

# Evaluate a specific run
./evaluate_model.py --run_dir output/run_001_20240327_123456

# Set a different confidence threshold
./evaluate_model.py --confidence 0.7

# Explicitly specify class names (overrides automatic detection)
./evaluate_model.py --class_names player ball logo
```

The evaluation script:
1. Loads the trained model
2. Runs inference on the test dataset
3. Calculates prediction metrics
4. Generates a confusion matrix
5. Updates the training analysis with test results

#### Automatic Class Name Detection

The evaluation script automatically detects class names from the dataset in the following order:
1. Looks for `classes.txt` or `classes.json` files in the dataset directory
2. Searches for dataset information files with class name definitions
3. Checks for COCO format annotation files with category definitions
4. Examines YOLO format data.yaml files
5. Analyzes test set label files to determine class indices
6. Falls back to default names if nothing is found (for 3 classes: "player", "ball", "logo")

You can override the automatic detection by specifying class names directly with the `--class_names` option.

### Analysis

To analyze training and evaluation results:

```bash
# Analyze the most recent training run
./analyze_results.py

# Analyze a specific run
./analyze_results.py --run_dir output/run_001_20240327_123456

# Analyze a run from a different output directory
./analyze_results.py --output_dir path/to/output
```

## Configuration

Training parameters are set in the `config.json` file. Key sections include:

- `training`: Parameters for the training process
- `model`: Model architecture and parameters
- `loss`: Loss function configuration
- `data`: Dataset configuration
- `distributed`: Distributed training settings
- `logging`: Logging and output settings

## Output Structure

Each training run creates a unique directory with the following structure:

```
output/
├── run_001_20240327_123456/
│   ├── config_used.json           # Configuration used for this run
│   ├── training_history.json      # Training metrics history
│   ├── test_predictions.json      # Test set predictions (after evaluation)
│   ├── class_names.json           # Class names used for confusion matrix (if explicitly set)
│   ├── weights/                   # Model weights
│   │   ├── checkpoint_*.pth       # (if checkpointing is enabled)
│   │   └── final_model.pth        # Final trained model
│   └── plots/                     # Analysis plots
│       ├── loss_curves.png        # Training/validation loss plots
│       ├── accuracy_metrics.png   # Accuracy metrics plots
│       ├── bbox_loss_curves.png   # Bounding box loss plots
│       ├── class_loss_curves.png  # Classification loss plots
│       ├── learning_rate.png      # Learning rate plots
│       ├── confusion_matrix.png   # Normalized confusion matrix (after evaluation)
│       ├── confusion_matrix_raw.png # Raw counts confusion matrix
│       └── training_summary.md    # Training summary report
└── ...
```

## Analyzing Results

After training and evaluation, the analysis tools generate several visualizations and reports:

1. **Loss Curves**: Training and validation loss over epochs
2. **Accuracy Metrics**: Plots of any accuracy-related metrics
3. **Component Losses**: Specific loss component breakdowns (class, bounding box)
4. **Learning Rate**: Learning rate progression during training
5. **Confusion Matrix**: When test predictions are available, two confusion matrices are generated:
   - Normalized matrix (shows per-class accuracy)
   - Raw counts matrix (shows actual counts)
6. **Summary Report**: A markdown report with:
   - Training configuration
   - Final metrics
   - Best metrics during training
   - Per-class precision, recall, and F1 scores (after evaluation) 