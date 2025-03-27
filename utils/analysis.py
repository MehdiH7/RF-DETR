"""
Analysis utilities for RF-DETR training results
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

def create_dataframe(history):
    """Convert training history to a pandas DataFrame with epoch index"""
    df = pd.DataFrame(history)
    
    # Add epoch column if not present
    if 'epoch' not in df.columns:
        df['epoch'] = range(1, len(df) + 1)
    
    return df

def setup_plot_dir(run_dir):
    """Create plots directory if it doesn't exist"""
    plots_dir = Path(run_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    return plots_dir

def plot_loss_curves(df, plots_dir):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 8))
    
    # Training loss
    if 'train_loss' in df.columns:
        plt.plot(
            df['epoch'],
            df['train_loss'],
            label='Training Loss',
            marker='o',
            linestyle='-',
            color='blue'
        )
    
    # Validation loss
    if 'test_loss' in df.columns:
        plt.plot(
            df['epoch'],
            df['test_loss'],
            label='Validation Loss',
            marker='o',
            linestyle='--',
            color='red'
        )
    
    plt.title('Train/Validation Loss over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(plots_dir / "loss_curves.png", dpi=300)
    plt.savefig(plots_dir / "loss_curves.pdf")
    plt.close()

def plot_class_loss_curves(df, plots_dir):
    """Plot class loss curves if available"""
    class_loss_columns = [col for col in df.columns if 'class_loss' in col or 'cls_loss' in col]
    if class_loss_columns:
        plt.figure(figsize=(12, 8))
        
        for col in class_loss_columns:
            plt.plot(
                df['epoch'],
                df[col],
                label=col.replace('_', ' ').title(),
                marker='o'
            )
        
        plt.title('Classification Loss Components over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(plots_dir / "class_loss_curves.png", dpi=300)
        plt.savefig(plots_dir / "class_loss_curves.pdf")
        plt.close()

def plot_bbox_loss_curves(df, plots_dir):
    """Plot bounding box loss curves if available"""
    bbox_loss_columns = [col for col in df.columns if 'bbox_loss' in col or 'giou_loss' in col]
    if bbox_loss_columns:
        plt.figure(figsize=(12, 8))
        
        for col in bbox_loss_columns:
            plt.plot(
                df['epoch'],
                df[col],
                label=col.replace('_', ' ').title(),
                marker='o'
            )
        
        plt.title('Bounding Box Loss Components over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(plots_dir / "bbox_loss_curves.png", dpi=300)
        plt.savefig(plots_dir / "bbox_loss_curves.pdf")
        plt.close()

def plot_learning_rate(df, plots_dir):
    """Plot learning rate over epochs if available"""
    lr_columns = [col for col in df.columns if 'lr' in col.lower() or 'learning_rate' in col.lower()]
    if lr_columns:
        plt.figure(figsize=(12, 8))
        
        for col in lr_columns:
            plt.plot(
                df['epoch'],
                df[col],
                label=col.replace('_', ' ').title(),
                marker='o'
            )
        
        plt.title('Learning Rate over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Learning Rate', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.yscale('log')  # Usually better to view LR on log scale
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(plots_dir / "learning_rate.png", dpi=300)
        plt.savefig(plots_dir / "learning_rate.pdf")
        plt.close()

def plot_accuracy_metrics(df, plots_dir):
    """Plot accuracy metrics if available"""
    acc_columns = [col for col in df.columns if 
                  'acc' in col.lower() or 
                  'accuracy' in col.lower() or 
                  'ap' in col.lower() or 
                  'map' in col.lower()]
    
    if acc_columns:
        plt.figure(figsize=(12, 8))
        
        for col in acc_columns:
            plt.plot(
                df['epoch'],
                df[col],
                label=col.replace('_', ' ').title(),
                marker='o'
            )
        
        plt.title('Accuracy Metrics over Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(plots_dir / "accuracy_metrics.png", dpi=300)
        plt.savefig(plots_dir / "accuracy_metrics.pdf")
        plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, plots_dir):
    """Plot confusion matrix for model predictions
    
    Args:
        y_true: List of true class indices
        y_pred: List of predicted class indices
        class_names: List of class names corresponding to indices
        plots_dir: Directory to save the plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot with seaborn for better styling
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Normalized Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=300)
    plt.savefig(plots_dir / "confusion_matrix.pdf")
    
    # Also save a non-normalized version
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix (Raw Counts)', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(plots_dir / "confusion_matrix_raw.png", dpi=300)
    plt.savefig(plots_dir / "confusion_matrix_raw.pdf")
    plt.close()
    
    return cm, cm_norm

def create_test_predictions_file(run_dir):
    """Creates a template for test set predictions that can be filled with actual data
    
    This function creates a JSON template file that can be used to store test set
    predictions for generating the confusion matrix.
    """
    predictions_path = run_dir / "test_predictions.json"
    
    if not predictions_path.exists():
        template = {
            "class_names": ["class1", "class2", "class3"],  # Replace with actual class names
            "true_labels": [],  # List of true class indices
            "pred_labels": []   # List of predicted class indices
        }
        
        with open(predictions_path, 'w') as f:
            json.dump(template, f, indent=4)
        
        print(f"Created template for test predictions at: {predictions_path}")
        print("Please fill this file with actual test predictions to generate a confusion matrix.")
    
    return predictions_path

def load_test_predictions(run_dir):
    """Load test set predictions from JSON file if it exists"""
    predictions_path = Path(run_dir) / "test_predictions.json"
    
    if predictions_path.exists():
        try:
            with open(predictions_path, 'r') as f:
                predictions = json.load(f)
            
            # Validate the data format
            required_keys = ["class_names", "true_labels", "pred_labels"]
            for key in required_keys:
                if key not in predictions:
                    raise KeyError(f"Missing required key: {key} in predictions file")
            
            if len(predictions["true_labels"]) != len(predictions["pred_labels"]):
                raise ValueError("Number of true labels and predicted labels do not match")
            
            if len(predictions["true_labels"]) == 0:
                raise ValueError("No prediction data found")
                
            return predictions
        except Exception as e:
            print(f"Error loading test predictions: {e}")
            return None
    else:
        # Create a template file for future use
        create_test_predictions_file(run_dir)
        return None

def create_summary_report(df, config, run_dir, plots_dir, confusion_matrix=None, class_names=None):
    """Create a summary report with key metrics"""
    # Get the last row for final metrics
    final_metrics = df.iloc[-1].to_dict()
    
    # Create summary text
    summary = ["# RF-DETR Training Summary\n"]
    summary.append(f"## Run Directory: {run_dir.name}\n")
    
    # Basic training config
    summary.append("## Training Configuration\n")
    summary.append(f"- Epochs: {config['training'].get('epochs', 'N/A')}")
    summary.append(f"- Batch Size: {config['training'].get('batch_size', 'N/A')}")
    summary.append(f"- Learning Rate: {config['training'].get('learning_rate', 'N/A')}")
    summary.append(f"- Dataset: {config['data'].get('dataset_file', 'N/A')}")
    summary.append(f"- Model: {config['model'].get('encoder', 'RF-DETR')}")
    
    # Final metrics
    summary.append("\n## Final Metrics\n")
    for key, value in final_metrics.items():
        if isinstance(value, (int, float)):
            summary.append(f"- {key}: {value:.4f}")
    
    # Best results
    summary.append("\n## Best Results\n")
    
    # Find minimum loss
    if 'train_loss' in df.columns:
        min_loss_idx = df['train_loss'].idxmin()
        min_loss = df.loc[min_loss_idx, 'train_loss']
        min_loss_epoch = df.loc[min_loss_idx, 'epoch']
        summary.append(f"- Best Training Loss: {min_loss:.4f} (Epoch {min_loss_epoch})")
    
    if 'test_loss' in df.columns:
        min_test_loss_idx = df['test_loss'].idxmin()
        min_test_loss = df.loc[min_test_loss_idx, 'test_loss']
        min_test_loss_epoch = df.loc[min_test_loss_idx, 'epoch']
        summary.append(f"- Best Validation Loss: {min_test_loss:.4f} (Epoch {min_test_loss_epoch})")
    
    # Find best accuracy metrics
    for col in df.columns:
        if ('acc' in col.lower() or 'ap' in col.lower() or 'map' in col.lower()) and col != 'epoch':
            max_idx = df[col].idxmax()
            max_val = df.loc[max_idx, col]
            max_epoch = df.loc[max_idx, 'epoch']
            summary.append(f"- Best {col}: {max_val:.4f} (Epoch {max_epoch})")
            
    # Add confusion matrix stats if available
    if confusion_matrix is not None and class_names is not None:
        summary.append("\n## Confusion Matrix Analysis\n")
        
        # Calculate per-class metrics
        class_precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
        class_recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        
        for i, class_name in enumerate(class_names):
            summary.append(f"### Class: {class_name}\n")
            summary.append(f"- Precision: {class_precision[i]:.4f}")
            summary.append(f"- Recall: {class_recall[i]:.4f}")
            if class_precision[i] + class_recall[i] > 0:
                f1 = 2 * (class_precision[i] * class_recall[i]) / (class_precision[i] + class_recall[i])
                summary.append(f"- F1 Score: {f1:.4f}")
            summary.append("")
            
        # Overall accuracy
        accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
        summary.append(f"### Overall Accuracy: {accuracy:.4f}\n")
    
    # Write summary to file
    with open(plots_dir / "training_summary.md", "w") as f:
        f.write("\n".join(summary))
    
    # Also create a plain text version
    with open(plots_dir / "training_summary.txt", "w") as f:
        f.write("\n".join(summary))

def analyze_training_results(history, config, run_dir):
    """Generate analysis plots and report for a training run"""
    try:
        print("\nGenerating training analysis plots...")
        # Convert to DataFrame for easier analysis
        df = create_dataframe(history)
        
        # Setup plots directory
        plots_dir = setup_plot_dir(run_dir)
        print(f"Saving plots to: {plots_dir}")
        
        # Generate plots
        plot_loss_curves(df, plots_dir)
        plot_class_loss_curves(df, plots_dir)
        plot_bbox_loss_curves(df, plots_dir)
        plot_learning_rate(df, plots_dir)
        plot_accuracy_metrics(df, plots_dir)
        
        # Check for test predictions and create confusion matrix if available
        predictions = load_test_predictions(run_dir)
        confusion_mat = None
        class_names = None
        
        if predictions:
            print("Generating confusion matrix from test predictions...")
            y_true = predictions["true_labels"]
            y_pred = predictions["pred_labels"]
            class_names = predictions["class_names"]
            
            # Plot confusion matrix
            confusion_mat, _ = plot_confusion_matrix(y_true, y_pred, class_names, plots_dir)
        
        # Create summary report
        create_summary_report(df, config, run_dir, plots_dir, confusion_mat, class_names)
        
        print(f"Analysis complete. Results saved to {plots_dir}")
        return True
    except Exception as e:
        print(f"Warning: Could not generate training analysis plots: {e}")
        import traceback
        traceback.print_exc()
        return False 