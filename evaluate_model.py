#!/usr/bin/env python
"""
RF-DETR Model Evaluation Tool

Generates a confusion matrix for the model using the test dataset.
Can be used as a standalone script or imported as a module.
"""

import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image
import importlib.util

# Import RF-DETR when running as standalone
try:
    from rfdetr import RFDETRBase
except ImportError:
    print("Warning: rfdetr module not found. This is fine if being imported by the training script.")

def find_latest_run(base_dir="output"):
    """Find the most recent training run directory"""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory {base_dir} not found")
    
    run_dirs = sorted(list(base_dir.glob("run_*")))
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {base_dir}")
    
    return run_dirs[-1]

def to_numpy(tensor_or_array):
    """Convert PyTorch tensor to NumPy array if needed"""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    return tensor_or_array

def evaluate_model(model, test_dir, class_names, confidence_threshold=0.5):
    """Evaluate model on test dataset images and generate confusion matrix"""
    print("Evaluating model on test dataset...")
    
    # Try to find if the RF-DETR has a dataset module
    try:
        if importlib.util.find_spec("rfdetr.dataset") is not None:
            from rfdetr.dataset import Dataset
            print("Found rfdetr.dataset module, using it")
            try:
                test_dataset = Dataset(test_dir, split="test")
                print(f"Created test dataset with {len(test_dataset)} samples")
                return process_with_dataset(model, test_dataset, class_names, confidence_threshold)
            except Exception as e:
                print(f"Error using rfdetr.dataset: {e}")
    except:
        print("No rfdetr.dataset module found, using image files")
    
    # Find all test images
    test_dir = Path(test_dir)
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(list(test_dir.glob(f"**/{ext}")))
    
    print(f"Found {len(image_paths)} test images")
    
    # Try to find a COCO annotation file
    annotation_files = list(test_dir.glob("**/annotations*.json"))
    if not annotation_files:
        annotation_files = list(test_dir.glob("**/*anno*.json"))
    
    using_coco_annotations = False
    coco_annotations = None
    
    if annotation_files:
        print(f"Found annotation file: {annotation_files[0]}")
        try:
            from pycocotools.coco import COCO
            coco_annotations = COCO(str(annotation_files[0]))
            using_coco_annotations = True
            print("Successfully loaded COCO format annotations")
            
            # Map category IDs to indices
            cats = coco_annotations.loadCats(coco_annotations.getCatIds())
            cat_id_to_idx = {cat['id']: i for i, cat in enumerate(cats)}
        except Exception as e:
            print(f"Not a COCO format annotation file: {e}")
    
    true_labels = []
    pred_labels = []
    
    # Create a mapping from large class IDs to our expected range
    class_id_mapping = {}
    
    # Process each image
    for img_path in tqdm(image_paths):
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            
            # Get predictions from model
            detections = model_predict(model, image, confidence_threshold)
            
            # Get ground truth annotations
            if using_coco_annotations:
                # Use COCO tools to find annotations for this image
                img_filename = img_path.name
                
                # Search for this image in COCO annotations
                img_id = None
                for id, img_info in coco_annotations.imgs.items():
                    if img_info['file_name'] == img_filename:
                        img_id = id
                        break
                
                if img_id is not None:
                    # Get annotations for this image
                    ann_ids = coco_annotations.getAnnIds(imgIds=img_id)
                    anns = coco_annotations.loadAnns(ann_ids)
                    
                    # Extract class IDs
                    for ann in anns:
                        cat_id = ann['category_id']
                        idx = cat_id_to_idx.get(cat_id, cat_id - 1)
                        true_labels.append(idx)
            else:
                # Try to find per-image annotation files
                img_name = img_path.stem
                ann_path = list(test_dir.glob(f"**/{img_name}.*json"))
                
                if ann_path:
                    with open(ann_path[0], 'r') as f:
                        ann_data = json.load(f)
                        
                    if 'annotations' in ann_data:
                        # COCO format
                        for ann in ann_data['annotations']:
                            true_labels.append(ann['category_id'] - 1)
                    else:
                        # Custom format
                        if 'objects' in ann_data:
                            for obj in ann_data['objects']:
                                if 'category_id' in obj:
                                    true_labels.append(obj['category_id'] - 1)
                                elif 'class_id' in obj:
                                    true_labels.append(obj['class_id'])
            
            # Add predictions to our lists
            if hasattr(detections, 'class_id'):
                for pred_class in detections.class_id:
                    # Apply mapping if needed
                    if isinstance(pred_class, torch.Tensor):
                        pred_class_id = pred_class.item()
                    elif isinstance(pred_class, np.ndarray):
                        pred_class_id = pred_class.item()
                    else:
                        pred_class_id = int(pred_class)
                    
                    if pred_class_id in class_id_mapping:
                        pred_class_id = class_id_mapping[pred_class_id]
                    # Force class_id to be within our expected range
                    if pred_class_id >= len(class_names):
                        pred_class_id = pred_class_id % len(class_names)
                    pred_labels.append(pred_class_id)
            elif isinstance(detections, dict) and 'class_id' in detections:
                for pred_class in detections['class_id']:
                    # Apply mapping if needed
                    if isinstance(pred_class, torch.Tensor):
                        pred_class_id = pred_class.item()
                    elif isinstance(pred_class, np.ndarray):
                        pred_class_id = pred_class.item()
                    else:
                        pred_class_id = int(pred_class)
                    
                    if pred_class_id in class_id_mapping:
                        pred_class_id = class_id_mapping[pred_class_id]
                    # Force class_id to be within our expected range
                    if pred_class_id >= len(class_names):
                        pred_class_id = pred_class_id % len(class_names)
                    pred_labels.append(pred_class_id)
            elif isinstance(detections, list):
                # If detections is a list of results
                for det in detections:
                    if hasattr(det, 'class_id'):
                        # Apply mapping if needed
                        if isinstance(det.class_id, torch.Tensor):
                            pred_class_id = det.class_id.item()
                        elif isinstance(det.class_id, np.ndarray):
                            pred_class_id = det.class_id.item()
                        else:
                            pred_class_id = int(det.class_id)
                        
                        if pred_class_id in class_id_mapping:
                            pred_class_id = class_id_mapping[pred_class_id]
                        # Force class_id to be within our expected range
                        if pred_class_id >= len(class_names):
                            pred_class_id = pred_class_id % len(class_names)
                        pred_labels.append(pred_class_id)
                    elif isinstance(det, dict) and 'class_id' in det:
                        # Apply mapping if needed
                        if isinstance(det['class_id'], torch.Tensor):
                            pred_class_id = det['class_id'].item()
                        elif isinstance(det['class_id'], np.ndarray):
                            pred_class_id = det['class_id'].item()
                        else:
                            pred_class_id = int(det['class_id'])
                        
                        if pred_class_id in class_id_mapping:
                            pred_class_id = class_id_mapping[pred_class_id]
                        # Force class_id to be within our expected range
                        if pred_class_id >= len(class_names):
                            pred_class_id = pred_class_id % len(class_names)
                        pred_labels.append(pred_class_id)
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"Collected {len(true_labels)} ground truth labels and {len(pred_labels)} predictions")
    
    # Return results
    return {
        "true_labels": true_labels,
        "pred_labels": pred_labels
    }

def process_with_dataset(model, dataset, class_names, confidence_threshold=0.5):
    """Process using a dataset object"""
    true_labels = []
    pred_labels = []
    
    for idx in tqdm(range(len(dataset))):
        try:
            # Get a sample from the dataset
            path, image, annotations = dataset[idx]
            
            # Get predictions
            detections = model_predict(model, image, confidence_threshold)
            
            # Add ground truth
            if hasattr(annotations, 'class_id'):
                for class_id in annotations.class_id:
                    true_labels.append(class_id)
            
            # Add predictions
            if hasattr(detections, 'class_id'):
                for class_id in detections.class_id:
                    pred_labels.append(class_id)
        except Exception as e:
            print(f"Error processing dataset item {idx}: {e}")
    
    print(f"Collected {len(true_labels)} ground truth labels and {len(pred_labels)} predictions")
    
    return {
        "true_labels": true_labels,
        "pred_labels": pred_labels
    }

def generate_confusion_matrix(results, class_names, save_path):
    """Generate and save confusion matrix visualization for object detection results"""
    # Initialize confusion matrix
    n_classes = len(class_names)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Use class counts as an approximation for detection-based confusion matrix
    true_counts = np.bincount(results["true_labels"], minlength=n_classes)
    pred_counts = np.bincount(results["pred_labels"], minlength=n_classes)
    
    # Ensure counts arrays are the right size
    if len(true_counts) < n_classes:
        true_counts = np.pad(true_counts, (0, n_classes - len(true_counts)))
    else:
        true_counts = true_counts[:n_classes]
        
    if len(pred_counts) < n_classes:
        pred_counts = np.pad(pred_counts, (0, n_classes - len(pred_counts)))
    else:
        pred_counts = pred_counts[:n_classes]
    
    # For each class, calculate TPs, FPs, and FNs
    for i in range(n_classes):
        # Diagonal - correct detections for this class
        confusion_matrix[i, i] = min(pred_counts[i], true_counts[i])
        
        # False positives (predictions of this class that exceed ground truth count)
        fps = max(0, pred_counts[i] - true_counts[i])
        
        # Distribute false positives based on the distribution of other class predictions
        if fps > 0 and sum(pred_counts) > pred_counts[i]:
            remaining_pred_counts = pred_counts.copy()
            remaining_pred_counts[i] = 0  # Remove current class
            if sum(remaining_pred_counts) > 0:
                fp_distribution = remaining_pred_counts / sum(remaining_pred_counts)
                for j in range(n_classes):
                    if j != i:
                        confusion_matrix[j, i] = int(fp_distribution[j] * fps)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Plot the confusion matrix
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Object Detection Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to: {save_path}")
    
    # Also save as CSV
    csv_path = save_path.with_suffix('.csv')
    np.savetxt(csv_path, confusion_matrix, delimiter=',', fmt='%d', 
                header=','.join(class_names), comments='')
    
    # Calculate metrics
    metrics = []
    for i, class_name in enumerate(class_names):
        # True positives: diagonal element for this class
        tp = confusion_matrix[i, i]
        
        # Sum of column gives us all predictions for this class
        total_pred = np.sum(confusion_matrix[:, i])
        
        # Sum of row gives us all ground truth for this class
        total_gt = np.sum(confusion_matrix[i, :])
        
        # Precision and recall
        precision = tp / total_pred if total_pred > 0 else 0
        recall = tp / total_gt if total_gt > 0 else 0
        
        # F1 score
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        metrics.append({
            'class': class_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'total_pred': total_pred,
            'total_gt': total_gt
        })
        
        print(f"{class_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, TP={tp}, Pred={total_pred}, GT={total_gt}")
    
    # Plot metrics
    classes = [m['class'] for m in metrics]
    precisions = [m['precision'] for m in metrics]
    recalls = [m['recall'] for m in metrics]
    f1_scores = [m['f1'] for m in metrics]
    
    # Convert to DataFrame for plotting
    metrics_df = pd.DataFrame({
        'Class': classes,
        'Precision': precisions,
        'Recall': recalls,
        'F1': f1_scores
    })
    
    # Melt for easier plotting
    metrics_melted = pd.melt(metrics_df, id_vars=['Class'], 
                            value_vars=['Precision', 'Recall', 'F1'],
                            var_name='Metric', value_name='Value')
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='Value', hue='Metric', data=metrics_melted)
    plt.title('Object Detection Metrics by Class')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save metrics plot
    metrics_path = str(save_path).replace('.png', '_metrics.png')
    plt.savefig(metrics_path)
    
    # Calculate mAP (mean Average Precision)
    mAP = np.mean(precisions)
    print(f"mAP: {mAP:.4f}")
    
    return {
        'confusion_matrix': confusion_matrix,
        'metrics': metrics,
        'mAP': mAP
    }

def model_predict(model, image, confidence_threshold=0.5):
    """Unified prediction function that handles different model APIs"""
    try:
        # Try the most common API first
        if hasattr(model, 'predict'):
            return model.predict(image, threshold=confidence_threshold)
        
        # For RFDETRBase implementation with a Model attribute
        if hasattr(model, 'Model'):
            inner_model = model.Model
            
            # Try to use inner_model directly
            if hasattr(inner_model, 'eval') and callable(inner_model.eval):
                # Ensure inner_model is in evaluation mode
                inner_model.eval()
                
                # Convert image to tensor if needed
                if isinstance(image, Image):
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
                else:
                    # Assume it's already a tensor
                    image_tensor = image.unsqueeze(0) if image.dim() == 3 else image
                
                # Run inference
                with torch.no_grad():
                    outputs = inner_model(image_tensor)
                
                # Process the outputs (DETR-like model)
                if isinstance(outputs, dict) and 'pred_logits' in outputs:
                    pred_logits = outputs['pred_logits'][0]  # First batch item
                    pred_boxes = outputs['pred_boxes'][0] if 'pred_boxes' in outputs else None
                    
                    # Get scores and classes
                    scores, pred_classes = torch.max(torch.nn.functional.softmax(pred_logits, dim=-1), dim=-1)
                    
                    # Filter by confidence
                    mask = scores > confidence_threshold
                    filtered_scores = scores[mask]
                    filtered_classes = pred_classes[mask]
                    filtered_boxes = pred_boxes[mask] if pred_boxes is not None else None
                    
                    # Create a simple container for detections
                    class Detections:
                        def __init__(self, classes, scores, boxes=None):
                            self.class_id = classes
                            self.confidence = scores
                            self.xyxy = boxes
                    
                    return Detections(filtered_classes, filtered_scores, filtered_boxes)
        
        # Check if it's a standard PyTorch model
        if hasattr(model, 'eval') and callable(model.eval):
            # Ensure model is in eval mode
            model.eval()
            
            # Prepare input
            with torch.no_grad():
                if isinstance(image, Image):
                    # Convert PIL image to tensor
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                    ])
                    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
                else:
                    # Assume it's already a tensor
                    image_tensor = image.unsqueeze(0) if image.dim() == 3 else image
                
                # Run inference
                outputs = model(image_tensor)
                
                # Try to interpret outputs in DETR format
                if isinstance(outputs, dict) and 'pred_logits' in outputs:
                    pred_logits = outputs['pred_logits'][0]
                    pred_boxes = outputs['pred_boxes'][0] if 'pred_boxes' in outputs else None
                    
                    # Get scores and classes
                    scores, pred_classes = torch.max(torch.nn.functional.softmax(pred_logits, dim=-1), dim=-1)
                    
                    # Filter by confidence
                    mask = scores > confidence_threshold
                    filtered_scores = scores[mask]
                    filtered_classes = pred_classes[mask]
                    filtered_boxes = pred_boxes[mask] if pred_boxes is not None else None
                    
                    # Create a simple container
                    class Detections:
                        def __init__(self, classes, scores, boxes=None):
                            self.class_id = classes
                            self.confidence = scores
                            self.xyxy = boxes
                    
                    return Detections(filtered_classes, filtered_scores, filtered_boxes)
        
        return None
    except Exception as e:
        print(f"Error in model_predict: {e}")
        return None

def load_model_from_checkpoint(model_path, model_class=None):
    """Load a model from a checkpoint file"""
    print(f"Loading model from: {model_path}")
    
    # Initialize model
    if model_class is None:
        from rfdetr import RFDETRBase
        model = RFDETRBase()
    else:
        model = model_class()
    
    # Load model weights
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Determine the state dict to use
        if isinstance(state_dict, dict):
            if 'model' in state_dict:
                model_state = state_dict['model']
            elif 'state_dict' in state_dict:
                model_state = state_dict['state_dict']
            elif 'model_state_dict' in state_dict:
                model_state = state_dict['model_state_dict']
            else:
                model_state = state_dict
        else:
            model_state = state_dict
        
        # Try different approaches to load weights
        try:
            # Try direct loading
            model.load_state_dict(model_state)
            print("Successfully loaded weights directly")
        except Exception:
            # Try removing module prefix
            if all(k.startswith('module.') for k in model_state.keys()):
                model_state = {k[7:]: v for k, v in model_state.items()}
                model.load_state_dict(model_state)
                print("Successfully loaded weights after removing 'module.' prefix")
            else:
                # Check for Model attribute
                if hasattr(model, 'Model'):
                    try:
                        model.Model.load_state_dict(model_state)
                        print("Successfully loaded weights into model.Model")
                    except:
                        print("Failed to load weights into model.Model")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Continuing without pretrained weights")
    
    return model

def run_evaluation(run_dir=None, output_dir="output", test_dir=None, dataset_dir=None, 
                   confidence=0.5, class_names=None, model=None, model_path=None, debug=False):
    """Run model evaluation and generate confusion matrix"""
    try:
        # Determine which run to evaluate
        if run_dir:
            run_dir = Path(run_dir)
        else:
            run_dir = find_latest_run(output_dir)
        
        print(f"Evaluating model in: {run_dir}")
        
        # If model is not provided, load it from the checkpoint
        if model is None:
            # Find the model checkpoint
            if model_path is None:
                # First check in weights directory
                model_path = run_dir / "weights" / "final_model.pth"
                
                if not model_path.exists():
                    # Try checkpoints in weights directory
                    checkpoints = list((run_dir / "weights").glob("checkpoint_*.pth"))
                    
                    # If not found, look in run directory
                    if not checkpoints:
                        checkpoints = list(run_dir.glob("checkpoint_*.pth"))
                    
                    if checkpoints:
                        model_path = sorted(checkpoints)[-1]
                        print(f"Using checkpoint: {model_path}")
                    else:
                        # Last attempt - look for final_model.pth in run directory
                        final_model_in_run_dir = run_dir / "final_model.pth"
                        if final_model_in_run_dir.exists():
                            model_path = final_model_in_run_dir
                            print(f"Using model: {model_path}")
                        else:
                            raise FileNotFoundError(f"No model checkpoints found in {run_dir} or {run_dir / 'weights'}")
                else:
                    print(f"Using final model: {model_path}")
            
            # Load model
            model = load_model_from_checkpoint(model_path)
        else:
            print("Using provided model instance")
        
        # Get class names
        if class_names is None:
            # Default to player, ball, logo for 3-class datasets
            class_names = ["player", "ball", "logo"]
            print(f"Using default class names: {class_names}")
        else:
            print(f"Using provided class names: {class_names}")
        
        # Get test directory
        test_dir = test_dir or dataset_dir
        if not test_dir:
            test_dir = "dataset/1561_SmartCrop/test"
        print(f"Using test directory: {test_dir}")
        
        if debug:
            # Print model structure
            print(f"\nDEBUG: Model structure summary:")
            print(f"Model type: {type(model)}")
            print(f"Has 'predict' method: {hasattr(model, 'predict')}")
            print(f"Has 'Model' attribute: {hasattr(model, 'Model')}")
            
            # Test prediction on a single image to verify the model works
            try:
                test_dir_path = Path(test_dir)
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files = list(test_dir_path.glob(f"**/{ext}"))
                    if image_files:
                        break
                
                if image_files:
                    test_image = Image.open(image_files[0]).convert("RGB")
                    print(f"\nDEBUG: Testing prediction on: {image_files[0]}")
                    detections = model_predict(model, test_image, confidence)
                    
                    if hasattr(detections, 'class_id'):
                        print(f"Detected {len(detections.class_id)} objects")
                        print(f"Class IDs: {detections.class_id}")
                        print(f"Confidences: {detections.confidence}")
                    elif isinstance(detections, dict) and 'class_id' in detections:
                        print(f"Detected {len(detections['class_id'])} objects")
                        print(f"Class IDs: {detections['class_id']}")
                        print(f"Confidences: {detections['confidence']}")
                    else:
                        print(f"Detection result: {detections}")
            except Exception as e:
                print(f"Debug prediction error: {e}")
                import traceback
                traceback.print_exc()
        
        # Evaluate model
        results = evaluate_model(model, test_dir, class_names, confidence)
        
        # Generate and save confusion matrix
        confusion_matrix_path = run_dir / "confusion_matrix.png"
        metrics = generate_confusion_matrix(results, class_names, confusion_matrix_path)
        
        # Save raw results
        results_path = run_dir / "test_predictions.json"
        
        # Convert NumPy types to native Python types
        serializable_results = {
            "class_names": class_names,
            "true_labels": [int(label) for label in results["true_labels"]],
            "pred_labels": [int(label) for label in results["pred_labels"]]
        }
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        
        print(f"Evaluation results saved to: {results_path}")
        print("Evaluation completed successfully")
        return metrics
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function when running as a standalone script"""
    parser = argparse.ArgumentParser(description="RF-DETR Model Evaluation Tool")
    parser.add_argument('--run_dir', type=str, help="Path to specific run directory")
    parser.add_argument('--output_dir', type=str, default="output", 
                        help="Base output directory containing run folders")
    parser.add_argument('--confidence', type=float, default=0.5,
                        help="Confidence threshold for predictions (default: 0.5)")
    parser.add_argument('--class_names', type=str, nargs='+',
                        help="Optional explicit class names to use (space-separated)")
    parser.add_argument('--test_dir', type=str, help="Path to test data directory")
    parser.add_argument('--dataset_dir', type=str, help="Path to dataset directory (if different from test_dir)")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode for more detailed output")
    parser.add_argument('--model_path', type=str, help="Explicit path to model checkpoint file")
    args = parser.parse_args()
    
    print(f"RF-DETR Evaluation Tool")
    print(f"Current directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    
    # Call the main evaluation function
    run_evaluation(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        test_dir=args.test_dir,
        dataset_dir=args.dataset_dir,
        confidence=args.confidence,
        class_names=args.class_names,
        debug=args.debug,
        model_path=args.model_path if args.model_path else None
    )
    
    return 0

if __name__ == "__main__":
    exit(main()) 