"""
Attribute Detection Evaluation Metrics

Provides comprehensive metrics for multi-label classification:
- F1 scores (macro, micro, weighted, per-class)
- Precision and Recall (macro, micro, weighted)
- Accuracy (exact match and hamming)
- Hamming Loss
- Subset Accuracy
"""

import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    hamming_loss as sklearn_hamming_loss
)
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_multilabel_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.5,
    label_names: Optional[list] = None
) -> Dict[str, float]:
    """
    Compute comprehensive multi-label classification metrics.
    
    Args:
        y_true: Ground truth labels [batch_size, num_classes]
        y_pred: Predicted logits or probabilities [batch_size, num_classes]
        threshold: Threshold for converting probabilities to binary predictions
        label_names: Optional list of label names for per-class metrics
    
    Returns:
        Dictionary with all computed metrics
    """
    # Convert to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Apply sigmoid if logits
    if y_pred.max() > 1.0 or y_pred.min() < 0.0:
        y_pred = 1 / (1 + np.exp(-y_pred))  # sigmoid
    
    # Convert to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Compute metrics
    metrics = {}
    
    # F1 scores
    metrics['f1_macro'] = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    metrics['f1_micro'] = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred_binary, average='weighted', zero_division=0)
    
    # Precision and Recall
    metrics['precision_macro'] = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
    metrics['precision_micro'] = precision_score(y_true, y_pred_binary, average='micro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred_binary, average='weighted', zero_division=0)
    
    metrics['recall_macro'] = recall_score(y_true, y_pred_binary, average='macro', zero_division=0)
    metrics['recall_micro'] = recall_score(y_true, y_pred_binary, average='micro', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred_binary, average='weighted', zero_division=0)
    
    # Accuracy metrics
    metrics['subset_accuracy'] = accuracy_score(y_true, y_pred_binary)
    metrics['hamming_loss'] = sklearn_hamming_loss(y_true, y_pred_binary)
    metrics['accuracy'] = 1 - metrics['hamming_loss']  # Element-wise accuracy
    
    # Per-class F1 scores
    per_class_f1 = f1_score(y_true, y_pred_binary, average=None, zero_division=0)
    if label_names and len(label_names) == len(per_class_f1):
        for label_name, f1_val in zip(label_names, per_class_f1):
            metrics[f'f1_{label_name}'] = f1_val
    else:
        for i, f1_val in enumerate(per_class_f1):
            metrics[f'f1_class_{i}'] = f1_val
    
    return metrics


def evaluate_attr_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_names: Optional[list] = None,
    threshold: float = 0.5
) -> Tuple[Dict[str, float], float]:
    """
    Evaluate attribute detection model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function (e.g., BCEWithLogitsLoss)
        device: Device to run evaluation on
        label_names: Optional list of attribute names
        threshold: Threshold for binary predictions
    
    Returns:
        Tuple of (metrics_dict, avg_loss)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions and labels
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    metrics = compute_multilabel_metrics(
        all_labels,
        all_preds,
        threshold=threshold,
        label_names=label_names
    )
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics['loss'] = avg_loss
    return metrics


def print_metrics_summary(metrics: Dict[str, float], split: str = "Validation"):
    """
    Pretty print metrics summary.
    
    Args:
        metrics: Dictionary of computed metrics
        split: Name of the data split (e.g., "Validation", "Test")
    """
    print(f"\n{'='*60}")
    print(f"{split} Metrics Summary")
    print(f"{'='*60}")
    
    # Main metrics
    print(f"F1 Score (macro):      {metrics.get('f1_macro', 0.0):.4f}")
    print(f"F1 Score (micro):      {metrics.get('f1_micro', 0.0):.4f}")
    print(f"F1 Score (weighted):   {metrics.get('f1_weighted', 0.0):.4f}")
    print(f"Precision (macro):     {metrics.get('precision_macro', 0.0):.4f}")
    print(f"Recall (macro):        {metrics.get('recall_macro', 0.0):.4f}")
    print(f"Accuracy:              {metrics.get('accuracy', 0.0):.4f}")
    print(f"Subset Accuracy:       {metrics.get('subset_accuracy', 0.0):.4f}")
    print(f"Hamming Loss:          {metrics.get('hamming_loss', 0.0):.4f}")
    
    # Per-class metrics if available
    per_class = {k: v for k, v in metrics.items() if k.startswith('f1_') and k not in ['f1_macro', 'f1_micro', 'f1_weighted']}
    if per_class:
        print(f"\nPer-Class F1 Scores:")
        for label, f1 in sorted(per_class.items()):
            print(f"  {label:30s}: {f1:.4f}")
    
    print(f"{'='*60}\n")


# Default ISIC 2018 Task 2 attribute names
ISIC_TASK2_LABELS = [
    'globules',
    'milia_like_cyst',
    'negative_network',
    'pigment_network',
    'streaks'
]


# Legacy function name for backward compatibility
def evaluate_multilabel(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Legacy function for backward compatibility.
    Use evaluate_attr_model() for new code.
    """
    metrics, avg_loss = evaluate_attr_model(model, dataloader, criterion, device)
    metrics['loss'] = avg_loss
    # Map new names to old names
    metrics['exact_match'] = metrics['subset_accuracy']
    return metrics
