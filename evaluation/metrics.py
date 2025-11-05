"""
Metric calculation utilities
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)


def _prepare_tensors(pred, target, threshold=0.5):
    """Prepare tensors for metric calculation"""
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred) if pred.max() > 1 else pred
        pred = (pred > threshold).float()
        pred = pred.cpu().numpy()
    
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    
    return pred, target


def calculate_iou(pred, target, threshold=0.5):
    """
    Calculate IoU (Intersection over Union) / Jaccard Index
    
    Args:
        pred: Predicted mask (B, 1, H, W) or (B, H, W)
        target: Ground truth mask (B, 1, H, W) or (B, H, W)
        threshold: Threshold for binary prediction
    
    Returns:
        Mean IoU across batch
    """
    pred, target = _prepare_tensors(pred, target, threshold)
    
    intersection = (pred * target).sum(axis=1)
    union = pred.sum(axis=1) + target.sum(axis=1) - intersection
    
    iou = intersection / (union + 1e-7)
    return iou.mean()


def calculate_dice(pred, target, threshold=0.5):
    """
    Calculate Dice Coefficient
    
    Args:
        pred: Predicted mask (B, 1, H, W) or (B, H, W)
        target: Ground truth mask (B, 1, H, W) or (B, H, W)
        threshold: Threshold for binary prediction
    
    Returns:
        Mean Dice coefficient across batch
    """
    pred, target = _prepare_tensors(pred, target, threshold)
    
    intersection = (pred * target).sum(axis=1)
    dice = (2. * intersection) / (
        pred.sum(axis=1) + target.sum(axis=1) + 1e-7
    )
    
    return dice.mean()


def calculate_pixel_accuracy(pred, target, threshold=0.5):
    """
    Calculate pixel-wise accuracy
    
    Args:
        pred: Predicted mask (B, 1, H, W) or (B, H, W)
        target: Ground truth mask (B, 1, H, W) or (B, H, W)
        threshold: Threshold for binary prediction
    
    Returns:
        Mean accuracy across batch
    """
    pred, target = _prepare_tensors(pred, target, threshold)
    
    accuracies = [
        accuracy_score(target[i], pred[i])
        for i in range(pred.shape[0])
    ]
    
    return np.mean(accuracies)


def calculate_precision_recall_f1(pred, target, threshold=0.5):
    """
    Calculate precision, recall, and F1 score
    
    Args:
        pred: Predicted mask (B, 1, H, W) or (B, H, W)
        target: Ground truth mask (B, 1, H, W) or (B, H, W)
        threshold: Threshold for binary prediction
    
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    pred, target = _prepare_tensors(pred, target, threshold)
    
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(pred.shape[0]):
        try:
            precision = precision_score(target[i], pred[i], zero_division=0)
            recall = recall_score(target[i], pred[i], zero_division=0)
            f1 = f1_score(target[i], pred[i], zero_division=0)
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        except:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
    
    return {
        'precision': np.mean(precisions),
        'recall': np.mean(recalls),
        'f1': np.mean(f1s)
    }
