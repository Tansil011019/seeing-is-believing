"""
Evaluation metrics for multilabel attribute classification
"""
import torch
import numpy as np
from sklearn.metrics import f1_score, hamming_loss, accuracy_score
from typing import Dict


def evaluate_multilabel(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model on multilabel classification task
    
    Args:
        model: Model to evaluate
        dataloader: Validation/test DataLoader
        criterion: Loss function (BCEWithLogitsLoss)
        device: Device for computation
        
    Returns:
        Dictionary with metrics: loss, f1_micro, f1_macro, hamming_loss, exact_match
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['image'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Forward pass
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Convert logits to binary predictions (threshold at 0.5)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate multilabel metrics
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    hamming = hamming_loss(all_labels, all_preds)
    exact_match = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': total_loss / len(dataloader),
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'hamming_loss': hamming,
        'exact_match': exact_match
    }
