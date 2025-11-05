"""
Model evaluation utilities
"""
import torch
import numpy as np
from .metrics import (
    calculate_iou,
    calculate_dice,
    calculate_pixel_accuracy,
    calculate_precision_recall_f1
)


def evaluate_model(model, dataloader, device, logger=None):
    """
    Evaluate model on a dataset
    
    Args:
        model: Segmentation model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        logger: Logger instance
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_metrics = {
        'ious': [],
        'dices': [],
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'f1s': []
    }
    
    if logger:
        logger.info("Starting model evaluation")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            _evaluate_batch(
                model, batch, device, all_metrics, batch_idx, 
                len(dataloader), logger
            )
    
    metrics = _compute_final_metrics(all_metrics)
    
    if logger:
        _log_metrics(metrics, logger)
    
    return metrics


def _evaluate_batch(model, batch, device, all_metrics, batch_idx, 
                    total_batches, logger):
    """Evaluate single batch"""
    images = batch['image'].to(device)
    masks = batch['mask'].to(device)
    
    # Forward pass
    outputs = _forward_model(model, images)
    
    # Calculate metrics
    iou = calculate_iou(outputs, masks)
    dice = calculate_dice(outputs, masks)
    accuracy = calculate_pixel_accuracy(outputs, masks)
    prf = calculate_precision_recall_f1(outputs, masks)
    
    # Store metrics
    all_metrics['ious'].append(iou)
    all_metrics['dices'].append(dice)
    all_metrics['accuracies'].append(accuracy)
    all_metrics['precisions'].append(prf['precision'])
    all_metrics['recalls'].append(prf['recall'])
    all_metrics['f1s'].append(prf['f1'])
    
    if logger and (batch_idx + 1) % 10 == 0:
        logger.debug(f"Evaluated batch {batch_idx + 1}/{total_batches}")


def _forward_model(model, images):
    """Forward pass handling DataParallel"""
    if isinstance(model, torch.nn.DataParallel):
        return model.module(images)
    return model(images)


def _compute_final_metrics(all_metrics):
    """Compute mean of all metrics"""
    return {
        'iou': np.mean(all_metrics['ious']),
        'dice': np.mean(all_metrics['dices']),
        'accuracy': np.mean(all_metrics['accuracies']),
        'precision': np.mean(all_metrics['precisions']),
        'recall': np.mean(all_metrics['recalls']),
        'f1': np.mean(all_metrics['f1s'])
    }


def _log_metrics(metrics, logger):
    """Log evaluation metrics"""
    logger.info("Evaluation complete")
    logger.info(f"IoU: {metrics['iou']:.4f}")
    logger.info(f"Dice: {metrics['dice']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1: {metrics['f1']:.4f}")
