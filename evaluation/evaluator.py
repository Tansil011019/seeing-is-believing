"""
Model evaluation utilities
"""
from tqdm import tqdm
import torch
import numpy as np
from .metrics import (
    calculate_iou,
    calculate_dice,
    calculate_pixel_accuracy,
    calculate_precision_recall_f1
)


@torch.no_grad()  # Disables gradient calculation, saving memory and compute
def evaluate_model(model, dataloader, device, threshold=0.5, smooth=1e-6, logger=None):
    """
    Evaluate model on a dataset using the fast accumulator pattern.

    Args:
        model: Segmentation model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        threshold: Binarization threshold for predictions
        smooth: Epsilon value to prevent division by zero
        logger: Logger instance

    Returns:
        Dictionary of dataset-level evaluation metrics
    """
    model.eval()  # Set model to evaluation mode (disables dropout, etc.)

    # Initialize accumulators for the core components
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0

    if logger:
        logger.info("Starting model evaluation")

    # Use tqdm for a progress bar
    pbar = tqdm(dataloader, desc="Validation", leave=False)
    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)

        # --- 1. Fast Forward Pass ---
        # Use Automatic Mixed Precision (AMP) for faster inference
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(images)

        # --- 2. Calculate Components Once ---
        # This is the core optimization: get TP, FP, FN, TN in one pass
        tp, fp, fn, tn = _get_batch_components(outputs, masks, threshold)

        # --- 3. Accumulate ---
        # Add batch results to dataset totals
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    # --- 4. Compute Final Metrics Once ---
    # All metrics are now calculated from the dataset-level totals
    metrics = _compute_final_metrics(
        total_tp, total_fp, total_fn, total_tn, smooth
    )

    if logger:
        _log_metrics(metrics, logger)

    model.train()  # Set model back to train mode

    return metrics


def _get_batch_components(preds, targets, threshold):
    """
    Calculates TP, FP, FN, TN for a batch of predictions and targets.
    All operations are vectorized and run on the GPU.
    """
    # Binarize predictions and targets
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > 0.5).float()  # Binarize targets just in case

    # Flatten to [B*H*W]
    preds_flat = preds_bin.view(-1)
    targets_flat = targets_bin.view(-1)

    # --- Calculate components using fast, parallel torch operations ---

    # True Positives (TP): Prediction is 1, Target is 1
    tp = (preds_flat * targets_flat).sum()

    # Total positive predictions (TP + FP)
    total_pred_pos = preds_flat.sum()

    # Total positive targets (TP + FN)
    total_target_pos = targets_flat.sum()

    # False Positives (FP): Prediction is 1, Target is 0
    fp = total_pred_pos - tp

    # False Negatives (FN): Prediction is 0, Target is 1
    fn = total_target_pos - tp

    # Total number of pixels
    total_pixels = preds_flat.numel()

    # True Negatives (TN): Prediction is 0, Target is 0
    tn = total_pixels - total_pred_pos - total_target_pos + tp

    # Return as .item() to move to CPU as simple floats
    return tp.item(), fp.item(), fn.item(), tn.item()


def _compute_final_metrics(tp, fp, fn, tn, smooth):
    """
    Computes all metrics from the total accumulated components.
    """

    # --- Denominators ---
    iou_denom = tp + fp + fn + smooth
    dice_denom = 2 * tp + fp + fn + smooth
    acc_denom = tp + fp + fn + tn + smooth
    prec_denom = tp + fp + smooth
    rec_denom = tp + fn + smooth

    # --- Metrics Calculation ---
    iou = (tp + smooth) / iou_denom
    dice = (2 * tp + smooth) / dice_denom
    accuracy = (tp + tn + smooth) / acc_denom
    precision = (tp + smooth) / prec_denom
    recall = (tp + smooth) / rec_denom
    f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)

    return {
        'iou': iou,
        'dice': dice,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def _log_metrics(metrics, logger):
    """Log evaluation metrics (unchanged)"""
    if not logger:
        # Get a basic logger if one isn't provided
        logger = logging.getLogger(__name__)
        if not logger.hasHandlers():
            logging.basicConfig(level=logging.INFO)

    logger.info("Evaluation complete")
    logger.info(f"IoU: {metrics['iou']:.4f}")
    logger.info(f"Dice: {metrics['dice']:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1: {metrics['f1']:.4f}")
