"""
Evaluation utility functions
"""
import torch
import numpy as np


def extract_bbox_from_mask(mask, normalize=True):
    """
    Extract bounding box from binary mask
    
    Args:
        mask: Binary mask (H, W) or (B, 1, H, W)
        normalize: Whether to normalize coordinates to [0, 1]
    
    Returns:
        Bounding box (x_min, y_min, x_max, y_max)
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if len(mask.shape) == 4:
        # Batch of masks
        return np.array([
            extract_bbox_from_mask(mask[i, 0], normalize)
            for i in range(mask.shape[0])
        ])
    
    # Single mask
    mask = (mask > 0.5).astype(np.uint8)
    
    # Find non-zero pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        # Empty mask
        h, w = mask.shape
        return np.array(
            [0.0, 0.0, 1.0, 1.0] if normalize else [0, 0, w, h]
        )
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    if normalize:
        h, w = mask.shape
        return np.array([x_min / w, y_min / h, x_max / w, y_max / h])
    else:
        return np.array([x_min, y_min, x_max, y_max])
