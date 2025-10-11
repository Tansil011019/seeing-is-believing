"""
Evaluation utilities for heatmap intersection over union (IoU) calculations
and visualization functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Union


def calculate_heatmap_iou(heatmap1: np.ndarray, heatmap2: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate Intersection over Union (IoU) between two heatmaps.
    
    Args:
        heatmap1 (np.ndarray): First heatmap of shape (n, m)
        heatmap2 (np.ndarray): Second heatmap of shape (n, m)
        threshold (float): Threshold to binarize heatmaps (default: 0.5)
    
    Returns:
        float: IoU ratio between 0 and 1
    
    Raises:
        ValueError: If heatmaps have different shapes
    """
    if heatmap1.shape != heatmap2.shape:
        raise ValueError(f"Heatmap shapes must match. Got {heatmap1.shape} and {heatmap2.shape}")
    
    # Normalize heatmaps to [0, 1] range
    heatmap1_norm = (heatmap1 - heatmap1.min()) / (heatmap1.max() - heatmap1.min() + 1e-8)
    heatmap2_norm = (heatmap2 - heatmap2.min()) / (heatmap2.max() - heatmap2.min() + 1e-8)
    
    # Binarize heatmaps based on threshold
    binary1 = (heatmap1_norm >= threshold).astype(np.float32)
    binary2 = (heatmap2_norm >= threshold).astype(np.float32)
    
    # Calculate intersection and union
    intersection = np.sum(binary1 * binary2)
    union = np.sum(binary1) + np.sum(binary2) - intersection
    
    # Handle edge case where union is zero
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


def display_heatmap_intersection(heatmap1: np.ndarray, heatmap2: np.ndarray, 
                               threshold: float = 0.5, 
                               figsize: Tuple[int, int] = (15, 5),
                               save_path: str = None) -> None:
    """
    Display the intersection of two heatmaps.
    
    Args:
        heatmap1 (np.ndarray): First heatmap of shape (n, m)
        heatmap2 (np.ndarray): Second heatmap of shape (n, m)
        threshold (float): Threshold to binarize heatmaps (default: 0.5)
        figsize (Tuple[int, int]): Figure size (default: (15, 5))
        save_path (str): Path to save the figure (optional)
    """
    if heatmap1.shape != heatmap2.shape:
        raise ValueError(f"Heatmap shapes must match. Got {heatmap1.shape} and {heatmap2.shape}")
    
    # Normalize heatmaps
    heatmap1_norm = (heatmap1 - heatmap1.min()) / (heatmap1.max() - heatmap1.min() + 1e-8)
    heatmap2_norm = (heatmap2 - heatmap2.min()) / (heatmap2.max() - heatmap2.min() + 1e-8)
    
    # Calculate intersection
    binary1 = (heatmap1_norm >= threshold).astype(np.float32)
    binary2 = (heatmap2_norm >= threshold).astype(np.float32)
    intersection = binary1 * binary2
    
    # Calculate IoU for display
    iou = calculate_heatmap_iou(heatmap1, heatmap2, threshold)
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Display original heatmaps
    im1 = axes[0].imshow(heatmap1_norm, cmap='hot', interpolation='nearest')
    axes[0].set_title('Heatmap 1')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(heatmap2_norm, cmap='hot', interpolation='nearest')
    axes[1].set_title('Heatmap 2')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Display binarized versions
    axes[2].imshow(binary1, cmap='gray', interpolation='nearest')
    axes[2].set_title(f'Binary 1 (t={threshold})')
    axes[2].axis('off')
    
    # Display intersection
    im_int = axes[3].imshow(intersection, cmap='Reds', interpolation='nearest')
    axes[3].set_title(f'Intersection\nIoU: {iou:.3f}')
    axes[3].axis('off')
    plt.colorbar(im_int, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def display_heatmap_union(heatmap1: np.ndarray, heatmap2: np.ndarray, 
                         threshold: float = 0.5,
                         figsize: Tuple[int, int] = (15, 5),
                         save_path: str = None) -> None:
    """
    Display the union of two heatmaps.
    
    Args:
        heatmap1 (np.ndarray): First heatmap of shape (n, m)
        heatmap2 (np.ndarray): Second heatmap of shape (n, m)
        threshold (float): Threshold to binarize heatmaps (default: 0.5)
        figsize (Tuple[int, int]): Figure size (default: (15, 5))
        save_path (str): Path to save the figure (optional)
    """
    if heatmap1.shape != heatmap2.shape:
        raise ValueError(f"Heatmap shapes must match. Got {heatmap1.shape} and {heatmap2.shape}")
    
    # Normalize heatmaps
    heatmap1_norm = (heatmap1 - heatmap1.min()) / (heatmap1.max() - heatmap1.min() + 1e-8)
    heatmap2_norm = (heatmap2 - heatmap2.min()) / (heatmap2.max() - heatmap2.min() + 1e-8)
    
    # Calculate union
    binary1 = (heatmap1_norm >= threshold).astype(np.float32)
    binary2 = (heatmap2_norm >= threshold).astype(np.float32)
    union = np.maximum(binary1, binary2)
    
    # Calculate IoU for display
    iou = calculate_heatmap_iou(heatmap1, heatmap2, threshold)
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Display original heatmaps
    im1 = axes[0].imshow(heatmap1_norm, cmap='hot', interpolation='nearest')
    axes[0].set_title('Heatmap 1')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    im2 = axes[1].imshow(heatmap2_norm, cmap='hot', interpolation='nearest')
    axes[1].set_title('Heatmap 2')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Display binarized versions
    axes[2].imshow(binary2, cmap='gray', interpolation='nearest')
    axes[2].set_title(f'Binary 2 (t={threshold})')
    axes[2].axis('off')
    
    # Display union
    im_union = axes[3].imshow(union, cmap='Blues', interpolation='nearest')
    axes[3].set_title(f'Union\nIoU: {iou:.3f}')
    axes[3].axis('off')
    plt.colorbar(im_union, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()