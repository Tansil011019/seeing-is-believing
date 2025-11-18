"""
Task 1 Segmentation Plotting Utilities
Visualizations for IoU, Dice, and loss curves
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


def plot_segmentation_metrics(metrics_history: Dict[str, List[float]], 
                              save_path: Optional[str] = None):
    """
    Plot training and validation metrics for segmentation
    
    Args:
        metrics_history: Dict with keys 'train_loss', 'val_loss', 'val_iou', 'val_dice'
        save_path: Path to save the figure
    """
    epochs = range(1, len(metrics_history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss plot
    axes[0].plot(epochs, metrics_history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, metrics_history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU plot
    if 'val_iou' in metrics_history:
        axes[1].plot(epochs, metrics_history['val_iou'], 'g-', label='Val IoU')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('IoU')
        axes[1].set_title('Validation IoU Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Dice plot
    if 'val_dice' in metrics_history:
        axes[2].plot(epochs, metrics_history['val_dice'], 'm-', label='Val Dice')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Dice')
        axes[2].set_title('Validation Dice Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_loss_comparison(train_losses: List[float], val_losses: List[float],
                         model_name: str, save_path: Optional[str] = None):
    """
    Simple loss comparison plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{model_name} - Training Progress', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_metric_trends(metric_values: List[float], metric_name: str,
                       save_path: Optional[str] = None):
    """
    Plot a single metric over epochs
    """
    epochs = range(1, len(metric_values) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metric_values, 'g-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.title(f'Validation {metric_name} Over Time', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line for best value
    best_val = max(metric_values)
    best_epoch = metric_values.index(best_val) + 1
    plt.axhline(y=best_val, color='r', linestyle='--', alpha=0.5,
                label=f'Best: {best_val:.4f} (Epoch {best_epoch})')
    plt.legend(fontsize=11)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()
