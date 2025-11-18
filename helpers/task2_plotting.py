"""
Task 2 Attribute Detection Plotting Utilities
Visualizations for F1-scores, hamming loss, and per-attribute metrics
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


ATTRIBUTES = ['globules', 'milia_like_cyst', 'negative_network', 
              'pigment_network', 'streaks']


def plot_attribute_metrics(metrics_history: Dict[str, List[float]],
                           save_path: Optional[str] = None):
    """
    Plot training and validation metrics for attribute detection
    
    Args:
        metrics_history: Dict with 'train_loss', 'val_loss', 'val_f1', etc.
        save_path: Path to save the figure
    """
    epochs = range(1, len(metrics_history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss plot
    axes[0].plot(epochs, metrics_history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, metrics_history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (BCE)')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # F1 score plot
    if 'val_f1' in metrics_history:
        axes[1].plot(epochs, metrics_history['val_f1'], 'g-', label='Val F1')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Validation F1 Score (Micro)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
    
    # Hamming loss plot
    if 'val_hamming' in metrics_history:
        axes[2].plot(epochs, metrics_history['val_hamming'], 'm-', label='Hamming Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Hamming Loss')
        axes[2].set_title('Validation Hamming Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_per_attribute_f1(per_attr_f1: Dict[str, List[float]],
                          save_path: Optional[str] = None):
    """
    Plot F1 scores for each attribute separately
    
    Args:
        per_attr_f1: Dict mapping attribute names to F1 score lists
        save_path: Path to save the figure
    """
    n_attrs = len(per_attr_f1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (attr_name, f1_scores) in enumerate(per_attr_f1.items()):
        epochs = range(1, len(f1_scores) + 1)
        axes[idx].plot(epochs, f1_scores, 'b-', linewidth=2, marker='o', markersize=4)
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('F1 Score')
        axes[idx].set_title(f'{attr_name.replace("_", " ").title()}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([0, 1])
        
        # Mark best score
        best_f1 = max(f1_scores)
        best_epoch = f1_scores.index(best_f1) + 1
        axes[idx].axhline(y=best_f1, color='r', linestyle='--', alpha=0.5)
        axes[idx].text(0.5, 0.95, f'Best: {best_f1:.3f}', 
                      transform=axes[idx].transAxes, ha='center', va='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide extra subplot
    if n_attrs < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_attribute_comparison(final_metrics: Dict[str, float],
                              save_path: Optional[str] = None):
    """
    Bar plot comparing final F1 scores across attributes
    
    Args:
        final_metrics: Dict mapping attribute names to final F1 scores
        save_path: Path to save the figure
    """
    attributes = list(final_metrics.keys())
    scores = list(final_metrics.values())
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(attributes)))
    bars = plt.bar(range(len(attributes)), scores, color=colors)
    
    plt.xlabel('Attribute', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Final F1 Scores by Attribute', fontsize=14)
    plt.xticks(range(len(attributes)), 
              [a.replace('_', ' ').title() for a in attributes],
              rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()


def plot_confusion_heatmap(confusion_stats: Dict[str, Dict[str, int]],
                          save_path: Optional[str] = None):
    """
    Heatmap showing true positives, false positives, false negatives per attribute
    
    Args:
        confusion_stats: Dict with 'tp', 'fp', 'fn' for each attribute
        save_path: Path to save the figure
    """
    attributes = list(confusion_stats.keys())
    metrics = ['TP', 'FP', 'FN']
    
    data = np.array([[confusion_stats[attr]['tp'],
                     confusion_stats[attr]['fp'],
                     confusion_stats[attr]['fn']] for attr in attributes])
    
    plt.figure(figsize=(8, 10))
    im = plt.imshow(data, cmap='YlOrRd', aspect='auto')
    
    plt.xticks(range(len(metrics)), metrics)
    plt.yticks(range(len(attributes)), 
              [a.replace('_', ' ').title() for a in attributes])
    plt.xlabel('Metric Type', fontsize=12)
    plt.ylabel('Attribute', fontsize=12)
    plt.title('Confusion Statistics by Attribute', fontsize=14)
    
    # Add text annotations
    for i in range(len(attributes)):
        for j in range(len(metrics)):
            text = plt.text(j, i, int(data[i, j]),
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, label='Count')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.close()
