"""
Segmentation Training Visualization

Visualizes segmentation training metrics from CSV file:
- Train Loss, Val Loss
- Train IoU, Val IoU  
- Train Dice, Val Dice

Usage:
    python utils/viz/seg_train_viz.py <csv_path>
    
Example:
    python utils/viz/seg_train_viz.py outputs/transfer_segformer_b0_metrics_20251117_120310.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_segmentation_training(csv_path):
    """
    Create 6 graphs for segmentation training metrics.
    
    Args:
        csv_path: Path to CSV file with training metrics
    """
    # Read CSV
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    
    # Extract CSV name without extension
    csv_name = csv_path.stem
    
    # Create figure with 3 rows, 2 columns (6 subplots)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'Segmentation Training Metrics - {csv_name}', fontsize=16, y=0.995)
    
    # Color scheme
    train_color = '#2E86AB'  # Blue
    val_color = '#A23B72'    # Purple
    
    # Plot 1: Train Loss
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['train_loss'], color=train_color, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=1)
    
    # Plot 2: Val Loss
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['val_loss'], color=val_color, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=1)
    
    # Check if we have train IoU and Dice (might not exist in CSV)
    has_train_iou = 'train_iou' in df.columns
    has_train_dice = 'train_dice' in df.columns
    
    # Plot 3: Train IoU
    ax = axes[1, 0]
    if has_train_iou:
        ax.plot(df['epoch'], df['train_iou'], color=train_color, linewidth=2, marker='o', markersize=3)
        ax.set_ylabel('IoU', fontsize=11)
    else:
        ax.text(0.5, 0.5, 'Train IoU\nNot Available', 
                ha='center', va='center', fontsize=12, color='gray',
                transform=ax.transAxes)
        ax.set_yticks([])
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_title('Training IoU', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=1)
    if has_train_iou:
        ax.set_ylim(bottom=0, top=1)
    
    # Plot 4: Val IoU
    ax = axes[1, 1]
    ax.plot(df['epoch'], df['val_iou'], color=val_color, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('IoU', fontsize=11)
    ax.set_title('Validation IoU', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0, top=1)
    
    # Add max value annotation
    max_iou = df['val_iou'].max()
    max_epoch = df.loc[df['val_iou'].idxmax(), 'epoch']
    ax.axhline(y=max_iou, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(0.98, max_iou, f'Max: {max_iou:.4f}\n(Epoch {int(max_epoch)})', 
            ha='right', va='bottom', fontsize=9, color='green',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='green'))
    
    # Plot 5: Train Dice
    ax = axes[2, 0]
    if has_train_dice:
        ax.plot(df['epoch'], df['train_dice'], color=train_color, linewidth=2, marker='o', markersize=3)
        ax.set_ylabel('Dice', fontsize=11)
    else:
        ax.text(0.5, 0.5, 'Train Dice\nNot Available', 
                ha='center', va='center', fontsize=12, color='gray',
                transform=ax.transAxes)
        ax.set_yticks([])
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_title('Training Dice', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=1)
    if has_train_dice:
        ax.set_ylim(bottom=0, top=1)
    
    # Plot 6: Val Dice
    ax = axes[2, 1]
    ax.plot(df['epoch'], df['val_dice'], color=val_color, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Dice', fontsize=11)
    ax.set_title('Validation Dice', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0, top=1)
    
    # Add max value annotation
    max_dice = df['val_dice'].max()
    max_epoch_dice = df.loc[df['val_dice'].idxmax(), 'epoch']
    ax.axhline(y=max_dice, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(0.98, max_dice, f'Max: {max_dice:.4f}\n(Epoch {int(max_epoch_dice)})', 
            ha='right', va='bottom', fontsize=9, color='green',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='green'))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'{csv_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    
    # Show summary statistics
    print(f"\n{'='*60}")
    print(f"Summary Statistics - {csv_name}")
    print(f"{'='*60}")
    print(f"Total Epochs:        {len(df)}")
    print(f"Final Train Loss:    {df['train_loss'].iloc[-1]:.6f}")
    print(f"Final Val Loss:      {df['val_loss'].iloc[-1]:.6f}")
    print(f"Best Val IoU:        {max_iou:.6f} (Epoch {int(max_epoch)})")
    print(f"Best Val Dice:       {max_dice:.6f} (Epoch {int(max_epoch_dice)})")
    print(f"Final Val IoU:       {df['val_iou'].iloc[-1]:.6f}")
    print(f"Final Val Dice:      {df['val_dice'].iloc[-1]:.6f}")
    print(f"{'='*60}\n")
    
    plt.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python utils/viz/seg_train_viz.py <csv_path>")
        print("\nExample:")
        print("  python utils/viz/seg_train_viz.py outputs/transfer_segformer_b0_metrics_20251117_120310.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    try:
        visualize_segmentation_training(csv_path)
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
