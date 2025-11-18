"""
Attribute Detection Training Visualization

Visualizes attribute detection training metrics from CSV file:
- Train Loss, Val Loss
- Train F1-Macro, Val F1-Macro

Usage:
    python utils/viz/attr_train_viz.py <csv_path>
    
Example:
    python utils/viz/attr_train_viz.py outputs/attr_transfer_ecvit_metrics_20251117_230519.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_attribute_training(csv_path):
    """
    Create 4 graphs for attribute detection training metrics.
    
    Args:
        csv_path: Path to CSV file with training metrics
    """
    # Read CSV
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    
    # Extract CSV name without extension
    csv_name = csv_path.stem
    
    # Create figure with 2 rows, 2 columns (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Attribute Detection Training Metrics - {csv_name}', fontsize=16, y=0.995)
    
    # Color scheme
    train_color = '#2E86AB'  # Blue
    val_color = '#A23B72'    # Purple
    
    # Check if we have train F1 macro (might not exist in CSV)
    has_train_f1 = 'train_f1_macro' in df.columns
    
    # Plot 1: Train Loss
    ax = axes[0, 0]
    ax.plot(df['epoch'], df['train_loss'], color=train_color, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=1)
    
    # Add min value annotation
    min_loss = df['train_loss'].min()
    min_epoch = df.loc[df['train_loss'].idxmin(), 'epoch']
    ax.axhline(y=min_loss, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(0.98, min_loss, f'Min: {min_loss:.6f}\n(Epoch {int(min_epoch)})', 
            ha='right', va='top', fontsize=9, color='green',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='green'))
    
    # Plot 2: Val Loss
    ax = axes[0, 1]
    ax.plot(df['epoch'], df['val_loss'], color=val_color, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=1)
    
    # Add min value annotation
    min_val_loss = df['val_loss'].min()
    min_val_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
    ax.axhline(y=min_val_loss, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(0.98, min_val_loss, f'Min: {min_val_loss:.6f}\n(Epoch {int(min_val_epoch)})', 
            ha='right', va='top', fontsize=9, color='green',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='green'))
    
    # Plot 3: Train F1-Macro
    ax = axes[1, 0]
    if has_train_f1:
        ax.plot(df['epoch'], df['train_f1_macro'], color=train_color, linewidth=2, marker='o', markersize=3)
        ax.set_ylabel('F1-Macro', fontsize=11)
        ax.set_ylim(bottom=0, top=1)
        
        # Add max value annotation
        max_f1 = df['train_f1_macro'].max()
        max_epoch = df.loc[df['train_f1_macro'].idxmax(), 'epoch']
        ax.axhline(y=max_f1, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
        ax.text(0.98, max_f1, f'Max: {max_f1:.4f}\n(Epoch {int(max_epoch)})', 
                ha='right', va='bottom', fontsize=9, color='green',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='green'))
    else:
        ax.text(0.5, 0.5, 'Train F1-Macro\nNot Available', 
                ha='center', va='center', fontsize=12, color='gray',
                transform=ax.transAxes)
        ax.set_yticks([])
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_title('Training F1-Macro', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=1)
    
    # Plot 4: Val F1-Macro
    ax = axes[1, 1]
    ax.plot(df['epoch'], df['val_f1_macro'], color=val_color, linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('F1-Macro', fontsize=11)
    ax.set_title('Validation F1-Macro', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=1)
    ax.set_ylim(bottom=0, top=1)
    
    # Add max value annotation
    max_val_f1 = df['val_f1_macro'].max()
    max_val_epoch = df.loc[df['val_f1_macro'].idxmax(), 'epoch']
    ax.axhline(y=max_val_f1, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(0.98, max_val_f1, f'Max: {max_val_f1:.4f}\n(Epoch {int(max_val_epoch)})', 
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
    print(f"Total Epochs:           {len(df)}")
    print(f"Final Train Loss:       {df['train_loss'].iloc[-1]:.6f}")
    print(f"Best Train Loss:        {min_loss:.6f} (Epoch {int(min_epoch)})")
    print(f"Final Val Loss:         {df['val_loss'].iloc[-1]:.6f}")
    print(f"Best Val Loss:          {min_val_loss:.6f} (Epoch {int(min_val_epoch)})")
    
    if has_train_f1:
        print(f"Final Train F1-Macro:   {df['train_f1_macro'].iloc[-1]:.6f}")
        print(f"Best Train F1-Macro:    {max_f1:.6f} (Epoch {int(max_epoch)})")
    else:
        print(f"Train F1-Macro:         Not Available")
    
    print(f"Final Val F1-Macro:     {df['val_f1_macro'].iloc[-1]:.6f}")
    print(f"Best Val F1-Macro:      {max_val_f1:.6f} (Epoch {int(max_val_epoch)})")
    
    # Show additional metrics if available
    if 'val_f1_micro' in df.columns:
        print(f"\nAdditional Metrics (Final Epoch):")
        print(f"Val F1-Micro:           {df['val_f1_micro'].iloc[-1]:.6f}")
    if 'val_f1_weighted' in df.columns:
        print(f"Val F1-Weighted:        {df['val_f1_weighted'].iloc[-1]:.6f}")
    if 'val_accuracy' in df.columns:
        print(f"Val Accuracy:           {df['val_accuracy'].iloc[-1]:.6f}")
    if 'val_subset_accuracy' in df.columns:
        print(f"Val Subset Accuracy:    {df['val_subset_accuracy'].iloc[-1]:.6f}")
    
    # Show per-class F1 scores if available
    per_class_cols = [col for col in df.columns if col.startswith('val_f1_') 
                      and col not in ['val_f1_macro', 'val_f1_micro', 'val_f1_weighted']]
    if per_class_cols:
        print(f"\nPer-Class F1 Scores (Final Epoch):")
        for col in sorted(per_class_cols):
            label = col.replace('val_f1_', '').replace('_', ' ').title()
            print(f"  {label:25s}: {df[col].iloc[-1]:.6f}")
    
    print(f"{'='*60}\n")
    
    plt.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python utils/viz/attr_train_viz.py <csv_path>")
        print("\nExample:")
        print("  python utils/viz/attr_train_viz.py outputs/attr_transfer_ecvit_metrics_20251117_230519.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    try:
        visualize_attribute_training(csv_path)
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
