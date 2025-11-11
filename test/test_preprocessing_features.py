#!/usr/bin/env python3
"""
Test script for preprocessing color features

This script tests various color preprocessing functions on skin lesion images
and saves the results for visual comparison.

Usage:
    python test/test_preprocessing_features.py --image_folder datasets/ISIC2018_Task1-2_Training_Input
    python test/test_preprocessing_features.py --image_folder datasets/ISIC2018_Task1-2_Training_Input --num_samples 5
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.color_features import (
    add_shade_of_gray,
    add_clahe_enhancement,
    add_white_balance,
    add_color_temperature_adjustment,
    add_hair_removal_filter,
    add_color_jitter
)
from preprocessing.image_io import load_image


def test_single_image(image_path: str, output_dir: Path) -> None:
    """
    Test all preprocessing functions on a single image
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
    """
    # Load image
    image = load_image(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    image_name = Path(image_path).stem
    print(f"Processing: {image_name}")
    
    # Apply all preprocessing functions
    results = {
        'Original': image,
        'Shade of Gray (p=6)': add_shade_of_gray(image, p=6),
        'Shade of Gray (p=12)': add_shade_of_gray(image, p=12),
        'CLAHE': add_clahe_enhancement(image),
        'White Balance': add_white_balance(image),
        'Warm Temperature': add_color_temperature_adjustment(image, temperature=0.3),
        'Cool Temperature': add_color_temperature_adjustment(image, temperature=-0.3),
        'Hair Removal': add_hair_removal_filter(image),
        'Color Jitter (Bright)': add_color_jitter(image, brightness=0.2, contrast=1.1, saturation=1.2),
        'Color Jitter (Dark)': add_color_jitter(image, brightness=-0.2, contrast=0.9, saturation=0.8),
    }
    
    # Create figure with subplots
    num_results = len(results)
    rows = (num_results + 2) // 3  # 3 columns
    
    fig = plt.figure(figsize=(15, 5 * rows))
    gs = GridSpec(rows, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, (title, result_img) in enumerate(results.items()):
        row = idx // 3
        col = idx % 3
        
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(result_img)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Save figure
    output_path = output_dir / f"{image_name}_preprocessing_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison to: {output_path}")
    
    # Save individual processed images
    individual_dir = output_dir / image_name
    individual_dir.mkdir(exist_ok=True)
    
    for title, result_img in results.items():
        # Clean title for filename
        clean_title = title.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
        save_path = individual_dir / f"{clean_title}.png"
        
        # Convert RGB to BGR for saving
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), result_bgr)
    
    print(f"  Saved individual images to: {individual_dir}")


def test_batch_comparison(image_paths: list, output_dir: Path) -> None:
    """
    Create a comparison grid showing one preprocessing method across multiple images
    
    Args:
        image_paths: List of image paths
        output_dir: Directory to save results
    """
    preprocessing_funcs = {
        'Shade of Gray': lambda img: add_shade_of_gray(img, p=6),
        'CLAHE': add_clahe_enhancement,
        'White Balance': add_white_balance,
        'Hair Removal': add_hair_removal_filter,
    }
    
    for func_name, func in preprocessing_funcs.items():
        print(f"\nCreating batch comparison for: {func_name}")
        
        num_images = len(image_paths)
        fig, axes = plt.subplots(num_images, 2, figsize=(8, 4 * num_images))
        
        if num_images == 1:
            axes = axes.reshape(1, -1)
        
        for idx, img_path in enumerate(image_paths):
            image = load_image(img_path)
            if image is None:
                continue
            
            processed = func(image)
            
            # Original
            axes[idx, 0].imshow(image)
            axes[idx, 0].set_title(f'Original - {Path(img_path).stem}', fontsize=9)
            axes[idx, 0].axis('off')
            
            # Processed
            axes[idx, 1].imshow(processed)
            axes[idx, 1].set_title(f'{func_name} - {Path(img_path).stem}', fontsize=9)
            axes[idx, 1].axis('off')
        
        plt.tight_layout()
        
        clean_name = func_name.replace(' ', '_').lower()
        output_path = output_dir / f"batch_comparison_{clean_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to: {output_path}")


def compute_statistics(image_paths: list) -> None:
    """
    Compute and display statistics about preprocessing effects
    
    Args:
        image_paths: List of image paths to analyze
    """
    print("\n" + "="*60)
    print("PREPROCESSING STATISTICS")
    print("="*60)
    
    stats = {
        'Original': [],
        'Shade of Gray': [],
        'CLAHE': [],
        'White Balance': [],
    }
    
    for img_path in image_paths:
        image = load_image(img_path)
        if image is None:
            continue
        
        # Calculate mean brightness for each version
        stats['Original'].append(np.mean(image))
        stats['Shade of Gray'].append(np.mean(add_shade_of_gray(image, p=6)))
        stats['CLAHE'].append(np.mean(add_clahe_enhancement(image)))
        stats['White Balance'].append(np.mean(add_white_balance(image)))
    
    print(f"\nAnalyzed {len(image_paths)} images")
    print("\nAverage Brightness (0-255):")
    for method, values in stats.items():
        if values:
            print(f"  {method:20s}: {np.mean(values):6.2f} ± {np.std(values):5.2f}")
    
    print("\nBrightness Change from Original:")
    original_mean = np.mean(stats['Original'])
    for method, values in stats.items():
        if method != 'Original' and values:
            method_mean = np.mean(values)
            change = method_mean - original_mean
            change_pct = (change / original_mean) * 100
            print(f"  {method:20s}: {change:+6.2f} ({change_pct:+5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description='Test preprocessing color features on skin lesion images',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--image_folder',
        type=str,
        required=True,
        help='Path to folder containing images to test'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=3,
        help='Number of sample images to process (default: 3)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='test/preprocessing_output',
        help='Directory to save test results (default: test/preprocessing_output)'
    )
    
    parser.add_argument(
        '--batch_comparison',
        action='store_true',
        help='Create batch comparison visualizations'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Compute and display preprocessing statistics'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    image_folder = Path(args.image_folder)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not image_folder.exists():
        print(f"Error: Image folder not found: {image_folder}")
        return 1
    
    # Get image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
    image_files = [
        f for f in image_folder.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"Error: No images found in {image_folder}")
        return 1
    
    # Select sample images
    num_samples = min(args.num_samples, len(image_files))
    sample_images = sorted(image_files)[:num_samples]
    
    print("="*60)
    print("PREPROCESSING FEATURES TEST")
    print("="*60)
    print(f"Image folder: {image_folder}")
    print(f"Output directory: {output_dir}")
    print(f"Processing {num_samples} sample images out of {len(image_files)} total")
    print("="*60)
    
    # Test each image individually
    print("\n--- Testing Individual Images ---")
    for img_path in sample_images:
        test_single_image(str(img_path), output_dir)
    
    # Batch comparison
    if args.batch_comparison:
        print("\n--- Creating Batch Comparisons ---")
        test_batch_comparison([str(p) for p in sample_images], output_dir)
    
    # Statistics
    if args.stats:
        compute_statistics([str(p) for p in sample_images])
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY")
    print(f"Results saved to: {output_dir}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
