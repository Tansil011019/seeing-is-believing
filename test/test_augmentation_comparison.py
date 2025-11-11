#!/usr/bin/env python3
"""
Test script to compare augmentation methods and verify the improvements

Usage:
    python test/test_augmentation_comparison.py --image_path path/to/image.jpg --mask_path path/to/mask.png
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.augmentation import rotate_image_and_mask, dilate_image_and_mask
from preprocessing.image_io import load_image, load_mask


def visualize_rotation_test(image: np.ndarray, mask: np.ndarray, output_dir: Path):
    """
    Test rotation with different angles and visualize results
    """
    print("\n--- Testing Rotation Augmentation ---")
    
    angles = [45, 90, 135, 180]
    
    fig = plt.figure(figsize=(20, 5 * len(angles)))
    gs = GridSpec(len(angles), 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, angle in enumerate(angles):
        print(f"  Testing rotation: {angle}°")
        
        # Apply rotation
        rotated_img, rotated_mask = rotate_image_and_mask(image, mask, angle)
        
        # Resize to 512x512 to simulate final output
        final_img = cv2.resize(rotated_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        final_mask = cv2.resize(rotated_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # Original image
        ax = fig.add_subplot(gs[idx, 0])
        ax.imshow(image)
        ax.set_title(f'Original\n{image.shape[1]}x{image.shape[0]}', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Rotated (before resize)
        ax = fig.add_subplot(gs[idx, 1])
        ax.imshow(rotated_img)
        ax.set_title(f'Rotated {angle}°\n{rotated_img.shape[1]}x{rotated_img.shape[0]}', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Final (after 512x512 resize)
        ax = fig.add_subplot(gs[idx, 2])
        ax.imshow(final_img)
        ax.set_title(f'Final Image\n512x512', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Final mask
        ax = fig.add_subplot(gs[idx, 3])
        ax.imshow(final_mask, cmap='gray')
        ax.set_title(f'Final Mask\n512x512', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        print(f"    Before resize: {rotated_img.shape}")
        print(f"    After resize: {final_img.shape}")
    
    output_path = output_dir / 'rotation_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved rotation comparison to: {output_path}")


def visualize_dilation_test(image: np.ndarray, mask: np.ndarray, output_dir: Path):
    """
    Test dilation with different scales and visualize results
    """
    print("\n--- Testing Dilation Augmentation ---")
    
    scales = [(1.0, 1.5), (1.5, 1.0), (1.5, 1.5), (2.0, 2.0)]
    
    fig = plt.figure(figsize=(20, 5 * len(scales)))
    gs = GridSpec(len(scales), 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, (scale_x, scale_y) in enumerate(scales):
        print(f"  Testing dilation: {scale_x}x{scale_y}")
        
        # Apply dilation
        dilated_img, dilated_mask = dilate_image_and_mask(image, mask, scale_x, scale_y)
        
        # Resize to 512x512 to simulate final output
        final_img = cv2.resize(dilated_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        final_mask = cv2.resize(dilated_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # Original image
        ax = fig.add_subplot(gs[idx, 0])
        ax.imshow(image)
        ax.set_title(f'Original\n{image.shape[1]}x{image.shape[0]}', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Dilated (should be same size as original due to cropping)
        ax = fig.add_subplot(gs[idx, 1])
        ax.imshow(dilated_img)
        ax.set_title(f'Dilated {scale_x}x{scale_y}\n{dilated_img.shape[1]}x{dilated_img.shape[0]}', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Final (after 512x512 resize)
        ax = fig.add_subplot(gs[idx, 2])
        ax.imshow(final_img)
        ax.set_title(f'Final Image\n512x512', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Final mask
        ax = fig.add_subplot(gs[idx, 3])
        ax.imshow(final_mask, cmap='gray')
        ax.set_title(f'Final Mask\n512x512', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        print(f"    After dilation (with crop): {dilated_img.shape}")
        print(f"    After final resize: {final_img.shape}")
    
    output_path = output_dir / 'dilation_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved dilation comparison to: {output_path}")


def test_full_pipeline(image: np.ndarray, mask: np.ndarray, output_dir: Path):
    """
    Test the full augmentation pipeline
    """
    print("\n--- Testing Full Pipeline ---")
    
    from preprocessing.augmentation_pipeline import augment_image_and_mask
    
    augmented_samples = augment_image_and_mask(image, mask)
    
    print(f"  Generated {len(augmented_samples)} augmented samples")
    
    # Show a few samples
    sample_count = min(12, len(augmented_samples))
    rows = (sample_count + 3) // 4
    
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    for idx in range(sample_count):
        aug_img, aug_mask, description = augmented_samples[idx]
        
        # Resize to 512x512
        final_img = cv2.resize(aug_img, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        axes[idx].imshow(final_img)
        axes[idx].set_title(f'{description}\n{aug_img.shape[1]}x{aug_img.shape[0]} -> 512x512', 
                           fontsize=8)
        axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(sample_count, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'full_pipeline_samples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved pipeline samples to: {output_path}")
    
    # Print size statistics
    print("\n  Size statistics:")
    sizes = {}
    for aug_img, aug_mask, description in augmented_samples:
        shape_key = f"{aug_img.shape[1]}x{aug_img.shape[0]}"
        sizes[shape_key] = sizes.get(shape_key, 0) + 1
    
    for size, count in sorted(sizes.items()):
        print(f"    {size}: {count} samples")


def main():
    parser = argparse.ArgumentParser(
        description='Test and visualize augmentation improvements',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--image_path',
        type=str,
        help='Path to test image'
    )
    
    parser.add_argument(
        '--mask_path',
        type=str,
        help='Path to test mask'
    )
    
    parser.add_argument(
        '--image_folder',
        type=str,
        help='Path to folder with images (uses first image found)'
    )
    
    parser.add_argument(
        '--mask_folder',
        type=str,
        help='Path to folder with masks (uses corresponding mask)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='test/augmentation_test_output',
        help='Directory to save test outputs'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image and mask
    if args.image_path and args.mask_path:
        image = load_image(args.image_path)
        mask = load_mask(args.mask_path)
        print(f"Loaded image from: {args.image_path}")
        print(f"Loaded mask from: {args.mask_path}")
    elif args.image_folder and args.mask_folder:
        # Find first image
        image_folder = Path(args.image_folder)
        mask_folder = Path(args.mask_folder)
        
        image_files = list(image_folder.glob('*.jpg')) + list(image_folder.glob('*.png'))
        if not image_files:
            print("No images found in folder")
            return 1
        
        image_path = str(image_files[0])
        image = load_image(image_path)
        
        # Find corresponding mask
        from preprocessing.image_io import find_mask_file
        image_id = image_files[0].stem
        mask_file = find_mask_file(image_id, str(mask_folder))
        
        if mask_file is None:
            print(f"No mask found for {image_id}")
            return 1
        
        mask_path = str(mask_folder / mask_file)
        mask = load_mask(mask_path)
        
        print(f"Loaded image from: {image_path}")
        print(f"Loaded mask from: {mask_path}")
    else:
        print("Error: Must provide either --image_path and --mask_path, or --image_folder and --mask_folder")
        return 1
    
    if image is None or mask is None:
        print("Failed to load image or mask")
        return 1
    
    print("="*60)
    print("AUGMENTATION IMPROVEMENT TEST")
    print("="*60)
    print(f"Original image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Output directory: {output_dir}")
    
    # Run tests
    visualize_rotation_test(image, mask, output_dir)
    visualize_dilation_test(image, mask, output_dir)
    test_full_pipeline(image, mask, output_dir)
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print(f"All outputs saved to: {output_dir}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
