"""
Attribute preprocessing for ISIC 2018 Task 2
Converts binary ground truth images to CSV format with threshold-based labeling
"""
import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm


# Attribute labels
ATTRIBUTE_LABELS = [
    'globules',
    'milia_like_cyst',
    'negative_network',
    'pigment_network',
    'streaks'
]


def process_binary_image(img_path: str, threshold: float = 0.05) -> int:
    """
    Process a binary ground truth image and determine label based on threshold
    
    Args:
        img_path: Path to the binary image
        threshold: Proportion threshold for positive label (default: 0.05)
    
    Returns:
        1 if proportion of white pixels > threshold, else 0
    """
    try:
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_array = np.array(img)
        
        # Normalize to 0-1
        img_array = img_array / 255.0
        
        # Calculate proportion of '1's (white pixels)
        proportion = np.mean(img_array)
        
        # Label 1 if proportion > threshold, else 0
        return 1 if proportion > threshold else 0
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return 0


def create_attr_csv(
    gt_folder: str,
    output_csv: str,
    threshold: float = 0.05,
    verbose: bool = True
):
    """
    Create CSV file from binary ground truth images
    
    Args:
        gt_folder: Path to folder containing binary ground truth images
        output_csv: Path to output CSV file
        threshold: Proportion threshold for positive label
        verbose: Print progress information
    """
    gt_path = Path(gt_folder)
    
    if not gt_path.exists():
        raise ValueError(f"Ground truth folder not found: {gt_folder}")
    
    # Collect all image IDs
    image_ids = set()
    for file in gt_path.glob("*.png"):
        filename = file.stem
        # Extract image ID from filename: ISIC_{img_id}_attribute_{label}
        if '_attribute_' in filename:
            img_id = filename.split('_attribute_')[0]
            image_ids.add(img_id)
    
    image_ids = sorted(list(image_ids))
    
    if verbose:
        print(f"Found {len(image_ids)} unique images")
        print(f"Processing with threshold={threshold}")
    
    # Process each image and create labels
    results = []
    
    for img_id in tqdm(image_ids, desc="Processing images", disable=not verbose):
        row = {'image_id': img_id}
        
        for label in ATTRIBUTE_LABELS:
            # Construct filename: ISIC_{img_id}_attribute_{label}.png
            img_file = gt_path / f"{img_id}_attribute_{label}.png"
            
            if img_file.exists():
                label_value = process_binary_image(str(img_file), threshold)
                row[label] = label_value
            else:
                if verbose:
                    print(f"Warning: Missing file {img_file}")
                row[label] = 0
        
        results.append(row)
    
    # Write to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['image_id'] + ATTRIBUTE_LABELS
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    if verbose:
        print(f"\nCSV file created: {output_csv}")
        print(f"Total samples: {len(results)}")
        
        # Print label distribution
        print("\nLabel distribution:")
        for label in ATTRIBUTE_LABELS:
            count = sum(1 for r in results if r[label] == 1)
            percentage = (count / len(results)) * 100
            print(f"  {label}: {count}/{len(results)} ({percentage:.1f}%)")


def main():
    """
    Main function to create CSV files for training and validation sets
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert binary attribute images to CSV labels'
    )
    parser.add_argument(
        '--train_gt_folder',
        type=str,
        default='datasets/ISIC2018_Task2_Training_GroundTruth_v3',
        help='Path to training ground truth folder'
    )
    parser.add_argument(
        '--val_gt_folder',
        type=str,
        default='datasets/ISIC2018_Task2_Validation_GroundTruth',
        help='Path to validation ground truth folder'
    )
    parser.add_argument(
        '--train_output',
        type=str,
        default='datasets/ISIC2018_Task2_Training_GroundTruth.csv',
        help='Output CSV file for training set'
    )
    parser.add_argument(
        '--val_output',
        type=str,
        default='datasets/ISIC2018_Task2_Validation_GroundTruth.csv',
        help='Output CSV file for validation set'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.05,
        help='Threshold for positive label (default: 0.05)'
    )
    
    args = parser.parse_args()
    
    # Process training set
    print("="*60)
    print("Processing Training Set")
    print("="*60)
    create_attr_csv(
        args.train_gt_folder,
        args.train_output,
        args.threshold,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Processing Validation Set")
    print("="*60)
    create_attr_csv(
        args.val_gt_folder,
        args.val_output,
        args.threshold,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Preprocessing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
