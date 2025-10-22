import numpy as np
import pandas as pd
import os
from typing import Tuple
from collections import Counter
from glob import glob

def cut_undersample(dataset_dir: str, X: pd.Series = None, y: pd.Series = None, random: int = 42) -> Tuple[pd.Series, pd.Series]:
    """
    Undersample majority classes to match minority class count.
    Takes dataset directory with structure {label}/{image}.jpg
    
    Args:
        dataset_dir (str): Directory containing label folders with images
        X (pd.Series): Image filenames 
        y (pd.Series): Labels
        random (int): Random state
    
    Returns:
        Tuple: X_resampled, y_resampled with balanced classes
    """
    # Get all label directories
    label_dirs = [d for d in os.listdir(dataset_dir) 
                 if os.path.isdir(os.path.join(dataset_dir, d))]
    
    np.random.seed(random)
    
    # Count files in each label directory
    label_counts = {}
    for label in label_dirs:
        label_path = os.path.join(dataset_dir, label)
        label_counts[label] = len(glob(os.path.join(label_path, "*.jpg")))
        print(f"Label '{label}' original count: {label_counts[label]}")
    
    min_count = min(label_counts.values())
    print(f"\nMinimum count (target for undersampling): {min_count}\n")
    
    X_resampled = []
    y_resampled = []
    
    for label in label_dirs:
        label_path = os.path.join(dataset_dir, label)
        # Get all image files in this label directory
        image_files = glob(os.path.join(label_path, "*.jpg"))
        
        # Sample min_count images from this class
        if len(image_files) > min_count:
            sampled_images = np.random.choice(image_files, size=min_count, replace=False)
        else:
            sampled_images = image_files
            
        # Get relative paths for consistency
        sampled_images = [os.path.relpath(img, dataset_dir) for img in sampled_images]
        
        X_resampled.extend(sampled_images)
        y_resampled.extend([label] * len(sampled_images))
    
    # Print final counts
    final_counts = Counter(y_resampled)
    print("\nAfter undersampling:")
    for label in label_dirs:
        print(f"Label '{label}' final count: {final_counts[label]}")
    
    return pd.Series(X_resampled), pd.Series(y_resampled)

if __name__ == "__main__":
    # Example usage
    dataset_dir = "datasets/ISIC2018_Task3_Training_Input"

    X_resampled, y_resampled = cut_undersample(dataset_dir)
    
    