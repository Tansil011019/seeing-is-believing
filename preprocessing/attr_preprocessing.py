"""
Attribute preprocessing for ISIC Task 2
Converts binary attribute masks to multilabel classification labels
"""
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Feature presence threshold - if pixel portion > FEAT_THRESHOLD, mark as positive
FEAT_THRESHOLD = 0.1

# Five attribute types in ISIC Task 2
ATTR_TYPES = [
    'globules',
    'milia_like_cyst', 
    'negative_network',
    'pigment_network',
    'streaks'
]


def compute_attribute_label(mask_path: str) -> int:
    """
    Convert attribute mask to binary label based on feature presence
    
    Args:
        mask_path: Path to attribute mask image
        
    Returns:
        1 if feature present (portion > FEAT_THRESHOLD), 0 otherwise
    """
    if not os.path.exists(mask_path):
        return 0
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0
    
    # Calculate portion of positive pixels
    positive_pixels = np.sum(mask > 127)
    total_pixels = mask.size
    portion = positive_pixels / total_pixels
    
    return 1 if portion > FEAT_THRESHOLD else 0


def get_multilabel_for_image(
    image_id: str,
    gt_folder: str
) -> np.ndarray:
    """
    Get multilabel vector for an image across all attributes
    
    Args:
        image_id: ISIC image ID (e.g., 'ISIC_0013794')
        gt_folder: Path to ground truth folder
        
    Returns:
        Binary vector of shape (5,) with labels for each attribute
    """
    labels = []
    
    for attr_type in ATTR_TYPES:
        mask_filename = f"{image_id}_attribute_{attr_type}.png"
        mask_path = os.path.join(gt_folder, mask_filename)
        label = compute_attribute_label(mask_path)
        labels.append(label)
    
    return np.array(labels, dtype=np.float32)


def process_dataset_labels(
    image_folder: str,
    gt_folder: str
) -> Tuple[List[str], np.ndarray]:
    """
    Process entire dataset to generate multilabel matrix
    
    Args:
        image_folder: Path to input images
        gt_folder: Path to ground truth masks
        
    Returns:
        Tuple of (image_ids, labels_matrix)
        - image_ids: List of image IDs
        - labels_matrix: numpy array of shape (N, 5) with labels
    """
    image_ids = []
    labels_list = []
    
    # Get all image files
    img_files = sorted([
        f for f in os.listdir(image_folder)
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    
    for img_file in img_files:
        # Extract image ID from filename
        image_id = os.path.splitext(img_file)[0]
        
        # Get multilabel vector for this image
        labels = get_multilabel_for_image(image_id, gt_folder)
        
        image_ids.append(image_id)
        labels_list.append(labels)
    
    labels_matrix = np.array(labels_list, dtype=np.float32)
    
    return image_ids, labels_matrix


def get_label_statistics(labels_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for multilabel dataset
    
    Args:
        labels_matrix: Labels array of shape (N, 5)
        
    Returns:
        Dictionary with statistics for each attribute
    """
    stats = {}
    
    for idx, attr_type in enumerate(ATTR_TYPES):
        positive_count = np.sum(labels_matrix[:, idx])
        total_count = labels_matrix.shape[0]
        positive_rate = positive_count / total_count
        
        stats[attr_type] = {
            'positive_count': int(positive_count),
            'total_count': total_count,
            'positive_rate': float(positive_rate)
        }
    
    return stats
