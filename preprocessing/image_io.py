"""
Image loading and saving utilities
"""
import os
import cv2
import numpy as np
from typing import Tuple, Optional


def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load and convert image to RGB"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(mask_path: str) -> Optional[np.ndarray]:
    """Load and binarize mask"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save RGB image (converts to BGR for cv2)"""
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_mask(mask: np.ndarray, output_path: str) -> None:
    """Save mask"""
    cv2.imwrite(output_path, mask)


def find_mask_file(
    image_id: str,
    mask_folder: str
) -> Optional[str]:
    """
    Find corresponding mask file for an image
    
    Args:
        image_id: Image identifier (filename without extension)
        mask_folder: Path to mask folder
        
    Returns:
        Mask filename if found, None otherwise
    """
    for ext in ['.png', '.jpg', '_segmentation.png']:
        potential_mask = image_id + ext
        if os.path.exists(os.path.join(mask_folder, potential_mask)):
            return potential_mask
    return None
