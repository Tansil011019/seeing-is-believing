"""
Image augmentation functions
"""
import cv2
import numpy as np
from typing import Tuple, List


def rotate_image_and_mask(
    image: np.ndarray, 
    mask: np.ndarray, 
    angle: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image and mask by given angle
    
    Args:
        image: Input RGB image (H, W, 3)
        mask: Binary mask (H, W)
        angle: Rotation angle in degrees
    
    Returns:
        Rotated image and mask
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate image
    rotated_image = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    # Rotate mask
    rotated_mask = cv2.warpAffine(
        mask, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return rotated_image, rotated_mask


def dilate_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    scale_x: float,
    scale_y: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dilate (resize) image and mask by given scale factors
    
    Args:
        image: Input RGB image (H, W, 3)
        mask: Binary mask (H, W)
        scale_x: Scale factor for width
        scale_y: Scale factor for height
    
    Returns:
        Dilated image and mask
    """
    h, w = image.shape[:2]
    new_w = int(w * scale_x)
    new_h = int(h * scale_y)
    
    dilated_image = cv2.resize(
        image, (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )
    dilated_mask = cv2.resize(
        mask, (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )
    
    return dilated_image, dilated_mask
