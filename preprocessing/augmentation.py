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
    Rotate image and mask by given angle with automatic size adjustment
    to fit the entire rotated content without black borders.
    
    Args:
        image: Input RGB image (H, W, 3)
        mask: Binary mask (H, W)
        angle: Rotation angle in degrees
    
    Returns:
        Rotated image and mask (may have different dimensions than input)
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new bounding box size to fit entire rotated image
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust the rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Rotate image with new dimensions (no black borders)
    rotated_image = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    
    # Rotate mask with new dimensions
    rotated_mask = cv2.warpAffine(
        mask, M, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT
    )
    
    return rotated_image, rotated_mask


def dilate_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    scale_x: float,
    scale_y: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dilate (scale up) image and mask, then crop from center to maintain original dimensions.
    This creates a zoomed-in effect without changing the output size.
    
    Args:
        image: Input RGB image (H, W, 3)
        mask: Binary mask (H, W)
        scale_x: Scale factor for width (>1.0 zooms in)
        scale_y: Scale factor for height (>1.0 zooms in)
    
    Returns:
        Dilated and center-cropped image and mask (same size as input)
    """
    h, w = image.shape[:2]
    
    # Calculate new scaled dimensions
    new_w = int(w * scale_x)
    new_h = int(h * scale_y)
    
    # Scale up the image and mask
    scaled_image = cv2.resize(
        image, (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )
    scaled_mask = cv2.resize(
        mask, (new_w, new_h),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Calculate center crop coordinates
    start_x = (new_w - w) // 2
    start_y = (new_h - h) // 2
    end_x = start_x + w
    end_y = start_y + h
    
    # Ensure we don't go out of bounds (in case of scale < 1.0)
    if new_w < w or new_h < h:
        # If scaled down, pad instead of crop
        pad_x = max(0, (w - new_w) // 2)
        pad_y = max(0, (h - new_h) // 2)
        
        dilated_image = cv2.copyMakeBorder(
            scaled_image,
            pad_y, h - new_h - pad_y,
            pad_x, w - new_w - pad_x,
            cv2.BORDER_REFLECT
        )
        dilated_mask = cv2.copyMakeBorder(
            scaled_mask,
            pad_y, h - new_h - pad_y,
            pad_x, w - new_w - pad_x,
            cv2.BORDER_CONSTANT,
            value=0
        )
    else:
        # Crop from center
        dilated_image = scaled_image[start_y:end_y, start_x:end_x]
        dilated_mask = scaled_mask[start_y:end_y, start_x:end_x]
    
    return dilated_image, dilated_mask
