"""
Augmentation pipeline that combines multiple transformations
"""
import numpy as np
from typing import List, Tuple
from .augmentation import rotate_image_and_mask, dilate_image_and_mask


def augment_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    logger=None
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    """
    Apply full augmentation pipeline
    
    Args:
        image: Input RGB image (H, W, 3)
        mask: Binary mask (H, W)
        logger: Logger instance
    
    Returns:
        List of (augmented_image, augmented_mask, description)
    """
    if logger:
        logger.debug("Starting augmentation process")
    
    augmented_data = []
    rotation_angles = [45, 90, 135, 180, 225, 270, 315]
    dilation_scales = [(1.0, 1.5), (1.5, 1.0)]
    
    # Original image
    augmented_data.append((image.copy(), mask.copy(), "original"))
    
    # Rotations with dilations
    for angle in rotation_angles:
        if logger:
            logger.debug(f"Applying rotation: {angle} degrees")
        
        rotated_img, rotated_mask = rotate_image_and_mask(image, mask, angle)
        augmented_data.append((rotated_img, rotated_mask, f"rot_{angle}"))
        
        # Apply dilations to rotated images
        for scale_x, scale_y in dilation_scales:
            if logger:
                logger.debug(f"Dilation [{scale_x}, {scale_y}] on rot {angle}")
            
            dilated_img, dilated_mask = dilate_image_and_mask(
                rotated_img, rotated_mask, scale_x, scale_y
            )
            augmented_data.append((
                dilated_img, dilated_mask,
                f"rot_{angle}_dil_{scale_x}x{scale_y}"
            ))
    
    # Apply dilations to original
    for scale_x, scale_y in dilation_scales:
        if logger:
            logger.debug(f"Dilation [{scale_x}, {scale_y}] on original")
        
        dilated_img, dilated_mask = dilate_image_and_mask(
            image, mask, scale_x, scale_y
        )
        augmented_data.append((
            dilated_img, dilated_mask,
            f"dil_{scale_x}x{scale_y}"
        ))
    
    if logger:
        logger.debug(f"Augmentation complete: {len(augmented_data)} variations")
    
    return augmented_data
