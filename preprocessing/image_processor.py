"""
Single image processing for parallel execution
"""
import os
from typing import Tuple
from .image_io import (
    load_image, load_mask, save_image, save_mask, find_mask_file
)
from .augmentation_pipeline import augment_image_and_mask


def process_single_image(args_tuple) -> Tuple[bool, int, str]:
    """
    Process a single image with augmentation
    
    Expects to be called under run_parrallel_processing function or something idk.
    
    Returns:
        (success, num_augmented, error_message)
    """
    (image_file, image_folder, mask_folder,
     output_image_folder, output_mask_folder,
     apply_augmentation, output_size) = args_tuple
    
    try:
        image_id = os.path.splitext(image_file)[0]
        image_path = os.path.join(image_folder, image_file)
        
        # Find mask file
        mask_file = find_mask_file(image_id, mask_folder)
        if mask_file is None:
            return (False, 0, f"Mask not found: {image_file}")
        
        mask_path = os.path.join(mask_folder, mask_file)
        
        # Load image and mask
        image = load_image(image_path)
        if image is None:
            return (False, 0, f"Failed to load image: {image_path}")
        
        mask = load_mask(mask_path)
        if mask is None:
            return (False, 0, f"Failed to load mask: {mask_path}")
        
        # Apply augmentation
        if apply_augmentation:
            augmented_samples = augment_image_and_mask(image, mask)
        else:
            augmented_samples = [(image, mask, "original")]
        
        # Save augmented samples
        for aug_img, aug_mask, description in augmented_samples:
            output_image_name = f"{image_id}_{description}.png"
            output_mask_name = f"{image_id}_{description}.png"
            
            output_image_path = os.path.join(
                output_image_folder, output_image_name
            )
            output_mask_path = os.path.join(
                output_mask_folder, output_mask_name
            )
            
            save_image(aug_img, output_image_path, size=output_size)
            save_mask(aug_mask, output_mask_path, size=output_size)
        
        return (True, len(augmented_samples), None)
    
    except Exception as e:
        return (False, 0, f"Error processing {image_file}: {str(e)}")
