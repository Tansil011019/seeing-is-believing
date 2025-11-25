"""
Single image processing for parallel execution
"""
import os
from typing import Tuple
from ..image_io import (
    load_image, load_mask, save_image, save_mask, find_mask_file
)

from .augmentation import augment_image_and_mask
import logging
from typing import List, Tuple
from tqdm import tqdm
from multiprocessing import Pool

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


def _run_parallel_processing(process_args, num_workers, logger):
    """Run processing with parallel workers or sequentially"""
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_image, process_args),
                total=len(process_args),
                desc="Processing images",
                disable=not (logger and logger.level <= logging.INFO)
            ))
    else:
        results = []
        for args in tqdm(
            process_args,
            desc="Processing images",
            disable=not (logger and logger.level <= logging.INFO)
        ):
            results.append(process_single_image(args))
    print(results[1:5])
    return results


def _collect_results(results: List[Tuple], logger) -> int:
    """Collect and log results from processing"""
    total_augmented = 0
    errors = []
    
    for success, num_aug, error_msg in results:
        if success:
            total_augmented += num_aug
        else:
            errors.append(error_msg)
            if logger:
                logger.warning(error_msg)
    
    if logger and errors:
        logger.warning(f"Encountered {len(errors)} errors")
    
    return total_augmented


def preprocess_segmentation_dataset_parallel(
    image_folder: str,
    mask_folder: str,
    output_image_folder: str,
    output_mask_folder: str,
    apply_augmentation: bool = False,
    num_workers: int = 4,
    logger=None,
    output_size: Tuple[int, int] = None
) -> int:
    """
    Preprocess entire dataset with parallel workers
    
    Args:
        image_folder: Path to input images
        mask_folder: Path to input masks
        output_image_folder: Path to save augmented images
        output_mask_folder: Path to save augmented masks
        apply_augmentation: Whether to apply augmentation
        num_workers: Number of parallel workers
        logger: Logger instance
        output_size: Optional (width, height) to resize outputs to
    """
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)
    
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.endswith('.png') or f.endswith('.jpg')
    ])
    
    mask_files = sorted([
        f for f in os.listdir(mask_folder)
        if f.endswith('.png') or f.endswith('.jpg')
    ])
    
    if logger:
        logger.info(f"Found {len(image_files)} images to process")
        if output_size:
            logger.info(f"Output images will be resized to {output_size}")
    
    process_args = [
        (img, image_folder, mask_folder,
         output_image_folder, output_mask_folder,
         apply_augmentation, output_size)
        for img in image_files
    ]
    
    results = _run_parallel_processing(process_args, num_workers, logger)
    return _collect_results(results, logger)
