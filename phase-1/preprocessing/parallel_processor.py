"""
Parallel dataset processing
"""
import os
import logging
from typing import List, Tuple
from tqdm import tqdm
from multiprocessing import Pool
from .image_processor import process_single_image


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


def process_dataset_parallel(
    image_folder: str,
    mask_folder: str,
    output_image_folder: str,
    output_mask_folder: str,
    apply_augmentation: bool = True,
    num_workers: int = 4,
    logger=None
) -> int:
    """
    Process entire dataset with parallel workers
    """
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)
    
    image_files = sorted([
        f for f in os.listdir(image_folder)
        if f.endswith('.png') or f.endswith('.jpg')
    ])
    
    if logger:
        logger.info(f"Found {len(image_files)} images to process")
    
    process_args = [
        (img, image_folder, mask_folder,
         output_image_folder, output_mask_folder,
         apply_augmentation)
        for img in image_files
    ]
    
    results = _run_parallel_processing(process_args, num_workers, logger)
    return _collect_results(results, logger)
