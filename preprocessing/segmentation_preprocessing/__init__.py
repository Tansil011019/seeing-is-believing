"""
Segmentation preprocessing module.

This module provides utilities for preprocessing data before segmentation tasks.
"""

from .augmentation import (
    rotate_image_and_mask,
    dilate_image_and_mask,
    augment_image_and_mask,
    AUGMENTATION_FACTOR,
    IMG_EXTENSIONS,
)

from .preprocessing import (
    process_single_image,
    preprocess_segmentation_dataset_parallel
)

__all__ = [
   'augment_image_and_mask',
   'preprocess_segmentation_dataset_parallel',
   'AUGMENTATION_FACTOR',
    'IMG_EXTENSIONS',
]