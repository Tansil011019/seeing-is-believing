"""
Preprocessing modules for image augmentation and dataset handling
"""
from .segmentation_preprocessing.augmentation import (
    rotate_image_and_mask,
    dilate_image_and_mask,
)

from .attr_dataset import AttributeDataset

from .attr_preprocessing import (
    compute_attribute_label,
    get_multilabel_for_image,
    process_dataset_labels,
    get_label_statistics,
    ATTR_TYPES,
    FEAT_THRESHOLD
)

from . import segmentation_preprocessing

__all__ = [
    'rotate_image_and_mask',
    'dilate_image_and_mask',
    'augment_image_and_mask',
    'process_dataset_parallel',
    'SegmentationDataset',
    'AttributeDataset',
    'compute_attribute_label',
    'get_multilabel_for_image',
    'process_dataset_labels',
    'get_label_statistics',
    'ATTR_TYPES',
    'FEAT_THRESHOLD',
    'segmentation_preprocessing',
]
