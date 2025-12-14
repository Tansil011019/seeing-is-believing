"""
PyTorch Dataset for segmentation
"""
import os
import cv2
import numpy as np
from pyparsing import Path
import torch
from torch.utils.data import Dataset
from typing import Tuple
from preprocessing.segmentation_preprocessing import (
    preprocess_segmentation_dataset_parallel,
    AUGMENTATION_FACTOR,
    IMG_EXTENSIONS
)
import logging

logger = logging.getLogger(__name__)

class SegmentationDataset(Dataset):
    """PyTorch Dataset for loading segmentation data"""
    
    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        preprocessed_img_dir: str = "",
        preprocessed_mask_dir: str = "",
        num_workers: int = 4,
        target_image_size: Tuple[int, int] = (512, 512),
        normalize: bool = True,
        do_preprocess: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            image_folder: Path to images
            mask_dir: Path to masks
            image_size: Target size for resizing
            normalize: Whether to normalize images
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.target_image_size = target_image_size
        self.normalize = normalize
        self.num_workers = num_workers
        
        self.preprocessed_image_dir = preprocessed_img_dir
        self.preprocessed_mask_dir = preprocessed_mask_dir
        
        # If preprocessing is requested, run it
        if do_preprocess:
            self._preprocess()
        # If no preprocessed dirs provided, use original dirs
        else :
            self.preprocessed_image_dir = img_dir
            self.preprocessed_mask_dir = mask_dir
            
        self.image_files = sorted([
            f for f in os.listdir(self.preprocessed_image_dir)
            if any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS)
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_id = os.path.splitext(image_file)[0]
        
        # Load and process image
        image_path = os.path.join(self.img_dir, image_file)
        image = self._load_image(image_path)
        
        # Load and process mask
        mask_path = self._find_mask_path(image_id)
        mask = self._load_mask(mask_path)
        
        # Convert to tensors
        image = self._to_tensor(image)
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        # Normalize if requested
        if self.normalize:
            image = self._normalize_image(image)
        
        return {
            'image': image,
            'mask': mask,
            'image_id': image_id
        }
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and resize image"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, self.target_image_size, interpolation=cv2.INTER_AREA)
    
    def _find_mask_path(self, image_id: str) -> str:
        """Find mask path for given image ID"""
        for ext in ['.png', '.jpg', '_segmentation.png']:
            mask_file = image_id + ext
            mask_path = os.path.join(self.mask_dir, mask_file)
            if os.path.exists(mask_path):
                return mask_path
        
        # Try base ID without suffix
        base_id = image_id.split('_')[0]
        for ext in ['.png', '.jpg', '_segmentation.png']:
            mask_file = base_id + ext
            mask_path = os.path.join(self.mask_dir, mask_file)
            if os.path.exists(mask_path):
                return mask_path
        
        return None
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load and process mask"""
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(
                mask, self.target_image_size,
                interpolation=cv2.INTER_NEAREST
            )
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        else:
            mask = np.zeros(self.target_image_size, dtype=np.uint8)
        return mask
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor"""
        return torch.from_numpy(image).permute(2, 0, 1).float()
    
    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image with ImageNet stats"""
        image = image / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (image - mean) / std
    
    def _preprocess(self):
        """Check for existing preprocessed data and run preprocessing if needed"""
        
        if self._check_augmented_data_exists():
            return
        
        preprocess_segmentation_dataset_parallel(
            str(self.img_dir),
            str(self.mask_dir),
            str(self.preprocessed_image_dir),
            str(self.preprocessed_mask_dir),
            True,
            self.num_workers,
            output_size=(512, 512)
        )
        
        
    def _check_augmented_data_exists(self) -> bool:
        """Check if augmented data exists with correct number of files"""
        
        aug_img_path = Path(self.preprocessed_image_dir)
        aug_mask_path = Path(self.preprocessed_mask_dir)
        img_path = Path(self.img_dir)
        
        if not aug_img_path.exists() or not aug_mask_path.exists():
            logger.warning("Augmented data/mask folder not found.")
            return False
        
        try:
            orig_img_count = sum(1 for f in img_path.iterdir()
                               if f.suffix.lower() in IMG_EXTENSIONS)
        except FileNotFoundError:
            logger.error(f"Original image folder not found: {img_path}")
            return False
        
        if orig_img_count == 0:
            logger.warning(f"No images found in {img_path}.")
            return False
        
        aug_img_count = sum(1 for f in aug_img_path.iterdir()
                           if f.suffix.lower() in IMG_EXTENSIONS)
        aug_mask_count = sum(1 for f in aug_mask_path.iterdir()
                            if f.suffix.lower() in IMG_EXTENSIONS)
        
        expected_count = orig_img_count * AUGMENTATION_FACTOR
        
        if aug_img_count >= expected_count and aug_mask_count >= expected_count:
            logger.info(f"Found {aug_img_count} aug images and {aug_mask_count} aug masks.")
            logger.info(f"(Expected >= {expected_count} based on {orig_img_count} originals)")
            return True
        
        logger.info(f"Found {aug_img_count} aug images, {aug_mask_count} aug masks.")
        logger.info(f"Expected {expected_count}. Preprocessing will run.")
        return False