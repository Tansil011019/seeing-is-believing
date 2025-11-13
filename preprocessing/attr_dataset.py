"""
PyTorch Dataset for multilabel attribute classification
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
from .attr_preprocessing import get_multilabel_for_image


class AttributeDataset(Dataset):
    """
    PyTorch Dataset for ISIC Task 2 attribute detection (multilabel classification)
    """
    
    def __init__(
        self,
        image_folder: str,
        gt_folder: str,
        image_size: Tuple[int, int] = (256, 256),
        normalize: bool = True
    ):
        """
        Initialize attribute dataset
        
        Args:
            image_folder: Path to images
            gt_folder: Path to attribute ground truth masks
            image_size: Target size for resizing
            normalize: Whether to normalize images with ImageNet stats
        """
        self.image_folder = image_folder
        self.gt_folder = gt_folder
        self.image_size = image_size
        self.normalize = normalize
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(image_folder)
            if f.endswith('.png') or f.endswith('.jpg')
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_id = os.path.splitext(image_file)[0]
        
        # Load and process image
        image_path = os.path.join(self.image_folder, image_file)
        image = self._load_image(image_path)
        
        # Get multilabel vector for this image
        labels = get_multilabel_for_image(image_id, self.gt_folder)
        
        # Convert to tensors
        image = self._to_tensor(image)
        labels = torch.from_numpy(labels).float()
        
        # Normalize if requested
        if self.normalize:
            image = self._normalize_image(image)
        
        return {
            'image': image,
            'labels': labels,
            'image_id': image_id
        }
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and resize image"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, self.image_size)
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor (C, H, W)"""
        return torch.from_numpy(image).permute(2, 0, 1).float()
    
    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image with ImageNet statistics"""
        image = image / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (image - mean) / std
