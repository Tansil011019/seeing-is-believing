"""
PyTorch Dataset for segmentation
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple


class SegmentationDataset(Dataset):
    """PyTorch Dataset for loading segmentation data"""
    
    def __init__(
        self,
        image_folder: str,
        mask_folder: str,
        image_size: Tuple[int, int] = (256, 256),
        normalize: bool = True
    ):
        """
        Initialize dataset
        
        Args:
            image_folder: Path to images
            mask_folder: Path to masks
            image_size: Target size for resizing
            normalize: Whether to normalize images
        """
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_size = image_size
        self.normalize = normalize
        
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
        return cv2.resize(image, self.image_size)
    
    def _find_mask_path(self, image_id: str) -> str:
        """Find mask path for given image ID"""
        for ext in ['.png', '.jpg', '_segmentation.png']:
            mask_file = image_id + ext
            mask_path = os.path.join(self.mask_folder, mask_file)
            if os.path.exists(mask_path):
                return mask_path
        
        # Try base ID without suffix
        base_id = image_id.split('_')[0]
        for ext in ['.png', '.jpg', '_segmentation.png']:
            mask_file = base_id + ext
            mask_path = os.path.join(self.mask_folder, mask_file)
            if os.path.exists(mask_path):
                return mask_path
        
        return None
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load and process mask"""
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(
                mask, self.image_size,
                interpolation=cv2.INTER_NEAREST
            )
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        else:
            mask = np.zeros(self.image_size, dtype=np.uint8)
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
