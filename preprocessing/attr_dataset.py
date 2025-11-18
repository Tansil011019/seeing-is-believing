"""
PyTorch Dataset for multilabel attribute classification
"""
import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Optional
from pathlib import Path


# Attribute labels
ATTRIBUTE_LABELS = [
    'globules',
    'milia_like_cyst',
    'negative_network',
    'pigment_network',
    'streaks'
]


class AttributeDataset(Dataset):
    """
    PyTorch Dataset for ISIC Task 2 attribute detection (multilabel classification)
    Now supports CSV-based labels for improved flexibility
    """
    
    def __init__(
        self,
        image_folder: str,
        csv_file: str,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True
    ):
        """
        Initialize attribute dataset
        
        Args:
            image_folder: Path to images
            csv_file: Path to CSV file containing labels
            image_size: Target size for resizing
            normalize: Whether to normalize images with ImageNet stats
        """
        self.image_folder = Path(image_folder)
        self.csv_file = Path(csv_file)
        self.image_size = image_size
        self.normalize = normalize
        self.num_classes = len(ATTRIBUTE_LABELS)
        
        # Load labels from CSV
        self.samples = self._load_labels()
    
    def _load_labels(self) -> List[Dict]:
        """Load labels from CSV file"""
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        
        samples = []
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row['image_id']
                labels = [int(row[label]) for label in ATTRIBUTE_LABELS]
                samples.append({
                    'image_id': img_id,
                    'labels': labels
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_id = sample['image_id']
        labels = sample['labels']
        
        # Load image
        img_path = self.image_folder / f"{img_id}.jpg"
        if not img_path.exists():
            # Try .png extension
            img_path = self.image_folder / f"{img_id}.png"
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Load and process image
        image = self._load_image(str(img_path))
        
        # Convert to tensors
        image = self._to_tensor(image)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        # Normalize if requested
        if self.normalize:
            image = self._normalize_image(image)
        
        return {
            'image': image,
            'labels': labels_tensor,
            'image_id': img_id
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
    
    def get_label_names(self) -> List[str]:
        """Get list of attribute label names"""
        return ATTRIBUTE_LABELS
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of positive labels"""
        distribution = {label: 0 for label in ATTRIBUTE_LABELS}
        
        for sample in self.samples:
            for i, label in enumerate(ATTRIBUTE_LABELS):
                if sample['labels'][i] == 1:
                    distribution[label] += 1
        
        return distribution
