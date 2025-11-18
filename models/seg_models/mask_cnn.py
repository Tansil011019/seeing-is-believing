"""
Mask CNN for bounding box prediction
Based on: https://arxiv.org/pdf/1703.06870
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MaskCNN(nn.Module):
    """
    Mask CNN for cropping/masking image to rectangular area
    Predicts bounding box coordinates (x_min, y_min, x_max, y_max)
    """
    
    def __init__(self, backbone='resnet18', pretrained=True):
        super(MaskCNN, self).__init__()
        
        # Load pretrained ResNet backbone
        resnet = self._get_resnet(backbone, pretrained)
        
        # Remove the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Get feature dimensions
        feature_dim = 512 if backbone in ['resnet18', 'resnet34'] else 2048
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Bounding box regression head
        self.bbox_head = self._build_bbox_head(feature_dim)
    
    def _get_resnet(self, backbone, pretrained):
        """Get ResNet model by name"""
        if backbone == 'resnet18':
            return models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            return models.resnet34(pretrained=pretrained)
        else:
            return models.resnet50(pretrained=pretrained)
    
    def _build_bbox_head(self, feature_dim):
        """Build bounding box regression head"""
        return nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 4),  # (x_min, y_min, x_max, y_max)
            nn.Sigmoid()  # Normalize to [0, 1]
        )
    
    def forward(self, x):
        """
        Args:
            x: Input image (B, 3, H, W)
        
        Returns:
            Bounding box coordinates (B, 4) in normalized [0, 1] range
        """
        features = self.features(x)  # (B, C, H', W')
        pooled = self.global_pool(features)  # (B, C, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C)
        bbox = self.bbox_head(pooled)  # (B, 4)
        return bbox
    
    def crop_image(self, image, bbox):
        """
        Crop image according to predicted bounding box
        
        Args:
            image: Input image (B, 3, H, W)
            bbox: Bounding box (B, 4) in normalized [0, 1] range
        
        Returns:
            Cropped image (B, 3, H, W) - resized to original size
        """
        B, C, H, W = image.shape
        cropped_images = []
        
        for i in range(B):
            coords = self._get_bbox_coords(bbox[i], H, W)
            cropped = self._crop_single_image(image[i:i+1], coords, H, W)
            cropped_images.append(cropped)
        
        return torch.cat(cropped_images, dim=0)
    
    def _get_bbox_coords(self, bbox, H, W):
        """Extract and validate bbox coordinates"""
        x_min = int(bbox[0].item() * W)
        y_min = int(bbox[1].item() * H)
        x_max = int(bbox[2].item() * W)
        y_max = int(bbox[3].item() * H)
        
        # Ensure valid coordinates
        x_min = max(0, min(x_min, W - 1))
        y_min = max(0, min(y_min, H - 1))
        x_max = max(x_min + 1, min(x_max, W))
        y_max = max(y_min + 1, min(y_max, H))
        
        return x_min, y_min, x_max, y_max
    
    def _crop_single_image(self, image, coords, H, W):
        """Crop and resize single image"""
        x_min, y_min, x_max, y_max = coords
        cropped = image[:, :, y_min:y_max, x_min:x_max]
        return F.interpolate(
            cropped, size=(H, W),
            mode='bilinear', align_corners=False
        )
