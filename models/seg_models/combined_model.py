"""
Combined model with Mask CNN and Segmentation Model
"""
import torch.nn as nn
from .mask_cnn import MaskCNN
from .segmentation_model import SegmentationModel


class CombinedModel(nn.Module):
    """Combined model with optional Mask CNN preprocessing"""
    
    def __init__(self, use_mask_cnn=True, mask_cnn_pretrained=True):
        super(CombinedModel, self).__init__()
        
        self.use_mask_cnn = use_mask_cnn
        
        if use_mask_cnn:
            self.mask_cnn = MaskCNN(pretrained=mask_cnn_pretrained)
        
        self.seg_model = SegmentationModel()
    
    def forward(self, x, return_bbox=False):
        """
        Args:
            x: Input image (B, 3, H, W)
            return_bbox: Whether to return bounding box predictions
        
        Returns:
            Segmentation mask (B, 1, H, W) and optionally bbox (B, 4)
        """
        if self.use_mask_cnn:
            bbox = self.mask_cnn(x)
            cropped_x = self.mask_cnn.crop_image(x, bbox)
            seg_mask = self.seg_model(cropped_x)
            
            if return_bbox:
                return seg_mask, bbox
            return seg_mask
        else:
            seg_mask = self.seg_model(x)
            return seg_mask
