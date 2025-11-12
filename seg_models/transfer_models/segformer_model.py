"""
Segformer-based transfer learning model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class SegformerTransferModel(nn.Module):
    """
    Segformer-based transfer learning model for binary segmentation
    Uses pretrained nvidia/segformer-b0-finetuned-ade-512-512
    """
    
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512", freeze_encoder=False):
        super(SegformerTransferModel, self).__init__()
        
        # Load pretrained Segformer model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=1,  # Binary segmentation
            ignore_mismatched_sizes=True
        )
        
        # Optionally freeze encoder layers for fine-tuning
        if freeze_encoder:
            for param in self.model.segformer.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor (B, 3, H, W)
        Returns:
            Output logits (B, 1, H, W)
        """
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # (B, 1, H/4, W/4)
        
        # Upsample to input resolution
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return logits
