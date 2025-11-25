"""
Segformer-based transfer learning model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class SegformerB0(nn.Module):
    """
    Segformer-based transfer learning model for binary segmentation
    Uses pretrained nvidia/segformer-b0-finetuned-ade-512-512
    """
    
    def __init__(self):
        super(SegformerB0, self).__init__()
        
        self.resizer = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
        # Load pretrained Segformer model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=1,  # Binary segmentation
            ignore_mismatched_sizes=True
        )
        
        # Optionally freeze encoder layers for fine-tuning
        # for param in self.model.segformer.encoder.parameters():
        #     param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor (B, 3, H, W)
        Returns:
            Output logits (B, 1, H, W)
        """
        x = self.resizer(x)
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
