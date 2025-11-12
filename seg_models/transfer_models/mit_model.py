"""
MiT (Mix Transformer) based transfer learning model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class MITTransferModel(nn.Module):
    """
    MiT (Mix Transformer) based transfer learning model
    Uses pretrained keras/mit_b1_ade20k_512
    Note: This model may require special handling for keras weights
    """
    
    def __init__(self, model_name="nvidia/segformer-b1-finetuned-ade-512-512", freeze_encoder=False):
        super(MITTransferModel, self).__init__()
        
        # Using Segformer B1 as MIT-B1 architecture
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        
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
        logits = outputs.logits
        
        # Upsample to input resolution
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return logits
