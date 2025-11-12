"""
BEiT-based transfer learning model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BeitForSemanticSegmentation


class BEiTTransferModel(nn.Module):
    """
    BEiT (BERT pre-training of Image Transformers) based transfer learning model
    Uses pretrained microsoft/beit-base-finetuned-ade-640-640
    """
    
    def __init__(self, model_name="microsoft/beit-base-finetuned-ade-640-640", freeze_encoder=False):
        super(BEiTTransferModel, self).__init__()
        
        # Load pretrained BEiT model
        self.model = BeitForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        
        if freeze_encoder:
            for param in self.model.beit.encoder.parameters():
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
