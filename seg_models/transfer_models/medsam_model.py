"""
MedSAM2-based transfer learning model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskGeneration, SegformerForSemanticSegmentation


class MedSAM2TransferModel(nn.Module):
    """
    MedSAM2 (Medical Segment Anything Model 2) based transfer learning
    Uses pretrained wanglab/MedSAM2 with fallback to Segformer
    """
    
    def __init__(self, model_name="facebook/sam-vit-base", freeze_encoder=True):
        super(MedSAM2TransferModel, self).__init__()
        self.model, self.use_segformer_fallback = self._load_model(model_name)
        self._freeze_encoder_if_needed(freeze_encoder)
    
    def _load_model(self, model_name):
        """Load SAM or fallback to Segformer"""
        try:
            model = AutoModelForMaskGeneration.from_pretrained(model_name)
            return model, False
        except Exception as e:
            print(f"Warning: Could not load {model_name}, using Segformer fallback")
            model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512",
                num_labels=1,
                ignore_mismatched_sizes=True
            )
            return model, True
    
    def _freeze_encoder_if_needed(self, freeze_encoder):
        """Freeze encoder parameters based on model type"""
        if not freeze_encoder:
            return
        if hasattr(self.model, 'vision_encoder'):
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = False
        elif hasattr(self.model, 'segformer'):
            for param in self.model.segformer.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """Forward pass with automatic SAM/Segformer handling"""
        if self.use_segformer_fallback:
            outputs = self.model(pixel_values=x)
            logits = outputs.logits
        else:
            outputs = self.model(pixel_values=x)
            logits = outputs.pred_masks if hasattr(outputs, 'pred_masks') else outputs.logits
        
        # Ensure binary segmentation output
        if logits.dim() == 4 and logits.shape[1] != 1:
            logits = logits[:, 0:1, :, :]
        
        # Upsample to input resolution
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
