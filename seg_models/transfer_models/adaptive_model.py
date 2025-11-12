"""
Adaptive transfer learning model with domain adaptation layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_loader import load_base_model_with_freeze


class AdaptationLayers(nn.Module):
    """Domain-specific adaptation layers for refining pretrained features"""
    
    def __init__(self):
        super(AdaptationLayers, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """Apply adaptation layers to input features"""
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


class AdaptiveTransferModel(nn.Module):
    """
    Adaptive transfer learning model with feature adaptation layers
    Adds learnable adaptation layers on top of frozen pretrained features
    """
    
    def __init__(self, base_model_name="nvidia/segformer-b0-finetuned-ade-512-512", 
                 model_type="segformer", freeze_encoder=True):
        super(AdaptiveTransferModel, self).__init__()
        
        self.model_type = model_type
        self.base_model = load_base_model_with_freeze(base_model_name, model_type, freeze_encoder)
        self.adaptation_layers = AdaptationLayers()
    
    def forward(self, x):
        """
        Forward pass with adaptation layers
        Args:
            x: Input tensor (B, 3, H, W)
        Returns:
            Output logits (B, 1, H, W)
        """
        # Get base model features
        outputs = self.base_model(pixel_values=x)
        logits = outputs.logits
        
        # Upsample to input resolution
        logits = F.interpolate(
            logits,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Apply adaptation layers and add residual connection
        adapted = self.adaptation_layers(logits)
        output = logits + adapted
        
        return output
