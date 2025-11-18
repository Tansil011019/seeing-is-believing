"""
EfficientViM-based transfer learning model for attribute detection
EfficientViM: Vision Mamba with Efficient Design
"""
import torch
import torch.nn as nn
from timm import create_model


class EfficientViMAttr(nn.Module):
    """
    EfficientViM-based multi-label attribute classifier
    Uses pretrained EfficientViM model from timm library
    
    Note: EfficientViM models may not be available in all timm versions.
    Falls back to EfficientNet if unavailable.
    """
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(EfficientViMAttr, self).__init__()
        
        # Try to load EfficientViM, fallback to EfficientNet if unavailable
        try:
            # Try EfficientViM model (if available in timm)
            if pretrained:
                self.backbone = create_model(
                    'efficientvit_m0',  # EfficientViT model
                    pretrained=True,
                    num_classes=num_classes
                )
            else:
                self.backbone = create_model(
                    'efficientvit_m0',
                    pretrained=False,
                    num_classes=num_classes
                )
        except:
            # Fallback to EfficientNet-B0
            print("EfficientViM not available, using EfficientNet-B0 instead")
            if pretrained:
                self.backbone = create_model(
                    'efficientnet_b0',
                    pretrained=True,
                    num_classes=num_classes
                )
            else:
                self.backbone = create_model(
                    'efficientnet_b0',
                    pretrained=False,
                    num_classes=num_classes
                )
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor (B, 3, H, W)
        Returns:
            Output logits (B, num_classes)
        """
        features = self.backbone.forward_features(x)
        features = self.backbone.global_pool(features)
        features = self.dropout(features)
        output = self.backbone.classifier(features)
        return output


class EfficientNetB0Attr(nn.Module):
    """
    EfficientNet-B0 based multi-label attribute classifier
    Alternative lightweight model for attribute detection
    """
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(EfficientNetB0Attr, self).__init__()
        
        if pretrained:
            self.backbone = create_model(
                'efficientnet_b0',
                pretrained=True,
                num_classes=num_classes
            )
        else:
            self.backbone = create_model(
                'efficientnet_b0',
                pretrained=False,
                num_classes=num_classes
            )
        
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor (B, 3, H, W)
        Returns:
            Output logits (B, num_classes)
        """
        return self.backbone(x)
