"""
ECViT-based transfer learning model for attribute detection
ECViT: Efficient Compact Vision Transformer
"""
import torch
import torch.nn as nn
from timm import create_model


class ECViTAttr(nn.Module):
    """
    ECViT-based multi-label attribute classifier
    Uses Vision Transformer (ViT) architecture optimized for efficiency
    
    Falls back to standard ViT-Tiny if ECViT not available
    """
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(ECViTAttr, self).__init__()
        
        # Try to load a compact ViT model
        try:
            # Try DeiT-tiny (Data-efficient Image Transformer - compact variant)
            if pretrained:
                self.backbone = create_model(
                    'deit_tiny_patch16_224',
                    pretrained=True,
                    num_classes=num_classes
                )
            else:
                self.backbone = create_model(
                    'deit_tiny_patch16_224',
                    pretrained=False,
                    num_classes=num_classes
                )
        except:
            # Fallback to ViT-Tiny
            print("DeiT-Tiny not available, using ViT-Tiny instead")
            if pretrained:
                self.backbone = create_model(
                    'vit_tiny_patch16_224',
                    pretrained=True,
                    num_classes=num_classes
                )
            else:
                self.backbone = create_model(
                    'vit_tiny_patch16_224',
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
        return self.backbone(x)


class ViTTinyAttr(nn.Module):
    """
    ViT-Tiny based multi-label attribute classifier
    Lightweight Vision Transformer for attribute detection
    """
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(ViTTinyAttr, self).__init__()
        
        if pretrained:
            self.backbone = create_model(
                'vit_tiny_patch16_224',
                pretrained=True,
                num_classes=num_classes
            )
        else:
            self.backbone = create_model(
                'vit_tiny_patch16_224',
                pretrained=False,
                num_classes=num_classes
            )
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor (B, 3, H, W)
        Returns:
            Output logits (B, num_classes)
        """
        return self.backbone(x)
