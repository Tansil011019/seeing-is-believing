"""
ResNet18-based transfer learning model for attribute detection
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Attr(nn.Module):
    """
    ResNet18-based multi-label attribute classifier
    Uses pretrained ResNet18 as backbone
    """
    
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super(ResNet18Attr, self).__init__()
        
        # Load pretrained ResNet18
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.backbone = resnet18(weights=weights)
        else:
            self.backbone = resnet18(weights=None)
        
        # Get number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final layer with a new classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
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
