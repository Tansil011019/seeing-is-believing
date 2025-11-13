"""
Attribute classification models using transfer learning
Adapts pretrained models for multilabel classification
"""
import torch
import torch.nn as nn


class AttributeTransferModel(nn.Module):
    """
    Adapter for transfer learning models to multilabel classification
    Replaces segmentation head with global pooling + classifier
    """
    
    def __init__(self, backbone_name: str = 'segformer', 
                 num_classes: int = 5, freeze_encoder: bool = False):
        super().__init__()
        self.backbone_name = backbone_name
        
        # Import and initialize backbone
        from seg_models.transfer_models import get_transfer_model
        self.backbone = get_transfer_model(backbone_name, freeze_encoder=freeze_encoder)
        
        # Replace segmentation head with classification head
        self._adapt_to_classification(num_classes)
    
    def _adapt_to_classification(self, num_classes: int):
        """Replace segmentation head with global pooling + classifier"""
        # Get feature dimension from backbone (fallback to 512)
        if hasattr(self.backbone, 'decode_head'):
            in_features = self.backbone.decode_head.in_channels[-1]
        else:
            in_features = 512
        
        # Global average pooling + classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through backbone and classifier
        Returns logits for multilabel classification
        """
        # Get features from backbone (before final layer)
        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)
            # Handle dict or tensor output
            if isinstance(features, dict):
                features = features.get('out', features.get('features'))
        
        # Global pooling and classification
        pooled = self.pool(features)
        logits = self.classifier(pooled)
        return logits
