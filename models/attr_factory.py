"""
Model factory for attribute detection models
Centralized model creation and management
"""
from typing import Dict, List
import torch.nn as nn

# Import all attribute detection models
from .resnet18_attr import ResNet18Attr
from .resnet34_attr import ResNet34Attr
from .resnet50_attr import ResNet50Attr
from .efficientvim_attr import EfficientViMAttr, EfficientNetB0Attr
from .ecvit_attr import ECViTAttr, ViTTinyAttr


# Model registry
ATTR_MODEL_REGISTRY: Dict[str, type] = {
    # ResNet variants
    'resnet18': ResNet18Attr,
    'resnet34': ResNet34Attr,
    'resnet50': ResNet50Attr,
    
    # Efficient models
    'efficientvim': EfficientViMAttr,
    'efficientnet_b0': EfficientNetB0Attr,
    
    # Vision Transformer variants
    'ecvit': ECViTAttr,
    'vit_tiny': ViTTinyAttr,
}


def get_attr_model(model_name: str, num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """
    Get attribute detection model by name
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes (default: 5 for ISIC Task 2)
        pretrained: Whether to use pretrained weights
    
    Returns:
        Model instance
    
    Raises:
        ValueError: If model name is not found in registry
    """
    model_name = model_name.lower()
    
    if model_name not in ATTR_MODEL_REGISTRY:
        available_models = ', '.join(ATTR_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available models: {available_models}"
        )
    
    model_class = ATTR_MODEL_REGISTRY[model_name]
    model = model_class(num_classes=num_classes, pretrained=pretrained)
    
    return model


def get_available_attr_models() -> List[str]:
    """
    Get list of available attribute detection models
    
    Returns:
        List of model names
    """
    return sorted(list(ATTR_MODEL_REGISTRY.keys()))


def print_model_info():
    """Print information about available models"""
    print("="*60)
    print("Available Attribute Detection Models")
    print("="*60)
    
    print("\nResNet Variants:")
    print("  - resnet18: ResNet-18 (lightweight)")
    print("  - resnet34: ResNet-34 (balanced)")
    print("  - resnet50: ResNet-50 (high capacity)")
    
    print("\nEfficient Models:")
    print("  - efficientvim: EfficientViM (Mamba-based)")
    print("  - efficientnet_b0: EfficientNet-B0 (lightweight CNN)")
    
    print("\nVision Transformer Variants:")
    print("  - ecvit: Efficient Compact ViT (DeiT-Tiny)")
    print("  - vit_tiny: ViT-Tiny")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Print model information
    print_model_info()
    
    # Test model creation
    print("\nTesting model creation...")
    for model_name in get_available_attr_models():
        try:
            model = get_attr_model(model_name, pretrained=False)
            print(f"✓ {model_name}: Successfully created")
        except Exception as e:
            print(f"✗ {model_name}: Failed - {e}")
