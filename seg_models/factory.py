from .mask_cnn import MaskCNN
from .segmentation_model import SegmentationModel
from .combined_model import CombinedModel
from .deeplab_v3p_torch import deeplabv3plus
from .fat_net import FAT_Net


# Model registry mapping model names to constructors
MODEL_REGISTRY = {
    'combined': lambda pretrained=True, use_mask_cnn=True: CombinedModel(
        use_mask_cnn=use_mask_cnn,
        mask_cnn_pretrained=pretrained
    ),
    'mask_cnn': lambda pretrained=True: MaskCNN(pretrained=pretrained),
    'segmentation': lambda pretrained=True: SegmentationModel(),
    'deeplabv3plus': lambda pretrained=False: deeplabv3plus(num_classes=1, pretrained=pretrained),
    'fat_net': lambda pretrained=False: FAT_Net(n_channels=3, n_classes=1),
}


def get_model(model_type='combined', use_mask_cnn=True, pretrained=True):
    """
    Factory function to create models
    
    Args:
        model_type: Model name from MODEL_REGISTRY
        use_mask_cnn: Whether to use mask CNN in combined model
        pretrained: Whether to use pretrained weights
    
    Returns:
        Model instance
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available models: {list(MODEL_REGISTRY.keys())}")
    
    # Special handling for combined model
    if model_type == 'combined':
        model = MODEL_REGISTRY[model_type](pretrained=pretrained, use_mask_cnn=use_mask_cnn)
    else:
        model = MODEL_REGISTRY[model_type](pretrained=pretrained)

    # Save model architecture
    arch_str = str(model)
    with open('model_arch.log', 'w') as f:
        f.write(arch_str)
    
    return model


def get_available_models():
    """Return list of available model names"""
    return list(MODEL_REGISTRY.keys())
