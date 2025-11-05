from .mask_cnn import MaskCNN
from .segmentation_model import SegmentationModel
from .combined_model import CombinedModel


def get_model(model_type='combined', use_mask_cnn=True, pretrained=True):
    """
    Factory function to create models
    
    Args:
        model_type: 'mask_cnn', 'segmentation', or 'combined'
        use_mask_cnn: Whether to use mask CNN in combined model
        pretrained: Whether to use pretrained weights
    
    Returns:
        Model instance
    """
    if model_type == 'mask_cnn':
        model = MaskCNN(pretrained=pretrained)
    elif model_type == 'segmentation':
        model = SegmentationModel()
    elif model_type == 'combined':
        model = CombinedModel(
            use_mask_cnn=use_mask_cnn,
            mask_cnn_pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Save model architecture
    arch_str = str(model)
    with open('model_arch.log', 'w') as f:
        f.write(arch_str)
    
    return model
