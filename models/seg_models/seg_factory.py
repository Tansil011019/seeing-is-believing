"""
Factory functions for creating transfer learning models
"""

from models.seg_models.transfer_models.segformer_b0 import SegformerB0
from models.seg_models.transfer_models.segformer_b1 import SegformerB1

def get_transfer_model(model_name, freeze_encoder=False, **kwargs):
    """
    Factory function to create transfer learning models
    
    Args:
        model_name: Name of the model architecture
        freeze_encoder: Whether to freeze encoder weights
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Transfer learning model instance
    """
    models = {
        'segformer_b0': lambda: SegformerB0(),
        'segformer_b1': lambda: SegformerB1(),

    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]()


def get_available_transfer_models():
    """Return list of available transfer learning model names"""
    return [
        'segformer_b0',
        'segformer_b1', 
        'efficient_sam_custom',
    ]
