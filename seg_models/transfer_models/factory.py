"""
Factory functions for creating transfer learning models
"""
from .segformer_model import SegformerTransferModel
from .mit_model import MITTransferModel
from .beit_model import BEiTTransferModel
from .medsam_model import MedSAM2TransferModel
from .adaptive_model import AdaptiveTransferModel


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
        'segformer': lambda: SegformerTransferModel(
            model_name="nvidia/segformer-b0-finetuned-ade-512-512",
            freeze_encoder=freeze_encoder
        ),
        'mit_b1': lambda: MITTransferModel(
            model_name="nvidia/segformer-b1-finetuned-ade-512-512",
            freeze_encoder=freeze_encoder
        ),
        'beit': lambda: BEiTTransferModel(
            model_name="microsoft/beit-base-finetuned-ade-640-640",
            freeze_encoder=freeze_encoder
        ),
        'medsam2': lambda: MedSAM2TransferModel(
            freeze_encoder=freeze_encoder
        ),
        'adaptive_segformer': lambda: AdaptiveTransferModel(
            base_model_name="nvidia/segformer-b0-finetuned-ade-512-512",
            model_type="segformer",
            freeze_encoder=freeze_encoder
        ),
        'adaptive_beit': lambda: AdaptiveTransferModel(
            base_model_name="microsoft/beit-base-finetuned-ade-640-640",
            model_type="beit",
            freeze_encoder=freeze_encoder
        ),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]()


def get_available_transfer_models():
    """Return list of available transfer learning model names"""
    return [
        'segformer',
        'mit_b1', 
        'beit',
        'medsam2',
        'adaptive_segformer',
        'adaptive_beit'
    ]
