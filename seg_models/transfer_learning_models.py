"""
Transfer learning models for segmentation using pretrained HuggingFace models

This module provides backward compatibility.
All models are now organized in the transfer_models subpackage.
"""
# Re-export everything from the new structure
from .transfer_models import (
    SegformerTransferModel,
    MITTransferModel,
    BEiTTransferModel,
    MedSAM2TransferModel,
    AdaptiveTransferModel,
    get_transfer_model,
    get_available_transfer_models
)

__all__ = [
    'SegformerTransferModel',
    'MITTransferModel',
    'BEiTTransferModel',
    'MedSAM2TransferModel',
    'AdaptiveTransferModel',
    'get_transfer_model',
    'get_available_transfer_models'
]
