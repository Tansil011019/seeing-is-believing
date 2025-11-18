"""
Base model loader utility for adaptive transfer learning
"""
from transformers import SegformerForSemanticSegmentation, BeitForSemanticSegmentation


def load_base_model_with_freeze(base_model_name, model_type, freeze_encoder):
    """
    Load and configure base model with optional encoder freezing
    
    Args:
        base_model_name: HuggingFace model identifier
        model_type: Type of model ('segformer' or 'beit')
        freeze_encoder: Whether to freeze encoder parameters
    
    Returns:
        Configured model instance
    """
    if model_type == "segformer":
        model = SegformerForSemanticSegmentation.from_pretrained(
            base_model_name,
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        if freeze_encoder:
            for param in model.segformer.encoder.parameters():
                param.requires_grad = False
                
    elif model_type == "beit":
        model = BeitForSemanticSegmentation.from_pretrained(
            base_model_name,
            num_labels=1,
            ignore_mismatched_sizes=True
        )
        if freeze_encoder:
            for param in model.beit.encoder.parameters():
                param.requires_grad = False
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model
