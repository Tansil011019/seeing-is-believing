"""
Utils package for skin disease classification project.
"""

# Import evaluation utilities
from .evaluation import (
    calculate_heatmap_iou,
    display_heatmap_intersection,
    display_heatmap_union
)

# Import training utilities
from .training import (
    train_and_evaluate_model,
    preprocess_images,
    plot_training_history,
    plot_confusion_matrix
)

# Import model instantiation functions
try:
    from .models_himvit import instantiate_model_hi_mvit, instantiate_model_himvit
    from .models_medkan import instantiate_model_medkan
    from .models_standard import (
        instantiate_model_vgg16,
        instantiate_model_resnet50,
        instantiate_model_resnet101,
        instantiate_model_resnet152,
        instantiate_model_resnet,
        get_best_resnet
    )
except ImportError as e:
    print(f"Warning: Some model imports failed: {e}")
    print("Make sure TensorFlow is installed: pip install tensorflow")

__all__ = [
    # Evaluation functions
    'calculate_heatmap_iou',
    'display_heatmap_intersection', 
    'display_heatmap_union',
    
    # Training functions
    'train_and_evaluate_model',
    'preprocess_images',
    'plot_training_history',
    'plot_confusion_matrix',
    
    # Model instantiation functions
    'instantiate_model_hi_mvit',
    'instantiate_model_himvit',
    'instantiate_model_medkan',
    'instantiate_model_vgg16',
    'instantiate_model_resnet50',
    'instantiate_model_resnet101', 
    'instantiate_model_resnet152',
    'instantiate_model_resnet',
    'get_best_resnet'
]