"""
Standard deep learning models for skin disease classification.
Includes VGG16 and ResNet implementations with transfer learning support.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
import numpy as np
from typing import Tuple, Optional


def instantiate_model_vgg16(input_shape: Tuple[int, int, int] = (224, 224, 3),
                           num_classes: int = 7,
                           include_top: bool = True,
                           weights: str = 'imagenet',
                           pooling: Optional[str] = None,
                           freeze_base: bool = False,
                           fine_tune_layers: int = 0) -> keras.Model:
    """
    Instantiate VGG16 model for skin disease classification.
    
    Args:
        input_shape (Tuple[int, int, int]): Input image shape (default: (224, 224, 3))
        num_classes (int): Number of output classes (default: 7)
        include_top (bool): Whether to include classification head (default: True)
        weights (str): Pre-trained weights ('imagenet' or None) (default: 'imagenet')
        pooling (str): Pooling mode for feature extraction (default: None)
        freeze_base (bool): Whether to freeze base model weights (default: False)
        fine_tune_layers (int): Number of top layers to fine-tune (default: 0)
    
    Returns:
        keras.Model: VGG16 model for classification
    """
    # Load pre-trained VGG16 base model
    base_model = applications.VGG16(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
        pooling=pooling if not include_top else None
    )
    
    # Freeze base model if specified
    if freeze_base:
        base_model.trainable = False
    elif fine_tune_layers > 0:
        # Freeze all layers except the top few
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
    
    if include_top:
        # Build custom classification head
        inputs = keras.Input(shape=input_shape, name="input_image")
        x = base_model(inputs, training=True)
        
        # Add custom classification layers
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.Dropout(0.5, name="dropout1")(x)
        
        x = layers.Dense(512, activation='relu', name="fc1")(x)
        x = layers.BatchNormalization(name="bn2")(x)
        x = layers.Dropout(0.3, name="dropout2")(x)
        
        x = layers.Dense(256, activation='relu', name="fc2")(x)
        x = layers.Dropout(0.2, name="dropout3")(x)
        
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name="predictions")(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)
        
        model = keras.Model(inputs, outputs, name="VGG16_Custom")
    else:
        model = base_model
    
    return model


def instantiate_model_resnet50(input_shape: Tuple[int, int, int] = (224, 224, 3),
                              num_classes: int = 7,
                              include_top: bool = True,
                              weights: str = 'imagenet',
                              pooling: Optional[str] = None,
                              freeze_base: bool = False,
                              fine_tune_layers: int = 0) -> keras.Model:
    """
    Instantiate ResNet50 model for skin disease classification.
    
    Args:
        input_shape (Tuple[int, int, int]): Input image shape (default: (224, 224, 3))
        num_classes (int): Number of output classes (default: 7)
        include_top (bool): Whether to include classification head (default: True)
        weights (str): Pre-trained weights ('imagenet' or None) (default: 'imagenet')
        pooling (str): Pooling mode for feature extraction (default: None)
        freeze_base (bool): Whether to freeze base model weights (default: False)
        fine_tune_layers (int): Number of top layers to fine-tune (default: 0)
    
    Returns:
        keras.Model: ResNet50 model for classification
    """
    # Load pre-trained ResNet50 base model
    base_model = applications.ResNet50(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
        pooling=pooling if not include_top else None
    )
    
    # Freeze base model if specified
    if freeze_base:
        base_model.trainable = False
    elif fine_tune_layers > 0:
        # Freeze all layers except the top few
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
    
    if include_top:
        # Build custom classification head
        inputs = keras.Input(shape=input_shape, name="input_image")
        x = base_model(inputs, training=True)
        
        # Add custom classification layers
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.Dropout(0.4, name="dropout1")(x)
        
        x = layers.Dense(1024, activation='relu', name="fc1")(x)
        x = layers.BatchNormalization(name="bn2")(x)
        x = layers.Dropout(0.3, name="dropout2")(x)
        
        x = layers.Dense(512, activation='relu', name="fc2")(x)
        x = layers.BatchNormalization(name="bn3")(x)
        x = layers.Dropout(0.2, name="dropout3")(x)
        
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name="predictions")(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)
        
        model = keras.Model(inputs, outputs, name="ResNet50_Custom")
    else:
        model = base_model
    
    return model


def instantiate_model_resnet101(input_shape: Tuple[int, int, int] = (224, 224, 3),
                               num_classes: int = 7,
                               include_top: bool = True,
                               weights: str = 'imagenet',
                               pooling: Optional[str] = None,
                               freeze_base: bool = False,
                               fine_tune_layers: int = 0) -> keras.Model:
    """
    Instantiate ResNet101 model for skin disease classification.
    
    Args:
        input_shape (Tuple[int, int, int]): Input image shape (default: (224, 224, 3))
        num_classes (int): Number of output classes (default: 7)
        include_top (bool): Whether to include classification head (default: True)
        weights (str): Pre-trained weights ('imagenet' or None) (default: 'imagenet')
        pooling (str): Pooling mode for feature extraction (default: None)
        freeze_base (bool): Whether to freeze base model weights (default: False)
        fine_tune_layers (int): Number of top layers to fine-tune (default: 0)
    
    Returns:
        keras.Model: ResNet101 model for classification
    """
    # Load pre-trained ResNet101 base model
    base_model = applications.ResNet101(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
        pooling=pooling if not include_top else None
    )
    
    # Freeze base model if specified
    if freeze_base:
        base_model.trainable = False
    elif fine_tune_layers > 0:
        # Freeze all layers except the top few
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
    
    if include_top:
        # Build custom classification head
        inputs = keras.Input(shape=input_shape, name="input_image")
        x = base_model(inputs, training=True)
        
        # Add custom classification layers with deeper head for ResNet101
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.Dropout(0.5, name="dropout1")(x)
        
        x = layers.Dense(2048, activation='relu', name="fc1")(x)
        x = layers.BatchNormalization(name="bn2")(x)
        x = layers.Dropout(0.4, name="dropout2")(x)
        
        x = layers.Dense(1024, activation='relu', name="fc2")(x)
        x = layers.BatchNormalization(name="bn3")(x)
        x = layers.Dropout(0.3, name="dropout3")(x)
        
        x = layers.Dense(512, activation='relu', name="fc3")(x)
        x = layers.Dropout(0.2, name="dropout4")(x)
        
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name="predictions")(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)
        
        model = keras.Model(inputs, outputs, name="ResNet101_Custom")
    else:
        model = base_model
    
    return model


def instantiate_model_resnet152(input_shape: Tuple[int, int, int] = (224, 224, 3),
                               num_classes: int = 7,
                               include_top: bool = True,
                               weights: str = 'imagenet',
                               pooling: Optional[str] = None,
                               freeze_base: bool = False,
                               fine_tune_layers: int = 0) -> keras.Model:
    """
    Instantiate ResNet152 model for skin disease classification.
    
    Args:
        input_shape (Tuple[int, int, int]): Input image shape (default: (224, 224, 3))
        num_classes (int): Number of output classes (default: 7)
        include_top (bool): Whether to include classification head (default: True)
        weights (str): Pre-trained weights ('imagenet' or None) (default: 'imagenet')
        pooling (str): Pooling mode for feature extraction (default: None)
        freeze_base (bool): Whether to freeze base model weights (default: False)
        fine_tune_layers (int): Number of top layers to fine-tune (default: 0)
    
    Returns:
        keras.Model: ResNet152 model for classification
    """
    # Load pre-trained ResNet152 base model
    base_model = applications.ResNet152(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
        pooling=pooling if not include_top else None
    )
    
    # Freeze base model if specified
    if freeze_base:
        base_model.trainable = False
    elif fine_tune_layers > 0:
        # Freeze all layers except the top few
        base_model.trainable = True
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
    
    if include_top:
        # Build custom classification head
        inputs = keras.Input(shape=input_shape, name="input_image")
        x = base_model(inputs, training=True)
        
        # Add custom classification layers with even deeper head for ResNet152
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.BatchNormalization(name="bn1")(x)
        x = layers.Dropout(0.5, name="dropout1")(x)
        
        x = layers.Dense(2048, activation='relu', name="fc1")(x)
        x = layers.BatchNormalization(name="bn2")(x)
        x = layers.Dropout(0.4, name="dropout2")(x)
        
        x = layers.Dense(1024, activation='relu', name="fc2")(x)
        x = layers.BatchNormalization(name="bn3")(x)
        x = layers.Dropout(0.4, name="dropout3")(x)
        
        x = layers.Dense(512, activation='relu', name="fc3")(x)
        x = layers.BatchNormalization(name="bn4")(x)
        x = layers.Dropout(0.3, name="dropout4")(x)
        
        x = layers.Dense(256, activation='relu', name="fc4")(x)
        x = layers.Dropout(0.2, name="dropout5")(x)
        
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name="predictions")(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)
        
        model = keras.Model(inputs, outputs, name="ResNet152_Custom")
    else:
        model = base_model
    
    return model


# Alias functions for consistency
def instantiate_model_resnet(variant: str = "resnet50", **kwargs) -> keras.Model:
    """
    Instantiate ResNet model with specified variant.
    
    Args:
        variant (str): ResNet variant ('resnet50', 'resnet101', 'resnet152')
        **kwargs: Additional arguments passed to model instantiation
    
    Returns:
        keras.Model: ResNet model
    """
    variant = variant.lower()
    
    if variant == "resnet50":
        return instantiate_model_resnet50(**kwargs)
    elif variant == "resnet101":
        return instantiate_model_resnet101(**kwargs)
    elif variant == "resnet152":
        return instantiate_model_resnet152(**kwargs)
    else:
        raise ValueError(f"Unknown ResNet variant: {variant}. Choose from 'resnet50', 'resnet101', 'resnet152'")


# For backwards compatibility and easy access
def get_best_resnet(**kwargs) -> keras.Model:
    """
    Get the best performing ResNet variant (ResNet152).
    
    Returns:
        keras.Model: ResNet152 model
    """
    return instantiate_model_resnet152(**kwargs)