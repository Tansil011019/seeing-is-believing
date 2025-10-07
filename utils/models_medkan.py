"""
MedKAN (Medical Kolmogorov-Arnold Networks) model implementation.
A KAN-based architecture for medical image classification.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, List


class KANLayer(layers.Layer):
    """
    Kolmogorov-Arnold Network Layer implementation.
    Uses learnable spline functions instead of fixed activation functions.
    """
    
    def __init__(self, 
                 output_dim: int,
                 num_knots: int = 5,
                 spline_order: int = 3,
                 name: str = "kan_layer",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.num_knots = num_knots
        self.spline_order = spline_order
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Initialize knot positions
        self.knots = self.add_weight(
            name="knots",
            shape=(input_dim, self.output_dim, self.num_knots),
            initializer="uniform",
            trainable=True
        )
        
        # Initialize spline coefficients
        self.coefficients = self.add_weight(
            name="coefficients",
            shape=(input_dim, self.output_dim, self.num_knots + self.spline_order),
            initializer="he_normal",
            trainable=True
        )
        
        # Base linear transformation
        self.linear_weight = self.add_weight(
            name="linear_weight",
            shape=(input_dim, self.output_dim),
            initializer="he_normal",
            trainable=True
        )
        
        self.bias = self.add_weight(
            name="bias",
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True
        )
        
        super().build(input_shape)
    
    def spline_basis(self, x, knots, order):
        """Compute B-spline basis functions."""
        # Simplified B-spline implementation
        batch_size = tf.shape(x)[0]
        input_dim = tf.shape(x)[1]
        
        # Expand dimensions for broadcasting
        x_expanded = tf.expand_dims(x, -1)  # [batch, input_dim, 1]
        knots_sorted = tf.sort(knots, axis=-1)  # [input_dim, output_dim, num_knots]
        
        # Compute basis functions (simplified linear interpolation)
        distances = tf.abs(x_expanded[..., None] - knots_sorted[None, ...])  # [batch, input_dim, output_dim, num_knots]
        weights = tf.nn.softmax(-distances * 10.0, axis=-1)  # Softmax to create weights
        
        return weights
    
    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # Compute spline basis for each input dimension
        basis_weights = self.spline_basis(inputs, self.knots, self.spline_order)
        
        # Apply spline transformation
        spline_output = tf.reduce_sum(
            basis_weights * self.coefficients[None, :, :, :self.num_knots], 
            axis=-1
        )  # [batch, input_dim, output_dim]
        
        # Sum across input dimensions and add linear component
        kan_output = tf.reduce_sum(spline_output, axis=1)  # [batch, output_dim]
        linear_output = tf.matmul(inputs, self.linear_weight)
        
        output = kan_output + linear_output + self.bias
        
        return output


class MedKANBlock(layers.Layer):
    """MedKAN block combining convolutional features with KAN layers."""
    
    def __init__(self,
                 filters: int,
                 kan_hidden_dim: int = 64,
                 num_knots: int = 5,
                 dropout_rate: float = 0.1,
                 name: str = "medkan_block",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kan_hidden_dim = kan_hidden_dim
        self.num_knots = num_knots
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        # Convolutional feature extraction
        self.conv1 = layers.Conv2D(
            self.filters, 3, padding='same', use_bias=False,
            name=f"{self.name}/conv1"
        )
        self.bn1 = layers.BatchNormalization(name=f"{self.name}/bn1")
        self.activation1 = layers.Activation('gelu', name=f"{self.name}/gelu1")
        
        self.conv2 = layers.Conv2D(
            self.filters, 3, padding='same', use_bias=False,
            name=f"{self.name}/conv2"
        )
        self.bn2 = layers.BatchNormalization(name=f"{self.name}/bn2")
        
        # Spatial attention using KAN
        self.spatial_kan = KANLayer(
            self.filters, 
            num_knots=self.num_knots,
            name=f"{self.name}/spatial_kan"
        )
        
        # Channel attention using KAN
        self.channel_kan = KANLayer(
            self.filters,
            num_knots=self.num_knots,
            name=f"{self.name}/channel_kan"
        )
        
        self.dropout = layers.Dropout(self.dropout_rate, name=f"{self.name}/dropout")
        
        # Residual connection
        if input_shape[-1] != self.filters:
            self.residual_conv = layers.Conv2D(
                self.filters, 1, use_bias=False,
                name=f"{self.name}/residual_conv"
            )
            self.residual_bn = layers.BatchNormalization(name=f"{self.name}/residual_bn")
        else:
            self.residual_conv = None
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        # Convolutional processing
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Spatial attention with KAN
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Global average pooling for spatial attention
        spatial_features = tf.reduce_mean(x, axis=[1, 2])  # [B, C]
        spatial_attention = self.spatial_kan(spatial_features, training=training)
        spatial_attention = tf.nn.sigmoid(spatial_attention)
        spatial_attention = tf.reshape(spatial_attention, [B, 1, 1, C])
        
        # Apply spatial attention
        x = x * spatial_attention
        
        # Channel attention with KAN
        channel_features = tf.reduce_mean(x, axis=-1)  # [B, H, W]
        channel_features = tf.reshape(channel_features, [B, H * W])
        
        # Reduce dimensions for KAN processing
        channel_features_reduced = layers.Dense(C, name=f"{self.name}/channel_reduce")(channel_features)
        channel_attention = self.channel_kan(channel_features_reduced, training=training)
        channel_attention = tf.nn.sigmoid(channel_attention)
        channel_attention = tf.reshape(channel_attention, [B, 1, 1, C])
        
        # Apply channel attention
        x = x * channel_attention
        
        x = self.dropout(x, training=training)
        
        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(inputs)
            residual = self.residual_bn(residual, training=training)
        else:
            residual = inputs
        
        x = layers.Add()([x, residual])
        x = layers.Activation('gelu')(x)
        
        return x


def instantiate_model_medkan(input_shape: Tuple[int, int, int] = (224, 224, 3),
                            num_classes: int = 7,
                            include_top: bool = True,
                            weights: Optional[str] = None,
                            pooling: Optional[str] = None) -> keras.Model:
    """
    Instantiate MedKAN model for medical image classification.
    
    Args:
        input_shape (Tuple[int, int, int]): Input image shape (default: (224, 224, 3))
        num_classes (int): Number of output classes (default: 7)
        include_top (bool): Whether to include classification head (default: True)
        weights (str): Pre-trained weights to load (default: None)
        pooling (str): Pooling mode for feature extraction (default: None)
    
    Returns:
        keras.Model: MedKAN model
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Initial conv stem
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation('gelu', name="stem_gelu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name="stem_pool")(x)
    
    # Stage 1: Low-level features
    x = MedKANBlock(64, kan_hidden_dim=32, name="stage1_block1")(x)
    x = MedKANBlock(64, kan_hidden_dim=32, name="stage1_block2")(x)
    x = layers.MaxPooling2D(2, strides=2, name="stage1_pool")(x)
    
    # Stage 2: Mid-level features
    x = MedKANBlock(128, kan_hidden_dim=64, name="stage2_block1")(x)
    x = MedKANBlock(128, kan_hidden_dim=64, name="stage2_block2")(x)
    x = MedKANBlock(128, kan_hidden_dim=64, name="stage2_block3")(x)
    x = layers.MaxPooling2D(2, strides=2, name="stage2_pool")(x)
    
    # Stage 3: High-level features with stronger KAN integration
    x = MedKANBlock(256, kan_hidden_dim=128, num_knots=7, name="stage3_block1")(x)
    x = MedKANBlock(256, kan_hidden_dim=128, num_knots=7, name="stage3_block2")(x)
    x = MedKANBlock(256, kan_hidden_dim=128, num_knots=7, name="stage3_block3")(x)
    x = MedKANBlock(256, kan_hidden_dim=128, num_knots=7, name="stage3_block4")(x)
    x = layers.MaxPooling2D(2, strides=2, name="stage3_pool")(x)
    
    # Stage 4: Deep semantic features
    x = MedKANBlock(512, kan_hidden_dim=256, num_knots=9, name="stage4_block1")(x)
    x = MedKANBlock(512, kan_hidden_dim=256, num_knots=9, name="stage4_block2")(x)
    x = MedKANBlock(512, kan_hidden_dim=256, num_knots=9, name="stage4_block3")(x)
    
    if include_top:
        # Global pooling and KAN-based classification
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        
        # KAN-based classifier
        x = KANLayer(256, num_knots=7, name="classifier_kan1")(x)
        x = layers.Dropout(0.3, name="classifier_dropout1")(x)
        x = layers.Activation('gelu', name="classifier_gelu")(x)
        
        x = KANLayer(128, num_knots=5, name="classifier_kan2")(x)
        x = layers.Dropout(0.2, name="classifier_dropout2")(x)
        
        if num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid', name="predictions")(x)
        else:
            outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
        outputs = x
    
    model = keras.Model(inputs, outputs, name="MedKAN")
    
    if weights is not None:
        model.load_weights(weights)
    
    return model