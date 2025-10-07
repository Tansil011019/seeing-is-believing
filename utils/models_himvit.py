"""
HI_MViT (Hierarchical Inverted MobileViT) model implementation.
Based on "HI-MViT: A Lightweight Model for Explainable Skin Disease Classification Based on Modified MobileViT"
by Ding et al. (2023)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class InvertedResidualBlock(layers.Layer):
    """Inverted Residual Block with depthwise separable convolutions."""
    
    def __init__(self, 
                 output_channels: int,
                 expansion_factor: int = 6,
                 stride: int = 1,
                 use_residual: bool = True,
                 name: str = "inverted_residual",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_channels = output_channels
        self.expansion_factor = expansion_factor
        self.stride = stride
        self.use_residual = use_residual
        
    def build(self, input_shape):
        input_channels = input_shape[-1]
        expanded_channels = input_channels * self.expansion_factor
        
        # Expansion phase
        if self.expansion_factor != 1:
            self.expand_conv = layers.Conv2D(
                expanded_channels, 1, use_bias=False, name=f"{self.name}/expand_conv"
            )
            self.expand_bn = layers.BatchNormalization(name=f"{self.name}/expand_bn")
            self.expand_activation = layers.ReLU6(name=f"{self.name}/expand_relu")
        
        # Depthwise convolution
        self.depthwise_conv = layers.DepthwiseConv2D(
            3, strides=self.stride, padding='same', use_bias=False,
            name=f"{self.name}/depthwise_conv"
        )
        self.depthwise_bn = layers.BatchNormalization(name=f"{self.name}/depthwise_bn")
        self.depthwise_activation = layers.ReLU6(name=f"{self.name}/depthwise_relu")
        
        # Pointwise convolution
        self.pointwise_conv = layers.Conv2D(
            self.output_channels, 1, use_bias=False, name=f"{self.name}/pointwise_conv"
        )
        self.pointwise_bn = layers.BatchNormalization(name=f"{self.name}/pointwise_bn")
        
        # Residual connection check
        self.can_use_residual = (
            self.use_residual and 
            self.stride == 1 and 
            input_channels == self.output_channels
        )
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        x = inputs
        
        # Expansion phase
        if self.expansion_factor != 1:
            x = self.expand_conv(x)
            x = self.expand_bn(x, training=training)
            x = self.expand_activation(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x, training=training)
        x = self.depthwise_activation(x)
        
        # Pointwise convolution
        x = self.pointwise_conv(x)
        x = self.pointwise_bn(x, training=training)
        
        # Residual connection
        if self.can_use_residual:
            x = layers.Add()([inputs, x])
        
        return x


class MobileViTBlock(layers.Layer):
    """MobileViT block combining local and global representations."""
    
    def __init__(self,
                 d_model: int = 96,
                 patch_size: int = 2,
                 num_heads: int = 4,
                 mlp_ratio: float = 2.0,
                 dropout_rate: float = 0.1,
                 name: str = "mobilevit_block",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        # Local representation
        self.local_conv1 = layers.Conv2D(
            self.d_model, 3, padding='same', use_bias=False,
            name=f"{self.name}/local_conv1"
        )
        self.local_bn1 = layers.BatchNormalization(name=f"{self.name}/local_bn1")
        self.local_activation1 = layers.Activation('swish', name=f"{self.name}/local_swish1")
        
        # Global representation (Transformer)
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model // self.num_heads,
            dropout=self.dropout_rate,
            name=f"{self.name}/attention"
        )
        self.norm1 = layers.LayerNormalization(name=f"{self.name}/norm1")
        self.norm2 = layers.LayerNormalization(name=f"{self.name}/norm2")
        
        # MLP
        mlp_hidden_dim = int(self.d_model * self.mlp_ratio)
        self.mlp = keras.Sequential([
            layers.Dense(mlp_hidden_dim, activation='gelu', name=f"{self.name}/mlp_dense1"),
            layers.Dropout(self.dropout_rate, name=f"{self.name}/mlp_dropout"),
            layers.Dense(self.d_model, name=f"{self.name}/mlp_dense2")
        ], name=f"{self.name}/mlp")
        
        # Fusion
        self.fusion_conv = layers.Conv2D(
            input_shape[-1], 3, padding='same', use_bias=False,
            name=f"{self.name}/fusion_conv"
        )
        self.fusion_bn = layers.BatchNormalization(name=f"{self.name}/fusion_bn")
        self.fusion_activation = layers.Activation('swish', name=f"{self.name}/fusion_swish")
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        B, H, W, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        
        # Local representation
        local_features = self.local_conv1(inputs)
        local_features = self.local_bn1(local_features, training=training)
        local_features = self.local_activation1(local_features)
        
        # Unfold to patches for global representation
        patches = tf.image.extract_patches(
            local_features,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        
        # Reshape for transformer
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        patches = tf.reshape(patches, [B, patch_h * patch_w, -1])
        
        # Project to d_model dimension
        patches = layers.Dense(self.d_model, name=f"{self.name}/patch_projection")(patches)
        
        # Apply transformer
        # Self-attention
        attn_output = self.attention(patches, patches, training=training)
        patches = self.norm1(patches + attn_output)
        
        # MLP
        mlp_output = self.mlp(patches, training=training)
        patches = self.norm2(patches + mlp_output)
        
        # Fold back to spatial representation
        global_features = tf.reshape(patches, [B, patch_h, patch_w, self.d_model])
        global_features = tf.image.resize(global_features, [H, W], method='bilinear')
        
        # Fusion
        fused = tf.concat([local_features, global_features], axis=-1)
        fused = self.fusion_conv(fused)
        fused = self.fusion_bn(fused, training=training)
        fused = self.fusion_activation(fused)
        
        return fused


def instantiate_model_hi_mvit(input_shape: Tuple[int, int, int] = (224, 224, 3),
                             num_classes: int = 7,
                             include_top: bool = True,
                             weights: Optional[str] = None,
                             pooling: Optional[str] = None) -> keras.Model:
    """
    Instantiate HI_MViT model for skin disease classification.
    
    Args:
        input_shape (Tuple[int, int, int]): Input image shape (default: (224, 224, 3))
        num_classes (int): Number of output classes (default: 7)
        include_top (bool): Whether to include classification head (default: True)
        weights (str): Pre-trained weights to load (default: None)
        pooling (str): Pooling mode for feature extraction (default: None)
    
    Returns:
        keras.Model: HI_MViT model
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Initial conv stem
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation('swish', name="stem_swish")(x)
    
    # Stage 1: Inverted residual blocks
    x = InvertedResidualBlock(64, expansion_factor=1, stride=1, name="stage1_block1")(x)
    x = InvertedResidualBlock(64, expansion_factor=6, stride=2, name="stage1_block2")(x)
    x = InvertedResidualBlock(64, expansion_factor=6, stride=1, name="stage1_block3")(x)
    
    # Stage 2: More inverted residual blocks
    x = InvertedResidualBlock(96, expansion_factor=6, stride=2, name="stage2_block1")(x)
    x = InvertedResidualBlock(96, expansion_factor=6, stride=1, name="stage2_block2")(x)
    x = InvertedResidualBlock(96, expansion_factor=6, stride=1, name="stage2_block3")(x)
    
    # Stage 3: MobileViT blocks for global representation
    x = MobileViTBlock(d_model=144, patch_size=2, num_heads=4, name="mobilevit_block1")(x)
    x = InvertedResidualBlock(128, expansion_factor=6, stride=2, name="stage3_block1")(x)
    x = MobileViTBlock(d_model=192, patch_size=2, num_heads=4, name="mobilevit_block2")(x)
    
    # Stage 4: Final processing
    x = InvertedResidualBlock(160, expansion_factor=6, stride=1, name="stage4_block1")(x)
    x = InvertedResidualBlock(160, expansion_factor=6, stride=1, name="stage4_block2")(x)
    
    # Final conv layer
    x = layers.Conv2D(640, 1, use_bias=False, name="final_conv")(x)
    x = layers.BatchNormalization(name="final_bn")(x)
    x = layers.Activation('swish', name="final_swish")(x)
    
    if include_top:
        # Global average pooling and classification
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.Dropout(0.2, name="dropout")(x)
        
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
    
    model = keras.Model(inputs, outputs, name="HI_MViT")
    
    if weights is not None:
        model.load_weights(weights)
    
    return model


# For backwards compatibility
def instantiate_model_himvit(*args, **kwargs):
    """Alias for instantiate_model_hi_mvit"""
    return instantiate_model_hi_mvit(*args, **kwargs)