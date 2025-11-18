"""
Main segmentation model with attention and multiscale layers
"""
import torch
import torch.nn as nn
from torchvision import models
from .layers import AttentionBlock, MultiscaleLayer


class SegmentationModel(nn.Module):
    """
    Main segmentation model
    
    Architecture:
    a. Conv layer for input
    b. 7 Attention blocks
    c. Multiscale layer
    d. Upsample x4 layer
    e. Concatenate with low features from ResNet
    f. Conv layer
    g. Upsample x4 layer
    h. Conv layer to get segmentation mask
    """
    
    def __init__(self, in_channels=3, num_classes=1, base_channels=64):
        super(SegmentationModel, self).__init__()
        
        # a. Conv layer for input
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # ResNet backbone for low-level features
        resnet = models.resnet34(pretrained=True)
        self.resnet_low = nn.Sequential(*list(resnet.children())[:4])
        
        # b. 7 Attention blocks
        self.attention_blocks = self._build_attention_blocks(base_channels)
        
        # Downsampling between attention blocks
        self.downsample1 = nn.MaxPool2d(2)
        self.downsample2 = nn.MaxPool2d(2)
        self.downsample3 = nn.MaxPool2d(2)
        
        # c. Multiscale layer
        self.multiscale = MultiscaleLayer(base_channels * 2, base_channels * 4)
        
        # d. Upsample x4 layer
        self.upsample_4x_1 = self._build_upsample(
            base_channels * 4, base_channels * 2
        )
        
        # f. Conv layer after concatenation
        self.conv_after_concat = self._build_fusion_conv(base_channels)
        
        # g. Upsample x4 layer
        self.upsample_4x_2 = self._build_upsample(
            base_channels, base_channels // 2
        )
        
        # h. Output conv
        self.output_conv = self._build_output_conv(base_channels, num_classes)
    
    def _build_attention_blocks(self, base_channels):
        """Build 7 attention blocks"""
        return nn.ModuleList([
            AttentionBlock(base_channels, base_channels * 2),
            AttentionBlock(base_channels * 2, base_channels * 4),
            AttentionBlock(base_channels * 4, base_channels * 8),
            AttentionBlock(base_channels * 8, base_channels * 8),
            AttentionBlock(base_channels * 8, base_channels * 4),
            AttentionBlock(base_channels * 4, base_channels * 2),
            AttentionBlock(base_channels * 2, base_channels * 2)
        ])
    
    def _build_upsample(self, in_channels, out_channels):
        """Build upsampling block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _build_fusion_conv(self, base_channels):
        """Build fusion convolution after concatenation"""
        return nn.Sequential(
            nn.Conv2d(base_channels * 2 + 64, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
    
    def _build_output_conv(self, base_channels, num_classes):
        """Build output convolution"""
        return nn.Sequential(
            nn.Conv2d(base_channels // 2, base_channels // 4, 3, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 4, num_classes, 1)
        )
    
    def forward(self, x):
        # a. Input conv
        x1 = self.input_conv(x)
        
        # Extract low-level features
        low_features = self.resnet_low(x)
        
        # b. 7 Attention blocks with downsampling
        x2 = self._apply_attention_blocks(x1)
        
        # c. Multiscale layer
        x9 = self.multiscale(x2)
        
        # d. Upsample x4
        x10 = self.upsample_4x_1(x9)
        
        # e. Concatenate with low-level features
        x11 = torch.cat([x10, low_features], dim=1)
        
        # f. Conv layer
        x12 = self.conv_after_concat(x11)
        
        # g. Upsample x4
        x13 = self.upsample_4x_2(x12)
        
        # h. Output conv
        output = self.output_conv(x13)
        
        return output
    
    def _apply_attention_blocks(self, x):
        """Apply 7 attention blocks with downsampling"""
        x = self.attention_blocks[0](x)
        x = self.downsample1(x)
        
        x = self.attention_blocks[1](x)
        x = self.downsample2(x)
        
        x = self.attention_blocks[2](x)
        x = self.downsample3(x)
        
        x = self.attention_blocks[3](x)
        x = self.attention_blocks[4](x)
        x = self.attention_blocks[5](x)
        x = self.attention_blocks[6](x)
        
        return x
