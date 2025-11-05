"""
Neural network layers for segmentation models
"""
import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    """Attention block for feature refinement"""
    
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Attention mechanism
        self.attention = self._build_attention(out_channels)
        
        # Skip connection
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def _build_attention(self, channels):
        """Build attention module"""
        return nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply attention
        attention_weights = self.attention(out)
        out = out * attention_weights
        
        out += identity
        out = self.relu(out)
        
        return out


class MultiscaleLayer(nn.Module):
    """Multiscale layer with pooling and dilated convolutions"""
    
    def __init__(self, in_channels, out_channels):
        super(MultiscaleLayer, self).__init__()
        
        # 1x1 conv for input
        self.conv_1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn_1x1 = nn.BatchNorm2d(out_channels)
        
        # Pooling layers
        self.pool_3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool_5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.pool_7 = nn.MaxPool2d(7, stride=1, padding=3)
        
        # Dilated convolutions
        self.dilated_3 = self._build_dilated_conv(out_channels, 3)
        self.dilated_6 = self._build_dilated_conv(out_channels, 6)
        self.dilated_12 = self._build_dilated_conv(out_channels, 12)
        
        # Output fusion (6 branches concatenated)
        self.fusion = nn.Conv2d(out_channels * 6, out_channels, 1)
        self.bn_fusion = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def _build_dilated_conv(self, channels, dilation):
        """Build dilated convolution block"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 1x1 conv
        x = self.conv_1x1(x)
        x = self.bn_1x1(x)
        x = self.relu(x)
        
        # Multi-scale branches
        branches = [
            self.pool_3(x),
            self.pool_5(x),
            self.pool_7(x),
            self.dilated_3(x),
            self.dilated_6(x),
            self.dilated_12(x)
        ]
        
        # Concatenate and fuse
        out = torch.cat(branches, dim=1)
        out = self.fusion(out)
        out = self.bn_fusion(out)
        out = self.relu(out)
        
        return out
