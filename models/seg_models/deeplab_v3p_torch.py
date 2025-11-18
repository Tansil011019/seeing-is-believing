"""
DeepLabv3+ implementation in PyTorch
Converted from Keras implementation to maintain same architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 dilation=1, bias=False, padding_mode='zeros'):
        super(SeparableConv2d, self).__init__()
        
        if stride == 1:
            padding = dilation * (kernel_size - 1) // 2
        else:
            padding = 0
            
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias, padding_mode=padding_mode
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SepConvBN(nn.Module):
    """Separable Convolution with Batch Normalization"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, depth_activation=False):
        super(SepConvBN, self).__init__()
        
        self.depth_activation = depth_activation
        
        if not depth_activation:
            self.relu_before = nn.ReLU(inplace=True)
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, dilation=dilation, 
            padding=dilation * (kernel_size - 1) // 2 if stride == 1 else 0,
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3)
        
        if depth_activation:
            self.relu_mid = nn.ReLU(inplace=True)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-3)
        
        if depth_activation:
            self.relu_after = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if not self.depth_activation:
            x = self.relu_before(x)
        
        x = self.depthwise(x)
        x = self.bn1(x)
        
        if self.depth_activation:
            x = self.relu_mid(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        
        if self.depth_activation:
            x = self.relu_after(x)
        
        return x


class XceptionBlock(nn.Module):
    """Xception Block"""
    def __init__(self, in_channels, out_channels_list, stride=1, dilation=1,
                 skip_connection_type='conv', depth_activation=False, return_skip=False):
        super(XceptionBlock, self).__init__()
        
        assert len(out_channels_list) == 3
        
        self.skip_connection_type = skip_connection_type
        self.return_skip = return_skip
        
        # Three separable convolutions
        self.sepconv1 = SepConvBN(
            in_channels, out_channels_list[0], stride=1,
            dilation=dilation, depth_activation=depth_activation
        )
        self.sepconv2 = SepConvBN(
            out_channels_list[0], out_channels_list[1], stride=1,
            dilation=dilation, depth_activation=depth_activation
        )
        self.sepconv3 = SepConvBN(
            out_channels_list[1], out_channels_list[2], stride=stride,
            dilation=dilation, depth_activation=depth_activation
        )
        
        # Skip connection
        if skip_connection_type == 'conv':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels_list[-1], kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels_list[-1])
            )
    
    def forward(self, x):
        residual = x
        
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        skip = x
        x = self.sepconv3(x)
        
        if self.skip_connection_type == 'conv':
            residual = self.shortcut(residual)
            x = x + residual
        elif self.skip_connection_type == 'sum':
            x = x + residual
        
        if self.return_skip:
            return x, skip
        else:
            return x


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolutions
        self.aspp1 = SepConvBN(
            in_channels, out_channels, kernel_size=3,
            dilation=atrous_rates[0], depth_activation=True
        )
        self.aspp2 = SepConvBN(
            in_channels, out_channels, kernel_size=3,
            dilation=atrous_rates[1], depth_activation=True
        )
        self.aspp3 = SepConvBN(
            in_channels, out_channels, kernel_size=3,
            dilation=atrous_rates[2], depth_activation=True
        )
        
        # Image pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        size = x.shape[2:]
        
        x1 = self.conv1(x)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=True)
        
        return x1, x2, x3, x4, x5


class DeepLabV3Plus(nn.Module):
    """
    DeepLabv3+ model in PyTorch
    Maintains the same architecture as the Keras version
    """
    def __init__(self, in_channels=3, num_classes=1, output_stride=16):
        super(DeepLabV3Plus, self).__init__()
        
        if output_stride == 8:
            entry_block3_stride = 1
            middle_block_rate = 2
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:  # output_stride == 16
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)
        
        # Entry flow
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.block1 = XceptionBlock(64, [128, 128, 128], stride=2, 
                                     skip_connection_type='conv')
        self.block2 = XceptionBlock(128, [256, 256, 256], stride=2,
                                     skip_connection_type='conv', return_skip=True)
        self.block3 = XceptionBlock(256, [728, 728, 728], stride=entry_block3_stride,
                                     skip_connection_type='conv')
        
        # Middle flow
        self.middle_blocks = nn.ModuleList([
            XceptionBlock(728, [728, 728, 728], stride=1, dilation=middle_block_rate,
                         skip_connection_type='sum')
            for _ in range(16)
        ])
        
        # Exit flow
        self.block17 = XceptionBlock(728, [728, 1024, 1024], stride=1,
                                      dilation=exit_block_rates[0],
                                      skip_connection_type='conv')
        self.block18 = XceptionBlock(1024, [1536, 1536, 2048], stride=1,
                                      dilation=exit_block_rates[1],
                                      skip_connection_type='none',
                                      depth_activation=True)
        
        # ASPP
        self.aspp = ASPP(2048, 256, atrous_rates)
        
        # Additional processing for ASPP outputs (matching Keras architecture)
        self.aspp_conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            )
            for _ in range(5)
        ])
        
        # SE-like attention for ASPP outputs
        self.aspp_attention = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 256),
            nn.Sigmoid()
        )
        
        # 3D convolution for combining ASPP branches
        self.conv3d = nn.Conv3d(256, 256, kernel_size=(1, 1, 5), bias=False)
        self.conv3d_relu = nn.ReLU(inplace=True)
        
        # Projection after ASPP combination
        self.aspp_projection = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Decoder
        self.decoder_skip = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48, eps=1e-5),
            nn.ReLU(inplace=True)
        )
        
        self.decoder_conv1 = SepConvBN(304, 256, depth_activation=True)
        self.decoder_conv2 = SepConvBN(256, 256, depth_activation=True)
        
        # Final convolutions
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, num_classes, 1)
        )
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Entry flow
        x = self.conv1(x)
        x = self.block1(x)
        x, skip1 = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        for block in self.middle_blocks:
            x = block(x)
        
        # Exit flow
        x = self.block17(x)
        x = self.block18(x)
        
        # ASPP
        aspp_outputs = self.aspp(x)  # Returns 5 tensors
        
        # Process each ASPP output with residual blocks
        processed_outputs = []
        for i, aspp_out in enumerate(aspp_outputs):
            # Apply conv blocks
            conv_out = self.aspp_conv_blocks[i](aspp_out)
            # Residual connection
            residual = aspp_out + conv_out
            
            # Apply attention
            b, c, h, w = residual.shape
            att = F.adaptive_avg_pool2d(residual, 1).view(b, c)
            att = self.aspp_attention(att).view(b, c, 1, 1)
            residual = residual * att
            
            processed_outputs.append(residual)
        
        # Stack for 3D convolution: (B, C, H, W, 5)
        aspp_stack = torch.stack(processed_outputs, dim=4)  # (B, 256, H, W, 5)
        aspp_stack = aspp_stack.permute(0, 1, 4, 2, 3)  # (B, 256, 5, H, W)
        
        # Apply 3D convolution
        x = self.conv3d(aspp_stack)  # (B, 256, 1, H, W)
        x = self.conv3d_relu(x)
        x = x.squeeze(2)  # (B, 256, H, W)
        
        # Additional projection
        x = self.aspp_projection(x)
        
        # Decoder
        x = F.interpolate(x, size=(skip1.shape[2], skip1.shape[3]),
                         mode='bilinear', align_corners=True)
        
        skip1 = self.decoder_skip(skip1)
        x = torch.cat([x, skip1], dim=1)
        
        x = self.decoder_conv1(x)
        x = self.decoder_conv2(x)
        
        # Final convolutions
        x = self.final_conv(x)
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x


def deeplabv3plus(num_classes=1, output_stride=16, pretrained=False):
    """
    Create DeepLabv3+ model
    
    Args:
        num_classes: Number of output classes (1 for binary segmentation)
        output_stride: Output stride (8 or 16)
        pretrained: Whether to load pretrained weights (not implemented)
    
    Returns:
        DeepLabV3Plus model
    """
    model = DeepLabV3Plus(num_classes=num_classes, output_stride=output_stride)
    
    if pretrained:
        print("Warning: Pretrained weights not available for PyTorch version")
    
    return model
