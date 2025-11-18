import torch
from transformers import AutoModel

import torch.nn as nn


class EfficientSAMCustom(nn.Module):
    def __init__(self, model_name: str = "yunyangx/EfficientSAM"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.input_size = 512
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Tensor of shape (B, 3, 512, 512) with values in [0, 1]
        
        Returns:
            output: Segmentation mask tensor
        """
        # Ensure input is correct size
        if image.shape[-2:] != (self.input_size, self.input_size):
            image = torch.nn.functional.interpolate(
                image, 
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Forward pass through model
        outputs = self.model(image)
        
        return outputs