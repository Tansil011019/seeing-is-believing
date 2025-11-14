import torch.nn as nn
from torchvision import models

class SqueezeNet_1_1_Custom(nn.Module):
    def __init__(self, weights, num_classes=10):
        super(SqueezeNet_1_1_Custom, self).__init__()
        self.model = models.squeezenet1_1(weights=weights)
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.num_classes = num_classes

    def forward(self, x):
        return self.model(x)