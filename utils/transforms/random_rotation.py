import random
from torchvision.transforms import functional as F

class RandomDiscreteRotation:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return F.rotate(img, angle)