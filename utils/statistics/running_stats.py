import torch

class RunningStatsCalculator:
    """
    References:
        https://www.kaggle.com/code/kozodoi/computing-dataset-mean-and-std
    """

    def __init__(self, channels=3):
        self.sum_pixels = torch.zeros(channels)
        self.sum_sq_pixels = torch.zeros(channels)
        self.total_pixels = 0

    def update(self, image_batch):
        image_batch = image_batch.to('cpu')
        self.sum_pixels += image_batch.sum(dim=[0, 2, 3])
        self.sum_sq_pixels += (image_batch**2).sum(dim=[0, 2, 3])
        self.total_pixels += image_batch.size(0) * image_batch.size(2) * image_batch.size(3)

    def compute(self):
        if self.total_pixels == 0:
            return torch.zeros(3), torch.zeros(3)
        
        mean = self.sum_pixels / self.total_pixels
        mean_of_squares = self.sum_sq_pixels / self.total_pixels
        std = torch.sqrt(mean_of_squares - mean**2)

        return mean, std