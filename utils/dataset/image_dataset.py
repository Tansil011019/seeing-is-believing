from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, df, file_path, transform=None):
        super().__init__()
        self.df = df
        self.transform = transform
        self.file_path = file_path

    def __getitem__(self, index):
        file_path = f"{self.file_path}/{self.df['image'][index]}.jpg"
        image = Image.open(file_path).convert("RGB")
        if image is None:
            raise FileNotFoundError(file_path)

        if self.transform:
            image = self.transform(image)
        
        label = self.df['label'][index]
        return image, label

    def __len__(self):
        return len(self.df)