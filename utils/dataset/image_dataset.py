from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, df, file_path, label_map, transform=None):
        super().__init__()
        self.df = df
        self.label_map = label_map
        self.transform = transform
        self.file_path = file_path

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = row['label']
        label_int = self.label_map[label]
        file_path = f"{self.file_path}/{row['image']}.jpg"
        image = Image.open(file_path).convert("RGB")
        if image is None:
            raise FileNotFoundError(file_path)

        if self.transform:
            image = self.transform(image)
        
        return image, label_int

    def __len__(self):
        return len(self.df)