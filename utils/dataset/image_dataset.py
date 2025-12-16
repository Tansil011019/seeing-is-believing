from torch.utils.data import Dataset
from PIL import Image
import logging
import pandas as pd
import io

logger = logging.getLogger(__name__)

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
    
class ParquetImageDataset(Dataset):
    def __init__(self, meta_df, parquet_path, label_map, transform=None):
        super().__init__()
        self.meta_df = meta_df
        self.parquet_path = parquet_path
        logger.info(f"Loading parquet file from {parquet_path}")
        image_data_df = pd.read_parquet(parquet_path)

        image_data_df['filename'] = image_data_df['filename'].str.replace(r'\.(jpg|png)$', '', regex=True)

        self.df = pd.merge(
            meta_df,
            image_data_df,
            left_on='image',
            right_on='filename',
            how='inner'
        )

        self.label_map = label_map
        self.transform = transform

    def __getitem__(self, index):
        row = self.df.iloc[index]

        label_str = row['label']
        label_int = self.label_map[label_str]

        images = row['image']

        raw_bytes = row['image_bytes']
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        if image is None:
            raise FileNotFoundError(label_str)

        if self.transform:
            image = self.transform(image)

        return image, label_int, images
    
    def __len__(self):
        return len(self.df)

        