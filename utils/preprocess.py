import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
import random


def preprocess_images(images: List[Image.Image], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess PIL images for model training.
    
    Args:
        images (List[Image.Image]): List of PIL Image objects
        target_size (Tuple[int, int]): Target size for resizing (default: (224, 224))
    
    Returns:
        np.ndarray: Preprocessed image array of shape (n_samples, height, width, channels)
    """
    processed_images = []
    
    for img in images:
        # Convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        processed_images.append(img_array)
    
    return np.array(processed_images)


def sample_and_split(dataset_dir: str, sample_size: int, sample_random: int = 42, test_size: float = 0.2, test_random: int = 42) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Sample dataset with preserved label ratios and split into train/test sets.
    
    Args:
        dataset_dir (str): Directory containing the CSV file and images
        sample_size (int): Number of samples to extract
        sample_random (int): Random state for sampling
        test_size (float): Proportion of test set
        test_random (int): Random state for train/test split
    
    Returns:
        Tuple: X_train, y_train, X_test, y_test
    """
    # Find CSV file in dataset directory
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {dataset_dir}")
    
    csv_path = os.path.join(dataset_dir, csv_files[0])
    df = pd.read_csv(csv_path)
    
    # Calculate label ratios
    label_counts = df['label'].value_counts()
    total_samples = len(df)
    
    # Sample with preserved ratios
    sampled_data = []
    for label, count in label_counts.items():
        ratio = count / total_samples
        label_sample_size = int(sample_size * ratio)
        label_data = df[df['label'] == label].sample(n=min(label_sample_size, count), random_state=sample_random)
        sampled_data.append(label_data)
    
    sampled_df = pd.concat(sampled_data, ignore_index=True)
    
    # Split into train and test
    X = sampled_df['image']
    y = sampled_df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=test_random, stratify=y
    )
    
    return X_train, y_train, X_test, y_test


def rotative_oversample(dataset_dir: str, X: pd.Series, y: pd.Series, random: int = 42) -> Tuple[pd.Series, pd.Series]:
    """
    Rotate minority class images to match majority class count (max 8x upsampling).
    
    Args:
        dataset_dir (str): Directory containing images
        X (pd.Series): Image filenames
        y (pd.Series): Labels
        random (int): Random state
    
    Returns:
        Tuple: X_resampled, y_resampled
    """
    np.random.seed(random)
    random.seed(random)
    
    # Count classes
    label_counts = Counter(y)
    max_count = max(label_counts.values())
    
    # Rotation angles (up to 8 rotations: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
    rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    X_resampled = []
    y_resampled = []
    
    for label, count in label_counts.items():
        label_indices = y[y == label].index
        label_images = X.iloc[label_indices]
        
        # Add original images
        X_resampled.extend(label_images.tolist())
        y_resampled.extend([label] * len(label_images))
        
        if count < max_count:
            # Calculate how many additional samples needed (max 8x original)
            needed_samples = min(max_count - count, count * 7)  # 7x more = 8x total
            
            # Create rotated versions
            samples_created = 0
            angle_idx = 1  # Start from 45° (skip 0° as it's original)
            
            while samples_created < needed_samples and angle_idx < len(rotation_angles):
                for img_name in label_images:
                    if samples_created >= needed_samples:
                        break
                    
                    # Load and rotate image
                    img_path = os.path.join(dataset_dir, img_name)
                    try:
                        img = Image.open(img_path)
                        rotated_img = img.rotate(rotation_angles[angle_idx], expand=True)
                        
                        # Save rotated image
                        name_parts = img_name.split('.')
                        rotated_name = f"{name_parts[0]}_rot{rotation_angles[angle_idx]}.{name_parts[1]}"
                        rotated_path = os.path.join(dataset_dir, rotated_name)
                        rotated_img.save(rotated_path)
                        
                        X_resampled.append(rotated_name)
                        y_resampled.append(label)
                        samples_created += 1
                        
                    except Exception as e:
                        print(f"Error processing {img_name}: {e}")
                        continue
                
                angle_idx += 1
    
    return pd.Series(X_resampled), pd.Series(y_resampled)


def cut_undersample(dataset_dir: str, X: pd.Series, y: pd.Series, random: int = 42) -> Tuple[pd.Series, pd.Series]:
    """
    Undersample majority classes to match minority class count.
    
    Args:
        dataset_dir (str): Directory containing images
        X (pd.Series): Image filenames
        y (pd.Series): Labels
        random (int): Random state
    
    Returns:
        Tuple: X_resampled, y_resampled
    """
    np.random.seed(random)
    
    # Count classes
    label_counts = Counter(y)
    min_count = min(label_counts.values())
    
    X_resampled = []
    y_resampled = []
    
    for label in label_counts.keys():
        label_indices = y[y == label].index
        label_images = X.iloc[label_indices]
        
        # Sample min_count images from this class
        if len(label_images) > min_count:
            sampled_images = label_images.sample(n=min_count, random_state=random)
        else:
            sampled_images = label_images
        
        X_resampled.extend(sampled_images.tolist())
        y_resampled.extend([label] * len(sampled_images))
    
    return pd.Series(X_resampled), pd.Series(y_resampled)