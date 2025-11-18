#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Full PyTorch Implementation of the "Ensembling CNN for Skin Cancer Classification" Paper.

This script includes all major phases described in the paper:
1.  **Preprocessing:** "Shades of Gray" color constancy.
2.  **Augmentation:** The specific training-time augmentation pipeline.
3.  **Data Splitting:** 5-Fold Stratified Group K-Fold.
4.  **Modeling (L0):** Training loop to generate Out-of-Sample (OOF) predictions.
5.  **Modeling (L1):** Training an XGBoost meta-model on the OOF predictions.
6.  **Evaluation:** Test Time Augmentation (TTA) with 24 predictions per image.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from torchvision.transforms import functional as F

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier
from tqdm import tqdm
import os
import warnings

# --- Global Settings ---
# Suppress warnings
warnings.filterwarnings("ignore")

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Model & Data Settings ---
# Model settings from the paper
MODEL_INPUT_SIZE = 224
# "resize all images so the short side is 1.25x larger"
RESIZE_SIZE = int(MODEL_INPUT_SIZE * 1.25)
N_FOLDS = 5
N_CLASSES = 7  # 7 diagnoses
RANDOM_STATE = 42

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# #############################################################################
# PHASE 1: PREPROCESSING & AUGMENTATION
# #############################################################################

class ShadesOfGrayTransform:
    """
    Implements the "Shades of Gray" color constancy algorithm from the paper.
    "using the Shades of Gray method ... with Minkowski norm p=6"
    
    This is a custom transform to be used with torchvision.
    """
    def __init__(self, p=6):
        self.p = p

    def __call__(self, img):
        # Convert PIL Image to numpy array
        img_np = np.array(img).astype(np.float32)
        
        # Avoid division by zero if image is black
        if img_np.max() == 0:
            return img

        # Apply Minkowski norm (p=6) for each channel
        # ill = (E[I^p])^(1/p)
        illuminant = np.power(np.mean(np.power(img_np, self.p), axis=(0, 1)), 1/self.p)

        # "White-balance" the image by dividing by the illuminant estimate
        # We also need to clip to [0, 255] and convert back to uint8
        img_balanced = (img_np / (illuminant + 1e-6)) * (np.mean(illuminant) + 1e-6)
        img_balanced = np.clip(img_balanced, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        return Image.fromarray(img_balanced)

def get_data_transforms(dataset_mean, dataset_std):
    """
    Returns the training and validation transforms.
    """
    
    # --- Training Augmentation Pipeline ---
    # As described in Section III. METHODS
    train_transform = transforms.Compose([
        # "perform colour constancy ... as a preprocessing step"
        ShadesOfGrayTransform(p=6),
        
        # "resize all images so the short side is 1.25x larger"
        transforms.Resize(RESIZE_SIZE),
        
        # "Next a random square crop with the size in [0.8, 1.0]"
        # "and resized to the desired input size of the model"
        transforms.RandomResizedCrop(MODEL_INPUT_SIZE, scale=(0.8, 1.0)),
        
        # "random horizontal flips"
        transforms.RandomHorizontalFlip(),
        
        # "random rotations of [0, 90, 180, 270] degrees"
        # We use RandomChoice to pick one of the 4 discrete rotations
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270)),
        ]),
        
        # "augment brightness, saturation and contrasts by a random factor in the range [0.9, 1.1]"
        # A factor of 0.1 corresponds to the [0.9, 1.1] range
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        
        transforms.ToTensor(),
        
        # "and the mean used for normalization" (using custom mean/std)
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

    # --- Validation Augmentation Pipeline ---
    # "During validation ... we simply take a center crop at 0.9" (from TTA section)
    # Note: The TTA section (Page 3) mentions a 0.9 center crop for validation.
    val_transform = transforms.Compose([
        ShadesOfGrayTransform(p=6),
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(int(RESIZE_SIZE * 0.9)), # 0.9 crop
        transforms.Resize(MODEL_INPUT_SIZE), # Resize to final
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])
    
    return train_transform, val_transform


# #############################################################################
# DUMMY DATASET (Replace with your own)
# #############################################################################

class DummySkinDataset(Dataset):
    """
    This is a dummy dataset.
    Replace __init__ and __getitem__ with your actual data loading logic.
    
    It MUST return:
    - image (PIL Image)
    - label (int)
    - lesion_id (int or str)
    - dataset_origin (int)
    """
    def __init__(self, num_samples=1000, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        
        # Create dummy metadata (replace this)
        self.metadata = pd.DataFrame({
            'image_id': [f'img_{i}.jpg' for i in range(num_samples)],
            'label': np.random.randint(0, N_CLASSES, num_samples),
            'lesion_id': np.random.randint(0, num_samples // 5, num_samples),
            'dataset_origin': np.random.randint(0, 3, num_samples) # e.g., 0=ISIC, 1=HAM, 2=Proprietary
        })
        
        self.image_paths = self.metadata['image_id'].values
        self.labels = self.metadata['label'].values
        self.lesion_ids = self.metadata['lesion_id'].values
        self.dataset_origins = self.metadata['dataset_origin'].values

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # --- Replace this with your data loading ---
        # 1. Load image from self.image_paths[idx]
        #    image = Image.open(self.image_paths[idx]).convert("RGB")
        # For dummy data, create a random PIL image
        image = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
        # --- End of dummy data ---
        
        label = self.labels[idx]
        lesion_id = self.lesion_ids[idx]
        dataset_origin = self.dataset_origins[idx]

        if self.transform:
            image = self.transform(image)
        
        # We return more than just image/label for the meta-model
        return image, label, lesion_id, dataset_origin, idx

    def get_labels(self):
        return self.labels

    def get_groups(self):
        return self.lesion_ids

    def get_meta_features(self):
        # "one-hot encoded categorical feature which encodes the dataset of origin"
        # We'll do the one-hot encoding later
        return self.dataset_origins

def calculate_mean_std(dataset):
    """
    Calculates the mean and std of the dataset for normalization.
    This should be run *once* before training.
    """
    # This is a placeholder. 
    # You should run the one-pass algorithm to calculate this.
    print("Using placeholder mean/std. You MUST calculate this for your dataset.")
    dataset_mean = [0.485, 0.456, 0.406]  # ImageNet mean
    dataset_std = [0.229, 0.224, 0.225]   # ImageNet std
    return dataset_mean, dataset_std

# #############################################################################
# PHASE 2: MODEL TRAINING & OOF PREDICTION
# #############################################################################

def get_model(n_classes=N_CLASSES):
    """
    Loads a pre-trained model and replaces the final layer.
    "All models have been initialized with weights obtained from pre-training on ImageNet"
    "we changed only ... the size of the last fully connected layer"
    """
    # We use ResNet50 as an example. The paper used multiple models.
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)
    
    return model.to(DEVICE)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        # Our custom dataset returns 5 items
        images, labels, _, _, _ = batch
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

def validate_one_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            images, labels, _, _, _ = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get probabilities for OOF
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # "The primary competition metric ... balanced accuracy"
    acc = balanced_accuracy_score(all_labels, np.argmax(all_preds, axis=1))
    
    return total_loss / len(loader), acc, all_preds


def main_cv_loop():
    """
    Main function to run the 5-fold cross-validation and
    generate out-of-sample (OOF) predictions.
    """
    
    # 1. Load the *full* dataset (without transforms)
    # We apply transforms later inside the loop
    full_dataset = DummySkinDataset(num_samples=1000, transform=None)
    
    # 2. Get data for splitting
    X = full_dataset.image_paths  # We just need a placeholder X
    y = full_dataset.get_labels() # "stratified by diagnosis"
    groups = full_dataset.get_groups() # "lesion identifiers are considered"
    
    # 3. Calculate Mean/Std (do this once)
    # You MUST calculate this properly on your *training data*
    # For now, we use a placeholder:
    dataset_mean, dataset_std = calculate_mean_std(full_dataset)
    train_transform, val_transform = get_data_transforms(dataset_mean, dataset_std)
    
    # 4. Initialize K-Fold Splitter
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    # 5. Initialize OOF arrays
    # "make out of sample predictions on the held out validation fold"
    oof_preds = np.zeros((len(full_dataset), N_CLASSES))
    oof_labels = np.zeros(len(full_dataset), dtype=int)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        print(f"\n{'='*20} FOLD {fold+1}/{N_FOLDS} {'='*20}")
        
        # 6. Create train/val datasets for this fold
        # We need to set the *correct* transform for each subset
        full_dataset.transform = train_transform
        train_subset = Subset(full_dataset, train_idx)
        
        full_dataset.transform = val_transform
        val_subset = Subset(full_dataset, val_idx)
        
        # 7. Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        # 8. Initialize model, optimizer, etc.
        model = get_model(n_classes=N_CLASSES)
        
        # "we changed only the initial learning rate"
        optimizer = optim.Adam(model.parameters(), lr=1e-4) 
        
        # "trained networks with a weighted loss functions"
        # (Optional) You can add class weights here if needed
        criterion = nn.CrossEntropyLoss()
        
        # --- Training Loop ---
        n_epochs = 10 # Example: set your number of epochs
        best_val_acc = 0
        best_fold_preds = None
        
        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_preds = validate_one_epoch(model, val_loader, criterion)
            
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_fold_preds = val_preds
                # torch.save(model.state_dict(), f"model_fold_{fold+1}.pth")
        
        # 9. Store OOF predictions
        oof_preds[val_idx] = best_fold_preds
        oof_labels[val_idx] = y[val_idx]
        
    # --- End of CV Loop ---
    # 10. Save OOF results
    np.save("oof_preds.npy", oof_preds)
    
    # 11. Report final OOF score
    total_oof_acc = balanced_accuracy_score(oof_labels, np.argmax(oof_preds, axis=1))
    print(f"\n{'='*20} CV FINISHED {'='*20}")
    print(f"Total OOF Balanced Accuracy: {total_oof_acc:.4f}")
    
    return oof_preds, full_dataset


# #############################################################################
# PHASE 3: META-MODEL TRAINING (STACKING)
# #############################################################################

def train_meta_model(oof_preds, dataset):
    """
    Trains the Level 1 XGBoost meta-model.
    "We treat these probabilities as features and train an XGBoost Classifier on them."
    """
    print(f"\n{'='*20} TRAINING L1 META-MODEL {'='*20}")
    
    # 1. Get the targets (true labels)
    y_meta = dataset.get_labels()
    
    # 2. Get the features
    # "We also include a one-hot encoded categorical feature which encodes the dataset of origin"
    dataset_origins = dataset.get_meta_features()
    origin_one_hot = pd.get_dummies(dataset_origins, prefix='origin').values
    
    # Combine OOF predictions + meta-features
    X_meta = np.hstack([oof_preds, origin_one_hot])
    
    print(f"Meta-model features shape: {X_meta.shape}")
    
    # 3. Train XGBoost
    meta_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=RANDOM_STATE
    )
    
    meta_model.fit(X_meta, y_meta)
    print("XGBoost meta-model trained.")
    
    # 4. Save the meta-model
    meta_model.save_model("xgb_meta_model.json")
    print("Meta-model saved to 'xgb_meta_model.json'")
    
    return meta_model


# #############################################################################
# PHASE 4: TEST TIME AUGMENTATION (TTA)
# #############################################################################

def predict_with_tta(model, image, tta_transform):
    """
    Performs the 24-prediction TTA as described in the paper.
    "proportional crops of [0.8, 0.9, 1.0]"
    "For each crop we perform all 8 combinations of 90 degree rotations and horizontal flips"
    3 crops * 8 transforms = 24 predictions
    """
    model.eval()
    all_preds = []
    
    # 1. Define the 8 transforms (4 rotations x 2 flips)
    rotations = [0, 90, 180, 270]
    flips = [False, True]
    
    with torch.no_grad():
        for crop_scale in [0.8, 0.9, 1.0]:
            # Create the 3 crops
            crop_size = int(RESIZE_SIZE * crop_scale)
            img_cropped = F.center_crop(image, crop_size)
            
            for rot in rotations:
                for flip in flips:
                    # Apply the 8 transforms
                    img_aug = img_cropped
                    if rot > 0:
                        img_aug = F.rotate(img_aug, rot)
                    if flip:
                        img_aug = F.hflip(img_aug)
                    
                    # Apply final transform (resize, ToTensor, Normalize)
                    tensor_aug = tta_transform(img_aug)
                    tensor_aug = tensor_aug.unsqueeze(0).to(DEVICE)
                    
                    # Get prediction
                    output = model(tensor_aug)
                    pred = torch.softmax(output, dim=1).cpu().numpy()
                    all_preds.append(pred)
    
    # "We ensemble these 24 predictions with a class-wise mean"
    final_prediction = np.mean(all_preds, axis=0)
    return final_prediction

def get_tta_transform(dataset_mean, dataset_std):
    """
    Minimal transform for TTA: resize, tensor, normalize.
    Cropping and augs are done manually in the TTA function.
    """
    return transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean, std=dataset_std)
    ])

# #############################################################################
# MAIN EXECUTION
# #############################################################################

if __name__ == "__main__":
    
    # --- PHASE 2 ---
    # Run the 5-fold CV loop to train L0 models and get OOF preds
    # oof_predictions, full_dataset_object = main_cv_loop()
    
    # --- PHASE 3 ---
    # Train the L1 meta-model (XGBoost)
    # meta_model = train_meta_model(oof_predictions, full_dataset_object)
    
    print("\n--- Full pipeline complete ---")
    print("To run TTA on a new image, you would:")
    print("1. Load your 5 trained fold-models.")
    print("2. Run `predict_with_tta()` for each model.")
    print("3. Average the 5 TTA predictions to get the L0 features.")
    print("4. Add the 'dataset_origin' one-hot features.")
    print("5. Feed these features into `meta_model.predict()`.")
    
    # --- Example of running TTA on a single dummy image ---
    print("\n--- Running TTA Example ---")
    
    # 1. Load a dummy model (replace with your *real* trained models)
    example_model = get_model().to(DEVICE)
    
    # 2. Get TTA transforms
    mean, std = calculate_mean_std(None) # Use placeholder mean/std
    tta_transform = get_tta_transform(mean, std)
    
    # 3. Load a dummy image (must be PIL)
    # "We center crop the resized image (1.25x larger..."
    dummy_pil_image = Image.fromarray(
        np.random.randint(0, 256, (RESIZE_SIZE, RESIZE_SIZE, 3), dtype=np.uint8)
    )
    
    # 4. Get 24-prediction average
    tta_pred = predict_with_tta(example_model, dummy_pil_image, tta_transform)
    
    print(f"TTA Prediction shape: {tta_pred.shape}")
    print(f"TTA probabilities: {tta_pred}")
    print(f"Final class: {np.argmax(tta_pred)}")