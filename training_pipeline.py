import os
import pandas as pd
import matplotlib.pyplot as plt
from utils.clean import clean_dataframe
from utils.preprocess import sample_and_split, rotative_oversample, cut_undersample

# Define dataset directory
dataset_dir = "datasets/ISIC2018_Task3_Training_Input"
ground_truth_file = "datasets/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv"
melt_output_file = "datasets/ISIC2018_Task3_Training_GroundTruth/melted_ground_truth.csv"

# 1. Sample and split the dataset
clean_dataframe(ground_truth_file, melt_output_file)

X_train, y_train, X_test, y_test = sample_and_split(dataset_dir,
                                                    melt_output_file, 
                                                    sample_size=1000, 
                                                    test_size=0.2)

# Create DataFrame for training data preview
df_train = pd.DataFrame({
    'image': X_train,
    'label': y_train
})

print("Sampled Training Data (first 5 rows):")
print(df_train.head())

# Visualize label distribution before resampling
plt.figure(figsize=(6,4))
df_train['label'].value_counts().plot(kind='bar')
plt.title('Label Distribution Before Resampling')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. Oversample minority classes via rotations
X_os, y_os = rotative_oversample(dataset_dir, X_train, y_train)
df_os = pd.DataFrame({'image': X_os, 'label': y_os})

print("Oversampled Data (first 5 rows):")
print(df_os.head())

plt.figure(figsize=(6,4))
df_os['label'].value_counts().plot(kind='bar')
plt.title('Label Distribution After Oversampling')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 3. Undersample majority classes
X_us, y_us = cut_undersample(dataset_dir, X_train, y_train)
df_us = pd.DataFrame({'image': X_us, 'label': y_us})

print("Undersampled Data (first 5 rows):")
print(df_us.head())

plt.figure(figsize=(6,4))
df_us['label'].value_counts().plot(kind='bar')
plt.title('Label Distribution After Undersampling')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()