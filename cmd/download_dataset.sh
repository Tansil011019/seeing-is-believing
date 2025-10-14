#!/bin/bash

# Download dataset script for skin disease classification
# This script installs gdown and downloads the dataset from Google Drive

set -e  # Exit on any error

echo "Checking for wget..."
if ! command -v wget &> /dev/null; then
 	echo "Error: wget is not installed. Please install wget and try again."
 	exit 1
fi

# Dataset URLs
t12_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip"
t3_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_Input.zip"
t12_val_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip"
t3_val_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Validation_Input.zip"
t12_test_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Test_Input.zip"
t3_test_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Test_Input.zip"

echo "Creating datasets directory if it doesn't exist..."
mkdir -p datasets

echo "Downloading datasets..."
for url in "${t12_url}" "${t3_url}" "${t12_val_url}" "${t3_val_url}" "${t12_test_url}" "${t3_test_url}"; do
 	echo "Downloading $url"
 	if ! wget -c "$url" -P datasets/; then
 		echo "Error: Failed to download $url"
 		exit 1
 	fi
done

echo "Looking for zip files to extract..."
# Find and extract any zip files in the datasets directory
find datasets/ -name "*.zip" -exec unzip -o {} -d datasets/ \;

# ground truth files
t1_gt_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip"
t2_gt_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task2_Training_GroundTruth_v3.zip"
t3_gt_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_GroundTruth.zip"
t1_val_gt_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip"
t2_val_gt_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task2_Validation_GroundTruth.zip"
t3_val_gt_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Validation_GroundTruth.zip"
t1_test_gt_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Test_GroundTruth.zip"
t2_test_gt_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task2_Test_GroundTruth.zip"
t3_test_gt_url="https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Test_GroundTruth.zip"

for url in "${t1_gt_url}" "${t2_gt_url}" "${t3_gt_url}" "${t1_val_gt_url}" "${t2_val_gt_url}" "${t3_val_gt_url}" "${t1_test_gt_url}" "${t2_test_gt_url}" "${t3_test_gt_url}"; do
 	echo "Downloading $url"
 	if ! wget -c "$url" -P datasets; then
 		echo "Error: Failed to download $url"
 		exit 1
 	fi
done

echo "Extracting ground truth files..."
# Find and extract any zip files in the datasets directory
find datasets/ -name "*GroundTruth.zip" -exec unzip -o {} -d datasets/ \;

echo "Dataset download and extraction completed!"
echo "Data is available in the datasets/ directory"