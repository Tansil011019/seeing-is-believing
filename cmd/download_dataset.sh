#!/bin/bash

# Download dataset script for skin disease classification
# This script installs gdown and downloads the dataset from Google Drive

set -e  # Exit on any error

echo "Installing gdown if not already installed..."
pip install gdown

echo "Creating data directory if it doesn't exist..."
mkdir -p data

echo "Downloading dataset from Google Drive..."
# Download the entire folder from Google Drive
gdown --folder https://drive.google.com/drive/folders/1XP38r7Aytadj2nUbiFfhRbmIQC9x-HOx -O data/

echo "Looking for zip files to extract..."
# Find and extract any zip files in the data directory
find data/ -name "*.zip" -exec unzip -o {} -d data/ \;

echo "Dataset download and extraction completed!"
echo "Data is available in the data/ directory"