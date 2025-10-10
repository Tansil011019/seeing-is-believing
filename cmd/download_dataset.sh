#!/bin/bash

# Download dataset script for skin disease classification
# This script installs gdown and downloads the dataset from Google Drive

set -e  # Exit on any error

echo "Installing gdown if not already installed..."
if ! command -v uv &> /dev/null; then
	echo "Error: 'uv' environment is not initiated. Please initiate it and try again."
	exit 1
fi

uv pip install gdown

echo "Creating data directory if it doesn't exist..."
mkdir -p data

echo "Downloading dataset from Google Drive..."
# Download the entire folder from Google Drive
if ! gdown --folder https://drive.google.com/drive/folders/$GDRIVE_DATASET_FOLDER_ID -O data/; then
	echo "Error: Failed to download dataset from Google Drive."
	exit 1
fi

echo "Looking for zip files to extract..."
# Find and extract any zip files in the data directory
find data/ -name "*.zip" -exec unzip -o {} -d data/ \;

echo "Dataset download and extraction completed!"
echo "Data is available in the data/ directory"