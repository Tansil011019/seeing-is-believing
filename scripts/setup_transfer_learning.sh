#!/bin/bash
# Setup script for transfer learning pipeline
# Installs required dependencies for HuggingFace models

echo "Setting up transfer learning environment..."

# Install transformers and accelerate for optimal performance
pip install transformers>=4.30.0
pip install accelerate>=0.20.0

# Install additional dependencies for specific models
pip install timm>=0.9.0  # For vision transformers

echo "Setup complete! You can now run transfer learning training:"
echo ""
echo "Examples:"
echo "  python seg_transfer_learning_pipeline.py --model segformer --freeze_encoder"
echo "  python seg_transfer_learning_pipeline.py --model beit --track_metrics"
echo "  python seg_transfer_learning_pipeline.py --model all --freeze_encoder"
