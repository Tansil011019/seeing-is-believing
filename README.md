# Seeing is Believing
A Comparative Analysis of Visual Explanations from CNN and ViT Architectures for Skin Cancer Diagnosis

## Overview

This project implements and compares various deep learning architectures for skin disease classification, with a focus on explainable AI through visual explanations and heatmap analysis. The implementation includes state-of-the-art models including HI_MViT, MedKAN, and traditional CNN architectures.

## Features

- **Multiple Model Architectures**:
  - HI_MViT (Hierarchical Inverted MobileViT) - Lightweight ViT variant
  - MedKAN (Medical Kolmogorov-Arnold Networks) - KAN-based medical imaging model
  - VGG16 - Classic CNN architecture
  - ResNet (50, 101, 152) - Residual networks

- **Evaluation Utils**:
  - Heatmap Intersection over Union (IoU) calculation
  - Visual heatmap comparison tools
  - Comprehensive training and evaluation pipeline

- **Easy-to-Use Training Pipeline**:
  - Automated data preprocessing
  - Model training with callbacks
  - Evaluation metrics calculation
  - Visualization tools

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tansil011019/seeing-is-believing.git
cd seeing-is-believing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

Download the skin disease dataset:

```bash
# On Linux/Mac
bash cmd/download_dataset.sh

# On Windows (Git Bash or WSL)
bash cmd/download_dataset.sh
```

This will:
- Install `gdown` for Google Drive downloads
- Download the dataset from the specified Google Drive folder
- Extract any zip files to the `data/` directory

## Quick Start

### Basic Usage Example

```python
from utils import (
    instantiate_model_hi_mvit,
    train_and_evaluate_model,
    calculate_heatmap_iou
)
from PIL import Image
import numpy as np

# Load your images and labels
images = [Image.open(f"path/to/image_{i}.jpg") for i in range(100)]
labels = np.array([0, 1, 2, ...])  # Your labels

# Instantiate a model
model = instantiate_model_hi_mvit(
    input_shape=(224, 224, 3),
    num_classes=7
)

# Train and evaluate
trained_model, confusion_matrix, metrics = train_and_evaluate_model(
    X=images,
    y=labels,
    model=model,
    test_ratio=0.2,
    epochs=50
)

print("Accuracy:", metrics['accuracy'])
```

### Run Demo

```bash
python example_usage.py
```

## Model Architectures

### HI_MViT (Hierarchical Inverted MobileViT)
Based on the paper "HI-MViT: A Lightweight Model for Explainable Skin Disease Classification Based on Modified MobileViT" by Ding et al. (2023). Combines the efficiency of MobileNet with the global reasoning capabilities of Vision Transformers.

### MedKAN (Medical Kolmogorov-Arnold Networks)
A novel KAN-based architecture designed for medical image classification, using learnable spline functions for enhanced feature representation.

### Traditional CNNs
- **VGG16**: Classic deep CNN with transfer learning support
- **ResNet**: Residual networks (50, 101, 152 layers) with skip connections

## Evaluation Tools

### Heatmap Analysis
```python
from utils import calculate_heatmap_iou, display_heatmap_intersection

# Calculate IoU between two heatmaps
iou = calculate_heatmap_iou(heatmap1, heatmap2, threshold=0.5)

# Visualize intersection
display_heatmap_intersection(heatmap1, heatmap2)
```

### Training Utilities
```python
from utils import train_and_evaluate_model

# Comprehensive training with built-in evaluation
model, cm, metrics = train_and_evaluate_model(
    X=images,
    y=labels,
    model=your_model,
    test_ratio=0.2,
    epochs=50,
    batch_size=32
)
```

## Project Structure

```
seeing-is-believing/
├── cmd/
│   └── download_dataset.sh         # Dataset download script
├── utils/
│   ├── __init__.py                # Package initialization
│   ├── evaluation.py              # Heatmap IoU and visualization
│   ├── training.py                # Training and evaluation utilities
│   ├── models_himvit.py           # HI_MViT implementation
│   ├── models_medkan.py           # MedKAN implementation
│   └── models_standard.py         # VGG16 and ResNet implementations
├── data/                          # Dataset directory (created after download)
├── example_usage.py               # Demo script
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Available Models

### Instantiate Models

```python
from utils import (
    instantiate_model_hi_mvit,
    instantiate_model_medkan,
    instantiate_model_vgg16,
    instantiate_model_resnet,
    get_best_resnet
)

# HI_MViT
model = instantiate_model_hi_mvit(num_classes=7)

# MedKAN
model = instantiate_model_medkan(num_classes=7)

# VGG16 with transfer learning
model = instantiate_model_vgg16(
    num_classes=7,
    weights='imagenet',
    freeze_base=False
)

# Best ResNet (ResNet152)
model = get_best_resnet(num_classes=7)

# Specific ResNet variant
model = instantiate_model_resnet(
    variant='resnet101',
    num_classes=7
)
```

## Skin Disease Classes

Typical skin disease classification includes:
1. Melanoma
2. Melanocytic nevi
3. Basal cell carcinoma
4. Actinic keratoses
5. Benign keratosis-like lesions
6. Dermatofibroma
7. Vascular lesions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{seeing-is-believing-2025,
  title={Seeing is Believing: A Comparative Analysis of Visual Explanations from CNN and ViT Architectures for Skin Cancer Diagnosis},
  author={Your Name},
  year={2025},
  url={https://github.com/Tansil011019/seeing-is-believing}
}
```

## Acknowledgments

- Based on HI_MViT paper by Ding et al. (2023)
- MedKAN architecture inspired by Kolmogorov-Arnold Networks
- Built with TensorFlow and Keras
