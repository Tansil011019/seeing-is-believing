# Seeing is Believing
WORK IN PROGRESS README
A Comparative Analysis of Visual Explanations from for Skin Cancer Diagnosis

## Overview

This project implements and compares various deep learning architectures for skin disease classification, with a focus on explainable AI through visual explanations and heatmap analysis. The implementation includes state-of-the-art models including HI_MViT, MedKAN, and traditional CNN architectures.

### 1. General Preprocessing
**Data Augmentation:**
- Automatically augments each image with 24 variations:
  - 7 rotations (45°, 90°, 135°, 180°, 225°, 270°, 315°)
  - 2 dilation variations per rotation (1×1.5, 1.5×1)
  - 2 dilation variations on original image
- **Smart skip**: If augmented data exists with 24× original files, augmentation is skipped automatically
- Output: `"Augmented images found, skipping augmentation process"`

### 2. Segmentation
[WIP]


## Features

- **Feature 1**:
  - Hi i am a description

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
- Download all datasets from the ISIC2018 bucket
- Extract any zip files to the `datasets/` directory

## Training

### Segmentation Training Pipeline

#### Segmentation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--image_folder` | str | `datasets/ISIC2018_Task1-2_Training_Input` | Path to original input images directory |
| `--mask_folder` | str | `datasets/ISIC2018_Task1_Training_GroundTruth` | Path to ground truth segmentation masks directory |
| `--aug_image_folder` | str | `./aug_img` | Path to save augmented images |
| `--aug_mask_folder` | str | `./aug_mask` | Path to save augmented masks |
| `--ckpt` | str | `./checkpoints` | Checkpoint directory for saving/loading models |
| `--batch_size` | int | `8` | Batch size for training (BS) |
| `--num_epochs` | int | `100` | Number of training epochs (E) |
| `--learning_rate` | float | `1e-4` | Adam optimizer learning rate (LR) |
| `--num_workers` | int | `4` | Parallel data loading workers. Use `-1` for all available CPUs (W) |
| `--visible_cuda_devices` | str | None | GPU device indices to use (e.g., "0,1,2"). Only effective with `--force_device cuda` |
| `--force_device` | str | None | Force device: "cuda" for GPU, "cpu" for CPU. Default: auto-detect |

#### Usage Examples

**Basic training with defaults:**
```bash
python seg_pipeline.py
```

**Resume training from checkpoint:**
```bash
python seg_pipeline.py \
    --ckpt ./checkpoints \
    --image_folder data/images \
    --mask_folder data/masks
```

**Force GPU training with specific devices:**
```bash
python seg_pipeline.py \
    --force_device cuda \
    --visible_cuda_devices 0,1
```



## Deployments

### Run Demo
[WIP]
```bash
python dashboard.py
```

## Model Architectures

### Model 1A : CNN Based model
A CNN model inspired by the ISIC2018 task 1 champion.
[WIP]

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
[WIP]

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


## Skin Disease Classes

Typical skin disease classification includes:
1. Melanoma
2. Melanocytic nevi
3. Basal cell carcinoma
4. Actinic keratoses
5. Benign keratosis-like lesions
6. Dermatofibroma
7. Vascular lesions


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
[WIP]
If you use this code in your research, please cite:

```bibtex
@misc{seeing-is-believing-2025,
  title={Seeing is Believing: A Comparative Analysis of Visual Explanations from CNN and ViT Architectures for Skin Cancer Diagnosis},
  author={},
  year={2025},
  url={https://github.com/Tansil011019/seeing-is-believing}
}
```

## Acknowledgments
[WIP]
- Based on HI_MViT paper by Ding et al. (2023)
- MedKAN architecture inspired by Kolmogorov-Arnold Networks
- Built with TensorFlow and Keras
