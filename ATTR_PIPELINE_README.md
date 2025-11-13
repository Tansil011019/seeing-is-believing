# Attribute Detection Transfer Learning Pipeline

## Overview
This module implements a multilabel classification pipeline for ISIC Task 2 - detecting 5 skin lesion attributes from dermoscopic images.

## Structure

### New Files Created

#### Core Pipeline
- **`attr_transfer_learning_pipeline.py`** (246 lines): Main training pipeline for attribute detection

#### Data Processing
- **`preprocessing/attr_preprocessing.py`** (137 lines): Functions to convert attribute masks to labels
  - `compute_attribute_label()`: Convert single mask to binary label based on threshold
  - `get_multilabel_for_image()`: Get labels for all 5 attributes for one image
  - `process_dataset_labels()`: Process entire dataset
  - `get_label_statistics()`: Compute dataset statistics
  - `FEAT_THRESHOLD = 0.1`: Threshold for feature presence (10% of pixels)
  - `ATTR_TYPES`: List of 5 attribute types

- **`preprocessing/attr_dataset.py`** (88 lines): PyTorch Dataset for multilabel classification
  - `AttributeDataset`: Loads images and generates multilabel vectors

#### Model Architecture
- **`models/attr_transfer_model.py`** (62 lines): Transfer learning model adapter
  - `AttributeTransferModel`: Adapts segmentation models to classification
  - Replaces segmentation head with global pooling + classifier

#### Evaluation
- **`evaluation/attr_metrics.py`** (65 lines): Multilabel classification metrics
  - `evaluate_multilabel()`: Computes F1 (micro/macro), hamming loss, exact match

#### Training
- **`training/attr_training.py`** (70 lines): Training utilities
  - `train_one_epoch()`: Single epoch training with AMP support

#### Exploration
- **`notebooks/attribute_exploration.ipynb`**: Interactive data exploration
  - Visualize attribute distributions
  - Analyze label co-occurrence
  - Verify data preprocessing

## Data Format

### Input
- Images: `datasets/ISIC2018_Task1-2_Training_Input/ISIC_{id}.jpg`
- GT Masks: `datasets/ISIC2018_Task2_Training_GroundTruth_v3/ISIC_{id}_attribute_{type}.png`

### Attributes (5 total)
1. globules
2. milia_like_cyst
3. negative_network
4. pigment_network
5. streaks

### Label Conversion
- For each attribute mask, calculate portion of positive pixels
- If portion > 0.1 (FEAT_THRESHOLD), label = 1, else label = 0
- Each image gets a 5-dimensional binary vector

## Usage

### Basic Training
```bash
python attr_transfer_learning_pipeline.py --model segformer --freeze_encoder
```

### Train All Models
```bash
python attr_transfer_learning_pipeline.py --model all --freeze_encoder --track_metrics
```

### Custom Configuration
```bash
python attr_transfer_learning_pipeline.py \
    --model beit \
    --batch_size 16 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --track_metrics
```

## Key Features

1. **Multilabel Classification**: Predicts 5 binary attributes simultaneously
2. **Transfer Learning**: Uses pretrained HuggingFace vision models
3. **Model Adaptation**: Converts segmentation models to classification
4. **Comprehensive Metrics**: F1-score, hamming loss, exact match accuracy
5. **Mixed Precision**: Automatic mixed precision (AMP) for faster training
6. **Flexible Architecture**: Support for multiple backbone models

## Architecture Changes

The pipeline modifies transfer learning models as follows:
1. Load pretrained backbone (Segformer, BEiT, MiT, etc.)
2. Remove segmentation head
3. Add global average pooling layer
4. Add classification head (256 hidden units, dropout 0.3)
5. Output: 5 logits for BCEWithLogitsLoss

## Metrics

- **F1-Score (Micro)**: Overall performance across all labels
- **F1-Score (Macro)**: Average performance per attribute
- **Hamming Loss**: Fraction of incorrectly predicted labels
- **Exact Match**: Percentage of samples with all labels correct

## Dependencies

All dependencies same as segmentation pipeline:
- PyTorch >= 1.9
- transformers >= 4.30.0
- scikit-learn (for metrics)
- OpenCV (for image loading)
- tqdm, numpy, pathlib

## Notes

- Feature threshold (0.1) is configurable via `FEAT_THRESHOLD` constant
- Pipeline supports all transfer models from `seg_models/transfer_models`
- Checkpoints saved to `checkpoints/attr_transfer_{model_name}/`
- Metrics logged to `outputs/attr_{model_name}_metrics_{timestamp}.csv`
