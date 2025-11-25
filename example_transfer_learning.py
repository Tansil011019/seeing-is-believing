"""
Example usage of transfer learning pipeline for skin lesion segmentation

This script demonstrates various ways to use the transfer learning pipeline
with different models and configurations.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from vanilla_scripts.seg_transfer_learning_pipeline import TransferLearningPipeline
from utils import setup_logger


def example_1_quick_start():
    """
    Example 1: Quick start with Segformer and frozen encoder
    Best for: Quick experiments, limited compute, small datasets
    """
    print("\n" + "="*60)
    print("Example 1: Quick Start - Segformer with Frozen Encoder")
    print("="*60)
    
    cfg = {
        'aug_img': Path('./aug_img'),
        'aug_mask': Path('./aug_mask'),
        'image_folder': Path('datasets/ISIC2018_Task1-2_Training_Input'),
        'mask_folder': Path('datasets/ISIC2018_Task1_Training_GroundTruth'),
        'ckpt': Path('./checkpoints'),
        'bs': 8,
        'epochs': 30,  # Shorter for quick testing
        'lr': 1e-4,
        'num_workers': 4,
        'force_device': None,
        'track_metrics': True,
        'use_amp': True,
    }
    
    pipeline = TransferLearningPipeline(
        cfg,
        model_name='segformer',
        freeze_encoder=True
    )
    
    pipeline.preprocess(cfg['image_folder'], cfg['mask_folder'])
    pipeline.prepare_data()
    pipeline.build_model()
    best_iou = pipeline.train()
    
    print(f"Training completed! Best IoU: {best_iou:.4f}")


def example_2_full_finetuning():
    """
    Example 2: Full fine-tuning with BEiT for maximum performance
    Best for: Maximum accuracy, sufficient compute, larger datasets
    """
    print("\n" + "="*60)
    print("Example 2: Full Fine-tuning - BEiT Model")
    print("="*60)
    
    cfg = {
        'aug_img': Path('./aug_img'),
        'aug_mask': Path('./aug_mask'),
        'image_folder': Path('datasets/ISIC2018_Task1-2_Training_Input'),
        'mask_folder': Path('datasets/ISIC2018_Task1_Training_GroundTruth'),
        'ckpt': Path('./checkpoints'),
        'bs': 8,
        'epochs': 100,
        'lr': 1e-4,  # Lower learning rate for full fine-tuning
        'num_workers': 4,
        'force_device': None,
        'track_metrics': True,
        'use_amp': True,
    }
    
    pipeline = TransferLearningPipeline(
        cfg,
        model_name='beit',
        freeze_encoder=False  # Full fine-tuning
    )
    
    pipeline.preprocess(cfg['image_folder'], cfg['mask_folder'])
    pipeline.prepare_data()
    pipeline.build_model()
    best_iou = pipeline.train()
    
    print(f"Training completed! Best IoU: {best_iou:.4f}")


def example_3_adaptive_model():
    """
    Example 3: Adaptive model with domain-specific layers
    Best for: Domain adaptation, specialized medical imaging tasks
    """
    print("\n" + "="*60)
    print("Example 3: Adaptive Model with Custom Layers")
    print("="*60)
    
    cfg = {
        'aug_img': Path('./aug_img'),
        'aug_mask': Path('./aug_mask'),
        'image_folder': Path('datasets/ISIC2018_Task1-2_Training_Input'),
        'mask_folder': Path('datasets/ISIC2018_Task1_Training_GroundTruth'),
        'ckpt': Path('./checkpoints'),
        'bs': 8,
        'epochs': 50,
        'lr': 2e-4,  # Higher LR for adaptation layers
        'num_workers': 4,
        'force_device': None,
        'track_metrics': True,
        'use_amp': True,
    }
    
    pipeline = TransferLearningPipeline(
        cfg,
        model_name='adaptive_segformer',
        freeze_encoder=True  # Freeze base, train adaptation layers
    )
    
    pipeline.preprocess(cfg['image_folder'], cfg['mask_folder'])
    pipeline.prepare_data()
    pipeline.build_model()
    best_iou = pipeline.train()
    
    print(f"Training completed! Best IoU: {best_iou:.4f}")


def example_4_compare_models():
    """
    Example 4: Compare multiple transfer learning models
    Best for: Model selection, finding best architecture for your data
    """
    print("\n" + "="*60)
    print("Example 4: Compare Multiple Models")
    print("="*60)
    
    models_to_compare = ['segformer', 'mit_b1', 'beit']
    results = {}
    
    cfg = {
        'aug_img': Path('./aug_img'),
        'aug_mask': Path('./aug_mask'),
        'image_folder': Path('datasets/ISIC2018_Task1-2_Training_Input'),
        'mask_folder': Path('datasets/ISIC2018_Task1_Training_GroundTruth'),
        'ckpt': Path('./checkpoints'),
        'bs': 8,
        'epochs': 30,
        'lr': 1e-4,
        'num_workers': 4,
        'force_device': None,
        'track_metrics': True,
        'use_amp': True,
    }
    
    # Preprocess data once
    first_pipeline = TransferLearningPipeline(cfg, model_name=models_to_compare[0])
    first_pipeline.preprocess(cfg['image_folder'], cfg['mask_folder'])
    
    # Train each model
    for model_name in models_to_compare:
        print(f"\nTraining {model_name}...")
        pipeline = TransferLearningPipeline(
            cfg,
            model_name=model_name,
            freeze_encoder=True
        )
        
        pipeline.prepare_data()
        pipeline.build_model()
        best_iou = pipeline.train()
        results[model_name] = best_iou
    
    # Print comparison
    print("\n" + "="*60)
    print("Model Comparison Results")
    print("="*60)
    for model_name, best_iou in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:20s}: {best_iou:.4f}")


def example_5_custom_config():
    """
    Example 5: Custom configuration for specific use case
    Shows how to customize hyperparameters
    """
    print("\n" + "="*60)
    print("Example 5: Custom Configuration")
    print("="*60)
    
    cfg = {
        'aug_img': Path('./aug_img'),
        'aug_mask': Path('./aug_mask'),
        'image_folder': Path('datasets/ISIC2018_Task1-2_Training_Input'),
        'mask_folder': Path('datasets/ISIC2018_Task1_Training_GroundTruth'),
        'ckpt': Path('./checkpoints/custom_experiment'),
        'bs': 16,  # Larger batch size
        'epochs': 75,  # Custom epoch count
        'lr': 5e-5,  # Conservative learning rate
        'num_workers': 8,
        'force_device': 'cuda',  # Force GPU
        'track_metrics': True,
        'use_amp': True,
    }
    
    # Custom logger
    logger = setup_logger(name='custom_experiment', log_file='logs/custom_experiment.log')
    
    pipeline = TransferLearningPipeline(
        cfg,
        model_name='segformer',
        freeze_encoder=False,  # Full fine-tuning
        logger=logger
    )
    
    pipeline.preprocess(cfg['image_folder'], cfg['mask_folder'])
    pipeline.prepare_data()
    pipeline.build_model()
    best_iou = pipeline.train()
    
    print(f"Custom experiment completed! Best IoU: {best_iou:.4f}")


if __name__ == "__main__":
    # Run the example you want to test
    # Uncomment one of the following:
    
    example_1_quick_start()
    # example_2_full_finetuning()
    # example_3_adaptive_model()
    # example_4_compare_models()
    # example_5_custom_config()
