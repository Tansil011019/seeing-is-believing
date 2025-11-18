"""
Transfer Learning Pipeline for Attribute Detection
Multi-label classification for skin lesion attributes (ISIC 2018 Task 2)

This pipeline provides transfer learning capabilities for multi-label attribute
detection using state-of-the-art pretrained models. It supports multiple 
architectures optimized for different use cases:

Available Models:
    ResNet Variants:
        - resnet18: ResNet-18 (lightweight, fast)
        - resnet34: ResNet-34 (balanced)
        - resnet50: ResNet-50 (high capacity)
    
    Efficient Models:
        - efficientvim: EfficientViM (Mamba-based)
        - efficientnet_b0: EfficientNet-B0
    
    Vision Transformers:
        - ecvit: Efficient Compact ViT (DeiT-Tiny)
        - vit_tiny: ViT-Tiny

Attributes (5 total):
    - globules
    - milia_like_cyst
    - negative_network
    - pigment_network
    - streaks

Training Strategy:
    - Multi-label classification with BCEWithLogitsLoss
    - Mixed precision training (AMP) for efficiency
    - Discriminative learning rates for backbone vs classifier

Usage Examples:
    # Train single model
    python attr_transfer_learning_pipeline.py --model resnet18
    
    # Train all models with metrics tracking
    python attr_transfer_learning_pipeline.py --model all --track_metrics
    
    # Custom training
    python attr_transfer_learning_pipeline.py --model ecvit --num_epochs 50 --batch_size 32
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any
import csv
from datetime import datetime

# Project imports
from preprocessing.attr_dataset import AttributeDataset, ATTRIBUTE_LABELS
from models.attr_factory import get_attr_model, get_available_attr_models
from evaluation.attr_metrics import (
    evaluate_attr_model, 
    compute_multilabel_metrics
)
from utils import setup_logger, get_num_workers

# Constants
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class AttributeTransferLearningPipeline:
    """
    Transfer learning pipeline for multi-label attribute detection
    Supports multiple pretrained model architectures
    """
    
    def __init__(self, 
                 cfg: Dict[str, Any], 
                 model_name: str = 'resnet18', 
                 logger: Optional[Any] = None):
        
        self.cfg = cfg
        self.model_name = model_name
        self.logger = logger or setup_logger(name='attr_transfer_pipe')
        
        # Device configuration
        if cfg.get('force_device'):
            self.device = torch.device(cfg['force_device'])
            self.logger.info(f"Forcing device: {self.device}")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Auto-detected device: {self.device}")
        
        self.num_workers = get_num_workers(cfg.get('num_workers', 4))
        self.logger.info(f"Using {self.num_workers} data loader workers")
        
        # Checkpoint management
        self.best_f1 = 0.0
        self.ckpt_dir: Path = self.cfg['ckpt'] / f'attr_transfer_{model_name}'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_path: Path = self.ckpt_dir / 'best.pth'
        
        # Metrics tracking
        self.track_metrics = cfg.get('track_metrics', False)
        if self.track_metrics:
            self.metrics_dir: Path = Path('outputs')
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.metrics_file: Path = self.metrics_dir / f'attr_transfer_{model_name}_metrics_{timestamp}.csv'
            self._init_metrics_file()
        
       
    
    def _init_metrics_file(self):
        """Initialize CSV file for tracking metrics"""
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'val_loss', 
                'val_f1_macro', 'val_f1_micro', 'val_f1_weighted',
                'val_accuracy', 'val_precision_macro', 'val_recall_macro',
                'val_hamming_loss', 'val_subset_accuracy',
                'learning_rate'
            ] + [f'val_f1_{label}' for label in ATTRIBUTE_LABELS])
        self.logger.info(f"Metrics will be tracked in: {self.metrics_file}")
    
    def _log_metrics(self, epoch: int, train_loss: float, metrics: Dict[str, float], lr: float):
        """Log metrics to CSV file"""
        print("DASDWADD")
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                epoch,
                f"{train_loss:.6f}",
                f"{metrics.get('loss', 0.0):.6f}",
                f"{metrics.get('f1_macro', 0.0):.6f}",
                f"{metrics.get('f1_micro', 0.0):.6f}",
                f"{metrics.get('f1_weighted', 0.0):.6f}",
                f"{metrics.get('accuracy', 0.0):.6f}",
                f"{metrics.get('precision_macro', 0.0):.6f}",
                f"{metrics.get('recall_macro', 0.0):.6f}",
                f"{metrics.get('hamming_loss', 0.0):.6f}",
                f"{metrics.get('subset_accuracy', 0.0):.6f}",
                f"{lr:.8f}"
            ]
            # Add per-class F1 scores
            for label in ATTRIBUTE_LABELS:
                row.append(f"{metrics.get(f'f1_{label}', 0.0):.6f}")
            writer.writerow(row)
    
    def prepare_data(self):
        """Create train and validation DataLoaders from CSV labels"""
        self.logger.info("Preparing data loaders...")
        
        # Training dataset
        train_ds = AttributeDataset(
            str(self.cfg['image_folder']),
            str(self.cfg['train_csv']),
            image_size=(224, 224),
            normalize=True
        )
        
        if len(train_ds) == 0:
            self.logger.error("Training dataset is empty. Check CSV file and image folder.")
            raise ValueError("Cannot train on an empty dataset.")
        
        self.logger.info(f"Training dataset size: {len(train_ds)}")
        
        # Log label distribution
        label_dist = train_ds.get_label_distribution()
        self.logger.info("Training label distribution:")
        for label, count in label_dist.items():
            percentage = (count / len(train_ds)) * 100
            self.logger.info(f"  {label}: {count}/{len(train_ds)} ({percentage:.1f}%)")
        
        # Validation dataset
        val_ds = AttributeDataset(
            str(self.cfg['val_image_folder']),
            str(self.cfg['val_csv']),
            image_size=(224, 224),
            normalize=True
        )
        
        if len(val_ds) == 0:
            self.logger.error("Validation dataset is empty. Check CSV file and image folder.")
            raise ValueError("Cannot validate on an empty dataset.")
        
        self.logger.info(f"Validation dataset size: {len(val_ds)}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_ds,
            self.cfg['bs'],
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_ds,
            self.cfg['bs'],
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        self.logger.info("Done preparing data loaders.\n")
    
    def build_model(self):
        """Initialize attribute detection model, optimizer, scheduler, and loss"""
        self.logger.info(f"Building attribute detection model: {self.model_name}...")
        
        try:
            self.model = get_attr_model(self.model_name, num_classes=5, pretrained=True)
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
        
        self.model.to(self.device)
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Optimizer - use AdamW with weight decay
        self.opt = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg['lr'],
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.sch = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='max', patience=5, factor=0.5
        )
        
        # Loss function for multi-label classification
        self.crit = nn.BCEWithLogitsLoss()
        
        self.logger.info(f"Model '{self.model_name}' built successfully.\n")
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model on the validation set
        
        Returns:
            Dictionary containing validation metrics
        """
        self.logger.info("Running validation...")
        self.model.eval()
        metrics = evaluate_attr_model(
            self.model,
            self.val_loader,
            self.crit,
            self.device,
        )
        
        self.logger.info(
            f"Validation F1 (macro): {metrics['f1_macro']:.4f}, "
            f"F1 (micro): {metrics['f1_micro']:.4f}"
        )
        
        return metrics
    
    def train(self):
        """Main training and validation loop"""
        self.logger.info(f"--- Starting Attribute Detection Training for {self.model_name} ---")
        self.logger.info(f"Epochs: {self.cfg['epochs']}")
        self.logger.info(f"Batch Size: {self.cfg['bs']}")
        self.logger.info(f"Learning Rate: {self.cfg['lr']}")
        self.logger.info(f"Device: {self.device}")
        
        for ep in range(1, self.cfg['epochs'] + 1):
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(
                self.train_loader,
                desc=f"[{self.model_name}] Ep {ep}/{self.cfg['epochs']} [Train]"
            )
            
            for b in pbar:
                imgs = b['image'].to(self.device, non_blocking=True)
                labels = b['labels'].to(self.device, non_blocking=True)
                
                self.opt.zero_grad(set_to_none=True)
                
                logits = self.model(imgs)
                loss = self.crit(logits, labels)
                
                loss.backward()
                self.opt.step()
                
                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_train_loss = train_loss / len(self.train_loader)
            
            # Validation
            m = self.validate()
            val_f1_macro = m['f1_macro']
            
            # Get current learning rate
            current_lr = self.opt.param_groups[0]['lr']
            
            self.logger.info(
                f"[{self.model_name}] Epoch {ep:03d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val F1 (macro): {val_f1_macro:.4f} | "
                f"Val F1 (micro): {m['f1_micro']:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Log metrics to CSV
            self._log_metrics(ep, avg_train_loss, m, current_lr)
            
            # Update learning rate scheduler
            self.sch.step(val_f1_macro)
            
            # Save best model
            if val_f1_macro > self.best_f1:
                self.best_f1 = val_f1_macro
                self.logger.info(
                    f"✨ New best F1 (macro): {self.best_f1:.4f}. "
                    f"Saving model to {self.best_ckpt_path}"
                )
                
                save_data = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'best_val_f1': self.best_f1,
                    'epoch': ep,
                    'model_name': self.model_name,
                }
                torch.save(save_data, self.best_ckpt_path)
        
        self.logger.info(
            f"[{self.model_name}] Training completed. "
            f"Best F1 (macro) achieved: {self.best_f1:.4f}\n"
        )
        return self.best_f1


def main():
    """
    Attribute Detection Transfer Learning Pipeline CLI
    
    Uses pretrained models for multi-label attribute classification.
    Supports: ResNet variants, EfficientViM/Net, ViT-based models.
    
    Examples:
        # Train with ResNet-18
        python attr_transfer_learning_pipeline.py --model resnet18
        
        # Train with ECViT
        python attr_transfer_learning_pipeline.py --model ecvit
        
        # Train all models
        python attr_transfer_learning_pipeline.py --model all
        
        # Custom training with metrics tracking
        python attr_transfer_learning_pipeline.py \\
            --model resnet50 \\
            --batch_size 32 \\
            --num_epochs 50 \\
            --learning_rate 1e-4 \\
            --track_metrics
    """
    
    p = argparse.ArgumentParser(
        description='Attribute detection transfer learning pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input/Output paths
    p.add_argument(
        '--image_folder', type=str,
        default="datasets/ISIC2018_Task1-2_Training_Input",
        help='Path to training input images (default: %(default)s)'
    )
    p.add_argument(
        '--train_csv', type=str,
        default="datasets/ISIC2018_Task2_Training_GroundTruth.csv",
        help='Path to training CSV labels (default: %(default)s)'
    )
    p.add_argument(
        '--val_image_folder', type=str,
        default="datasets/ISIC2018_Task1-2_Validation_Input",
        help='Path to validation images (default: %(default)s)'
    )
    p.add_argument(
        '--val_csv', type=str,
        default="datasets/ISIC2018_Task2_Validation_GroundTruth.csv",
        help='Path to validation CSV labels (default: %(default)s)'
    )   
    p.add_argument(
        '--ckpt', type=str,
        default="./checkpoints",
        help='Checkpoint directory (default: %(default)s)'
    )
    
    # Model selection
    available_models = get_available_attr_models()
    p.add_argument(
        '--model', type=str, default='resnet18',
        help=f'Attribute detection model. Options: {", ".join(available_models)}, or "all" (default: %(default)s)'
    )
    
    # Training hyperparameters
    p.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size (default: %(default)s)'
    )
    p.add_argument(
        '--num_epochs', type=int, default=50,
        help='Number of epochs (default: %(default)s)'
    )
    p.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help='Learning rate (default: %(default)s)'
    )
    
    # Training options
    p.add_argument(
        '--use_amp', action='store_true', default=True,
        help='Use Automatic Mixed Precision for faster training (default: True)'
    )
    p.add_argument(
        '--no_amp', action='store_false', dest='use_amp',
        help='Disable Automatic Mixed Precision'
    )
    p.add_argument(
        '--track_metrics', action='store_true',
        help='Track metrics in CSV file'
    )
    
    # System configuration
    p.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers (default: %(default)s)'
    )
    p.add_argument(
        '--visible_cuda_devices', type=str, default=None,
        help='GPU device indices (e.g., "0,1,2")'
    )
    p.add_argument(
        '--force_device', type=str, default=None,
        choices=['cuda', 'cpu'],
        help='Force specific device (default: auto-detect)'
    )
    
    # Preprocessing options
    p.add_argument(
        '--preprocess', action='store_true',
        help='Run preprocessing to create CSV labels from binary images'
    )
    p.add_argument(
        '--threshold', type=float, default=0.05,
        help='Threshold for positive label in preprocessing (default: 0.05)'
    )
    
    args = p.parse_args()
    
    # Set visible CUDA devices
    if args.visible_cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cuda_devices
    
    # Run preprocessing if requested
    if args.preprocess:
        from preprocessing.attr_preprocess import create_attr_csv
        
        print("\n" + "="*60)
        print("Running Preprocessing")
        print("="*60 + "\n")
        
        # Training set
        create_attr_csv(
            "datasets/ISIC2018_Task2_Training_GroundTruth_v3",
            args.train_csv,
            threshold=args.threshold,
            verbose=True
        )
        
        # Validation set
        create_attr_csv(
            "datasets/ISIC2018_Task2_Validation_GroundTruth",
            args.val_csv,
            threshold=args.threshold,
            verbose=True
        )
        
        print("\n" + "="*60)
        print("Preprocessing Complete!")
        print("="*60 + "\n")
    
    # Determine which models to train
    available_models = get_available_attr_models()
    if args.model == 'all':
        models_to_train = available_models
    elif args.model in available_models:
        models_to_train = [args.model]
    else:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {', '.join(available_models)}, or 'all'")
        return
    
    print(f"\n{'='*60}")
    print(f"Attribute Detection Transfer Learning Pipeline")
    print(f"Models to train: {', '.join(models_to_train)}")
    print(f"{'='*60}\n")
    
    # Configuration dictionary
    cfg = {
        'image_folder': Path(args.image_folder),
        'train_csv': Path(args.train_csv),
        'val_image_folder': Path(args.val_image_folder),
        'val_csv': Path(args.val_csv),
        'ckpt': Path(args.ckpt),
        'bs': args.batch_size,
        'epochs': args.num_epochs,
        'lr': args.learning_rate,
        'num_workers': args.num_workers,
        'force_device': args.force_device,
        'track_metrics': args.track_metrics,
        'use_amp': args.use_amp,
    }
    
    # Create necessary directories
    try:
        cfg['ckpt'].mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating directories: {e}")
        return
    
    # Verify CSV files and image folders exist
    if not cfg['train_csv'].exists():
        print(f"Error: Training CSV not found: {cfg['train_csv']}")
        print("Run with --preprocess flag to generate CSV files")
        return
    if not cfg['val_csv'].exists():
        print(f"Error: Validation CSV not found: {cfg['val_csv']}")
        print("Run with --preprocess flag to generate CSV files")
        return
    if not cfg['image_folder'].exists():
        print(f"Error: Training image folder not found: {cfg['image_folder']}")
        return
    if not cfg['val_image_folder'].exists():
        print(f"Error: Validation image folder not found: {cfg['val_image_folder']}")
        return
    
    # Train models
    results = {}
    for idx, model_name in enumerate(models_to_train):
        print(f"\n{'='*60}")
        print(f"Training model: {model_name} ({idx+1}/{len(models_to_train)})")
        print(f"{'='*60}\n")
        
        try:
            pipeline = AttributeTransferLearningPipeline(
                cfg,
                model_name=model_name,
            )
            
            pipeline.prepare_data()
            pipeline.build_model()
            best_f1 = pipeline.train()
            results[model_name] = best_f1
            
        except Exception as e:
            logger = getattr(pipeline, 'logger', None) if 'pipeline' in locals() else None
            if logger:
                logger.error(f"Pipeline failed for model '{model_name}': {e}", exc_info=True)
            else:
                print(f"Error training model '{model_name}': {e}")
                import traceback
                traceback.print_exc()
            results[model_name] = None
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY - Attribute Detection Models")
    print(f"{'='*60}")
    for model_name, best_f1 in results.items():
        if best_f1 is not None:
            print(f"{model_name:25s}: Best F1 (macro) = {best_f1:.4f}")
        else:
            print(f"{model_name:25s}: FAILED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
