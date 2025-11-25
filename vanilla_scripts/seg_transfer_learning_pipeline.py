"""
Transfer Learning Pipeline for Segmentation
Leverages pretrained HuggingFace models for skin lesion segmentation

This pipeline provides transfer learning capabilities using state-of-the-art
pretrained models from HuggingFace. It supports multiple architectures optimized
for different use cases:

Available Models:
    - segformer: NVIDIA's Segformer-B0 pretrained on ADE20K (efficient, fast)
    - mit_b1: Mix Transformer B1 variant (balanced performance)
    - beit: Microsoft BEiT pretrained on ADE20K (high accuracy)
    - medsam2: Medical SAM2 for medical image segmentation
    - adaptive_segformer: Segformer with additional adaptation layers
    - adaptive_beit: BEiT with additional adaptation layers

Training Strategies:
    - Full fine-tuning: Better performance, requires more data and compute
    - Discriminative learning rates: Different LR for encoder/decoder when not frozen
    - Mixed precision training: Faster training with AMP (enabled by default)

Installation:
    pip install transformers>=4.30.0

Usage Examples:
    # Quick start with frozen encoder (recommended for first try)
    python seg_transfer_learning_pipeline.py --model segformer
    
    # Full fine-tuning for maximum performance
    python seg_transfer_learning_pipeline.py --model beit --num_epochs 100
    
    # Train all models with metrics tracking
    python seg_transfer_learning_pipeline.py --model all --track_metrics
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
from preprocessing import process_dataset_parallel, SegmentationDataset
from models.seg_factory import (
    get_transfer_model, 
    get_available_transfer_models
)
from evaluation import CombinedLoss, evaluate_model
from utils import setup_logger, get_num_workers

# Constants
AUGMENTATION_FACTOR = 4
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


class TransferLearningPipeline:
    """
    Transfer learning pipeline for segmentation using pretrained models
    Supports multiple backbone architectures from HuggingFace
    """
    
    def __init__(self, 
                 cfg: Dict[str, Any], 
                 model_name: str = 'segformer_b0', 
                 logger: Optional[Any] = None):
        
        self.cfg = cfg
        self.model_name = model_name
        self.logger = logger or setup_logger(name='transfer_pipe')
        
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
        self.best_iou = 0.0
        self.ckpt_dir: Path = self.cfg['ckpt'] / f'transfer_{model_name}'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_path: Path = self.ckpt_dir / 'best.pth'
        
        # Metrics tracking
        self.track_metrics = cfg.get('track_metrics', False)
        if self.track_metrics:
            self.metrics_dir: Path = Path('outputs')
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.metrics_file: Path = self.metrics_dir / f'transfer_{model_name}_metrics_{timestamp}.csv'
            self._init_metrics_file()
        
        # Mixed precision training for efficiency
        self.use_amp = cfg.get('use_amp', True) and (self.device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            self.logger.info("Using Automatic Mixed Precision (AMP) training")
    
    def _init_metrics_file(self):
        """Initialize CSV file for tracking metrics"""
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'val_loss', 'val_iou', 'val_dice',
                'val_accuracy', 'val_precision', 'val_recall', 'val_f1',
                'learning_rate'
            ])
        self.logger.info(f"Metrics will be tracked in: {self.metrics_file}")
    
    def _log_metrics(self, epoch: int, train_loss: float, metrics: Dict[str, float], lr: float):
        """Log metrics to CSV file"""
        if not self.track_metrics:
            return
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{metrics.get('loss', 0.0):.6f}",
                f"{metrics.get('iou', 0.0):.6f}",
                f"{metrics.get('dice', 0.0):.6f}",
                f"{metrics.get('accuracy', 0.0):.6f}",
                f"{metrics.get('precision', 0.0):.6f}",
                f"{metrics.get('recall', 0.0):.6f}",
                f"{metrics.get('f1', 0.0):.6f}",
                f"{lr:.8f}"
            ])
    
    def preprocess(self, img_f: Path, mask_f: Path):
        """Check for existing augmented data and run preprocessing if needed"""
        self.logger.info("Starting data preprocessing...")
        
        if self._check_augmented_data_exists(img_f):
            self.logger.info("Sufficient augmented data found. Skipping preprocessing.")
            return
        
        self.logger.info("No/incomplete augmented data found. Starting parallel processing...")
        process_dataset_parallel(
            str(img_f),
            str(mask_f),
            str(self.cfg['aug_img']),
            str(self.cfg['aug_mask']),
            True,
            self.num_workers,
            self.logger,
            output_size=(512, 512)
        )
        self.logger.info("Data preprocessing completed.\n")
    
    def _check_augmented_data_exists(self, img_f: Path) -> bool:
        """Check if augmented data exists with correct number of files"""
        aug_img_folder: Path = self.cfg['aug_img']
        aug_mask_folder: Path = self.cfg['aug_mask']
        
        if not aug_img_folder.exists() or not aug_mask_folder.exists():
            self.logger.warning("Augmented data/mask folder not found.")
            return False
        
        try:
            orig_img_count = sum(1 for f in img_f.iterdir()
                               if f.suffix.lower() in IMG_EXTENSIONS)
        except FileNotFoundError:
            self.logger.error(f"Original image folder not found: {img_f}")
            return False
        
        if orig_img_count == 0:
            self.logger.warning(f"No images found in {img_f}.")
            return False
        
        aug_img_count = sum(1 for f in aug_img_folder.iterdir()
                           if f.suffix.lower() in IMG_EXTENSIONS)
        aug_mask_count = sum(1 for f in aug_mask_folder.iterdir()
                            if f.suffix.lower() in IMG_EXTENSIONS)
        
        expected_count = orig_img_count * AUGMENTATION_FACTOR
        
        if aug_img_count >= expected_count and aug_mask_count >= expected_count:
            self.logger.info(f"Found {aug_img_count} aug images and {aug_mask_count} aug masks.")
            self.logger.info(f"(Expected >= {expected_count} based on {orig_img_count} originals)")
            return True
        
        self.logger.info(f"Found {aug_img_count} aug images, {aug_mask_count} aug masks.")
        self.logger.info(f"Expected {expected_count}. Preprocessing will run.")
        return False
    
    def prepare_data(self):
        """Create train and validation DataLoaders from separate datasets"""
        self.logger.info("Preparing data loaders...")
        
        # Training dataset (augmented)
        train_ds = SegmentationDataset(
            str(self.cfg['aug_img']),
            str(self.cfg['aug_mask']),
            (256, 256)
        )
        
        if len(train_ds) == 0:
            self.logger.error("Training dataset is empty. Check augmented data folders.")
            raise ValueError("Cannot train on an empty dataset.")
        
        self.logger.info(f"Training dataset size: {len(train_ds)}")
        
        # Validation dataset (separate validation set)
        val_ds = SegmentationDataset(
            str(self.cfg['val_image_folder']),
            str(self.cfg['val_mask_folder']),
            (256, 256)
        )
        
        if len(val_ds) == 0:
            self.logger.error("Validation dataset is empty. Check validation data folders.")
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
        """Initialize transfer learning model, optimizer, scheduler, and loss"""
        self.logger.info(f"Building transfer learning model: {self.model_name}...")
        
        try:
            self.model = get_transfer_model(self.model_name)
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
        
        self.model.to(self.device)
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Optimizer with different learning rates for pretrained vs new layers
        # if self.freeze_encoder:
        #     # Only optimize unfrozen parameters
        #     params = [p for p in self.model.parameters() if p.requires_grad]
        #     self.opt = optim.AdamW(params, lr=self.cfg['lr'], weight_decay=0.01)
        # else:
        #     # Use discriminative learning rates
        #     base_lr = self.cfg['lr']
        #     params = [
        #         {'params': self._get_encoder_params(), 'lr': base_lr * 0.1},
        #         {'params': self._get_decoder_params(), 'lr': base_lr}
        #     ]
        #     self.opt = optim.AdamW(params, weight_decay=0.01)
        base_lr = self.cfg['lr']
        params = [
            {'params': self._get_encoder_params(), 'lr': base_lr * 0.1},
            {'params': self._get_decoder_params(), 'lr': base_lr}
        ]
        self.opt = optim.AdamW(params, weight_decay=0.01)
        
        self.sch = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='max', patience=5, factor=0.5
        )
        
        # Loss function
        self.crit = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
        
        
        self.logger.info(f"Model '{self.model_name}' built successfully.\n")
    
    def _get_encoder_params(self):
        """Get encoder parameters for discriminative learning rates"""
        encoder_params = []
        for name, param in self.model.named_parameters():
            if 'encoder' in name or 'backbone' in name or 'vision' in name:
                encoder_params.append(param)
        return encoder_params
    
    def _get_decoder_params(self):
        """Get decoder/head parameters for discriminative learning rates"""
        decoder_params = []
        for name, param in self.model.named_parameters():
            if not any(x in name for x in ['encoder', 'backbone', 'vision']):
                decoder_params.append(param)
        return decoder_params
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model on the validation set
        
        Returns:
            Dictionary containing validation metrics (loss, iou, dice, etc.)
        """
        self.logger.info("Running validation...")
        metrics = evaluate_model(self.model, self.val_loader, self.device)
        # Compute validation loss
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b in self.val_loader:
                imgs = b['image'].to(self.device, non_blocking=True)
                masks = b['mask'].to(self.device, non_blocking=True)
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    preds = self.model(imgs)
                    loss = self.crit(preds, masks)
                
                val_loss += loss.item()

        metrics['loss'] = val_loss / len(self.val_loader)
        self.logger.info(
            f"Validation IoU: {metrics['iou']:.4f}"
        )
        
        return metrics
    
    def train(self):
        """Main training and validation loop"""
        self.logger.info(f"--- Starting Transfer Learning Training for {self.model_name} ---")
        self.logger.info(f"Epochs: {self.cfg['epochs']}")
        self.logger.info(f"Batch Size: {self.cfg['bs']}")
        self.logger.info(f"Base Learning Rate: {self.cfg['lr']}")
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
                masks = b['mask'].to(self.device, non_blocking=True)
                
                self.opt.zero_grad(set_to_none=True)
                
                # Mixed precision training
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    preds = self.model(imgs)
                    loss = self.crit(preds, masks)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                
                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            avg_train_loss = train_loss / len(self.train_loader)
            
            # Validation on separate validation set
            m = self.validate()
            val_iou = m['iou']
            
            # Get current learning rate
            current_lr = self.opt.param_groups[0]['lr']
            
            self.logger.info(
                f"[{self.model_name}] Epoch {ep:03d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val IoU: {val_iou:.4f} | "
                f"Val Dice: {m['dice']:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Log metrics to CSV
            self._log_metrics(ep, avg_train_loss, m, current_lr)
            
            # Update learning rate scheduler
            self.sch.step(val_iou)
            
            # Save best model
            if val_iou > self.best_iou:
                self.best_iou = val_iou
                self.logger.info(
                    f"✨ New best IoU: {self.best_iou:.4f}. "
                    f"Saving model to {self.best_ckpt_path}"
                )
                
                save_data = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'best_val_iou': self.best_iou,
                    'epoch': ep,
                    'model_name': self.model_name,
                }
                torch.save(save_data, self.best_ckpt_path)
        
        self.logger.info(
            f"[{self.model_name}] Training completed. "
            f"Best IoU achieved: {self.best_iou:.4f}\n"
        )
        return self.best_iou


def main():
    """
    Transfer Learning Segmentation Pipeline CLI
    
    Uses pretrained HuggingFace models for skin lesion segmentation.
    Supports models: Segformer, MiT-B1, BEiT, MedSAM2, and adaptive variants.
    
    Examples:
        # Train with Segformer (frozen encoder)
        python seg_transfer_learning_pipeline.py --model segformer
        
        # Train with BEiT (full fine-tuning)
        python seg_transfer_learning_pipeline.py --model beit
        
        # Train all transfer learning models
        python seg_transfer_learning_pipeline.py --model all
        
        # Custom training with metrics tracking
        python seg_transfer_learning_pipeline.py \\
            --model adaptive_segformer \\
            --batch_size 16 \\
            --num_epochs 50 \\
            --learning_rate 1e-4 \\
            --track_metrics \\
            --use_amp
        
        # Train on specific GPU with custom paths
        python seg_transfer_learning_pipeline.py \\
            --model mit_b1 \\
            --image_folder data/images \\
            --mask_folder data/masks \\
            --force_device cuda \\
            --visible_cuda_devices 0
    """
    
    p = argparse.ArgumentParser(
        description='Transfer learning segmentation pipeline with HuggingFace models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input/Output paths
    p.add_argument(
        '--image_folder', type=str,
        default="datasets/ISIC2018_Task1-2_Training_Input",
        help='Path to original input images (default: %(default)s)'
    )
    p.add_argument(
        '--mask_folder', type=str,
        default="datasets/ISIC2018_Task1_Training_GroundTruth",
        help='Path to ground truth masks (default: %(default)s)'
    )
    p.add_argument(
        '--aug_image_folder', type=str,
        default="./aug_img",
        help='Path to save augmented images (default: %(default)s)'
    )
    p.add_argument(
        '--aug_mask_folder', type=str,
        default="./aug_mask",
        help='Path to save augmented masks (default: %(default)s)'
    )
    p.add_argument(
        '--val_image_folder', type=str,
        default="datasets/ISIC2018_Task1-2_Validation_Input",
        help='Path to validation images (default: %(default)s)'
    )
    p.add_argument(
        '--val_mask_folder', type=str,
        default="datasets/ISIC2018_Task1_Validation_GroundTruth",
        help='Path to validation masks (default: %(default)s)'
    )   
    p.add_argument(
        '--ckpt', type=str,
        default="./checkpoints",
        help='Checkpoint directory (default: %(default)s)'
    )
    
    # Model selection
    available_models = get_available_transfer_models()
    p.add_argument(
        '--model', type=str, default='segformer',
        help=f'Transfer learning model. Options: {", ".join(available_models)}, or "all" (default: %(default)s)'
    )
    # p.add_argument(
    #     '--freeze_encoder', action='store_true',
    #     help='Freeze encoder weights for faster training'
    # )
    
    # Training hyperparameters
    p.add_argument(
        '--batch_size', type=int, default=8,
        help='Batch size (default: %(default)s)'
    )
    p.add_argument(
        '--num_epochs', type=int, default=100,
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
        '--num_workers', type=int, default=64,
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
    
    args = p.parse_args()
    
    # Set visible CUDA devices
    if args.visible_cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cuda_devices
    
    # Determine which models to train
    available_models = get_available_transfer_models()
    if args.model == 'all':
        models_to_train = available_models
    elif args.model in available_models:
        models_to_train = [args.model]
    else:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {', '.join(available_models)}, or 'all'")
        return
    
    print(f"\n{'='*60}")
    print(f"Transfer Learning Pipeline")
    print(f"Models to train: {', '.join(models_to_train)}")
    print(f"{'='*60}\n")
    
    # Configuration dictionary
    cfg = {
        'aug_img': Path(args.aug_image_folder),
        'aug_mask': Path(args.aug_mask_folder),
        'image_folder': Path(args.image_folder),
        'mask_folder': Path(args.mask_folder),
        'val_image_folder': Path(args.val_image_folder),
        'val_mask_folder': Path(args.val_mask_folder),
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
        cfg['aug_img'].mkdir(parents=True, exist_ok=True)
        cfg['aug_mask'].mkdir(parents=True, exist_ok=True)
        cfg['ckpt'].mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating directories: {e}")
        return
    
    # Verify validation folders exist
    if not cfg['val_image_folder'].exists():
        print(f"Error: Validation image folder not found: {cfg['val_image_folder']}")
        return
    if not cfg['val_mask_folder'].exists():
        print(f"Error: Validation mask folder not found: {cfg['val_mask_folder']}")
        return
    
    # Train models
    results = {}
    for idx, model_name in enumerate(models_to_train):
        print(f"\n{'='*60}")
        print(f"Training model: {model_name} ({idx+1}/{len(models_to_train)})")
        print(f"{'='*60}\n")
        
        try:
            pipeline = TransferLearningPipeline(
                cfg,
                model_name=model_name,
            )
            
            # Preprocess only once for the first model
            if idx == 0:
                pipeline.preprocess(cfg['image_folder'], cfg['mask_folder'])
                pipeline.prepare_data()
            else:
                # Reuse augmented data for subsequent models
                pipeline.prepare_data()
            
            pipeline.build_model()
            best_iou = pipeline.train()
            results[model_name] = best_iou
            
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
    print("TRAINING SUMMARY - Transfer Learning Models")
    print(f"{'='*60}")
    for model_name, best_iou in results.items():
        if best_iou is not None:
            print(f"{model_name:25s}: Best IoU = {best_iou:.4f}")
        else:
            print(f"{model_name:25s}: FAILED")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
