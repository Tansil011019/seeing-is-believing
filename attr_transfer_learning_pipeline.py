"""
Attribute Transfer Learning Pipeline - Main Training Loop
Multilabel classification for 5 skin lesion attributes
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Dict, Any
import csv
from datetime import datetime

from preprocessing import AttributeDataset
from models.attr_transfer_model import AttributeTransferModel
from evaluation.attr_metrics import evaluate_multilabel
from training.attr_training import train_one_epoch
from seg_models.transfer_models import get_available_transfer_models
from utils import setup_logger, get_num_workers


class AttributePipeline:
    """Transfer learning pipeline for attribute multilabel classification"""
    
    def __init__(self, cfg: Dict[str, Any], model_name: str = 'segformer',
                 freeze_encoder: bool = False, logger=None):
        self.cfg = cfg
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder
        self.logger = logger or setup_logger(name='attr_pipe')
        
        # Device setup
        self.device = torch.device(
            cfg.get('force_device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.num_workers = get_num_workers(cfg.get('num_workers', 4))
        
        # Checkpoint and metrics
        self.best_f1 = 0.0
        self.ckpt_dir = cfg['ckpt'] / f'attr_transfer_{model_name}'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_path = self.ckpt_dir / 'best.pth'
        
        # Mixed precision training
        self.use_amp = cfg.get('use_amp', True) and (self.device.type == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Metrics tracking
        self.track_metrics = cfg.get('track_metrics', False)
        if self.track_metrics:
            self._setup_metrics_tracking()
    
    def _setup_metrics_tracking(self):
        """Initialize metrics CSV file"""
        self.metrics_dir = Path('outputs')
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.metrics_file = self.metrics_dir / f'attr_{self.model_name}_{timestamp}.csv'
        
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_f1_micro',
                           'val_f1_macro', 'hamming_loss', 'exact_match', 'lr'])
    
    def _log_metrics(self, epoch: int, train_loss: float, 
                    metrics: Dict[str, float], lr: float):
        """Log metrics to CSV"""
        if not self.track_metrics:
            return
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{train_loss:.6f}", f"{metrics['loss']:.6f}",
                f"{metrics['f1_micro']:.6f}", f"{metrics['f1_macro']:.6f}",
                f"{metrics['hamming_loss']:.6f}", f"{metrics['exact_match']:.6f}",
                f"{lr:.8f}"
            ])
    
    def prepare_data(self):
        """Create train/val DataLoaders"""
        self.logger.info("Preparing data loaders...")
        
        ds = AttributeDataset(
            str(self.cfg['image_folder']),
            str(self.cfg['gt_folder']),
            (256, 256)
        )
        
        # Split 80/20
        tr_sz = int(0.8 * len(ds))
        val_sz = len(ds) - tr_sz
        tr_ds, val_ds = random_split(ds, [tr_sz, val_sz])
        
        self.train_loader = DataLoader(tr_ds, self.cfg['bs'], shuffle=True,
                                      num_workers=self.num_workers, pin_memory=True)
        self.val_loader = DataLoader(val_ds, self.cfg['bs'], shuffle=False,
                                    num_workers=self.num_workers, pin_memory=True)
        self.logger.info(f"Data: {len(ds)} total, {tr_sz} train, {val_sz} val\n")
    
    def build_model(self):
        """Initialize model, optimizer, scheduler, loss"""
        self.logger.info(f"Building model: {self.model_name}")
        
        self.model = AttributeTransferModel(
            self.model_name, num_classes=5, freeze_encoder=self.freeze_encoder
        )
        self.model.to(self.device)
        
        # Optimizer, scheduler, loss
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = optim.AdamW(params, lr=self.cfg['lr'], weight_decay=0.01)
        self.sch = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode='max', patience=5, factor=0.5
        )
        self.crit = nn.BCEWithLogitsLoss()
        
        self.logger.info("Model built\n")
    
    def train(self):
        """Main training loop"""
        self.logger.info(f"--- Training {self.model_name} ---")
        
        for ep in range(1, self.cfg['epochs'] + 1):
            # Train one epoch
            avg_train_loss = train_one_epoch(
                self.model, self.train_loader, self.opt, self.crit,
                self.device, self.use_amp, self.scaler,
                self.model_name, ep, self.cfg['epochs']
            )
            
            # Validation
            metrics = evaluate_multilabel(self.model, self.val_loader, 
                                         self.crit, self.device)
            val_f1 = metrics['f1_macro']
            current_lr = self.opt.param_groups[0]['lr']
            
            self.logger.info(
                f"[{self.model_name}] Ep {ep:03d} | "
                f"Train: {avg_train_loss:.4f} | Val F1: {val_f1:.4f} | "
                f"Hamming: {metrics['hamming_loss']:.4f} | LR: {current_lr:.6f}"
            )
            
            # Log and save
            self._log_metrics(ep, avg_train_loss, metrics, current_lr)
            self.sch.step(val_f1)
            
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.logger.info(f"✨ New best F1: {self.best_f1:.4f}")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'best_f1': self.best_f1,
                    'epoch': ep,
                    'model_name': self.model_name
                }, self.best_ckpt_path)
        
        self.logger.info(f"Complete. Best F1: {self.best_f1:.4f}\n")
        return self.best_f1


def main():
    """CLI for attribute transfer learning"""
    p = argparse.ArgumentParser(description='Attribute detection pipeline')
    
    # Paths
    p.add_argument('--image_folder', type=str,
                   default="datasets/ISIC2018_Task1-2_Training_Input")
    p.add_argument('--gt_folder', type=str,
                   default="datasets/ISIC2018_Task2_Training_GroundTruth_v3")
    p.add_argument('--ckpt', type=str, default="./checkpoints")
    
    # Model
    available = get_available_transfer_models()
    p.add_argument('--model', type=str, default='segformer',
                   help=f'Model: {", ".join(available)} or "all"')
    p.add_argument('--freeze_encoder', action='store_true')
    
    # Training
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_epochs', type=int, default=50)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--use_amp', action='store_true', default=True)
    p.add_argument('--track_metrics', action='store_true')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--force_device', type=str, default='cuda')
    
    args = p.parse_args()
    
    # Determine models to train
    available = get_available_transfer_models()
    models = available if args.model == 'all' else [args.model]
    
    print(f"\n{'='*60}")
    print(f"Attribute Detection Transfer Learning")
    print(f"Models: {', '.join(models)}")
    print(f"{'='*60}\n")
    
    # Config
    cfg = {
        'image_folder': Path(args.image_folder),
        'gt_folder': Path(args.gt_folder),
        'ckpt': Path(args.ckpt),
        'bs': args.batch_size,
        'epochs': args.num_epochs,
        'lr': args.learning_rate,
        'num_workers': args.num_workers,
        'force_device': args.force_device,
        'track_metrics': args.track_metrics,
        'use_amp': args.use_amp,
    }
    
    cfg['ckpt'].mkdir(parents=True, exist_ok=True)
    
    # Train models
    results = {}
    for idx, model_name in enumerate(models):
        print(f"\n{'='*60}")
        print(f"Training: {model_name} ({idx+1}/{len(models)})")
        print(f"{'='*60}\n")
        
        try:
            pipeline = AttributePipeline(cfg, model_name, args.freeze_encoder)
            pipeline.prepare_data()
            pipeline.build_model()
            results[model_name] = pipeline.train()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = None
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for model_name, best_f1 in results.items():
        status = f"Best F1 = {best_f1:.4f}" if best_f1 else "FAILED"
        print(f"{model_name:25s}: {status}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
