import os, argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path  # Import Path
from typing import Optional, Dict, Any
import csv
from datetime import datetime

# Assuming these imports exist and are correct
from preprocessing import process_dataset_parallel, SegmentationDataset
from seg_models import get_model, get_available_models, MODEL_REGISTRY
from evaluation import CombinedLoss, evaluate_model
from utils import setup_logger, get_num_workers

# --- Constants ---
# Moved magic number 24 to a configurable constant
# TODO: This could be passed in cfg if augmentation logic changes
AUGMENTATION_FACTOR = 24
# Define common image extensions
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


class SegmentationPipeline:
    """
    Encapsulates the end-to-end segmentation pipeline, from preprocessing
    to training and evaluation.
    """

    def __init__(self, cfg: Dict[str, Any], model_name: str = 'combined', logger: Optional[Any] = None):
        self.cfg = cfg
        self.model_name = model_name
        self.logger = logger or setup_logger(name='pipe')

        # --- Device Configuration ---
        if cfg.get('force_device'):
            self.device = torch.device(cfg['force_device'])
            self.logger.info(f"Forcing device: {self.device}")
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Auto-detected device: {self.device}")

        self.num_workers = get_num_workers(cfg.get('num_workers', 4))
        self.logger.info(f"Using {self.num_workers} data loader workers.")

        # --- Checkpoint & Model State ---
        self.best_iou = 0.0
        self.ckpt_dir: Path = self.cfg['ckpt'] / model_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_path: Path = self.ckpt_dir / 'best.pth'

        # --- Metric Tracking ---
        self.track_metrics = cfg.get('track_metrics', False)
        if self.track_metrics:
            self.metrics_dir: Path = Path('outputs')
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.metrics_file: Path = self.metrics_dir / f'{model_name}_metrics_{timestamp}.csv'
            self._init_metrics_file()

        # --- Optional: Mixed Precision Scaler ---
        # self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))

    def _init_metrics_file(self):
        """Initialize CSV file for tracking metrics"""
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'val_dice', 
                           'val_accuracy', 'val_precision', 'val_recall', 'val_f1'])
        self.logger.info(f"Metrics will be tracked in: {self.metrics_file}")

    def _log_metrics(self, epoch: int, train_loss: float, metrics: Dict[str, float]):
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
            ])

    def preprocess(self, img_f: Path, mask_f: Path):
        """
        Checks for existing augmented data and runs parallel preprocessing
        if data is not found.
        """
        self.logger.info("Starting data preprocessing...")

        # Check if augmented data already exists
        if self._check_augmented_data_exists(img_f):
            self.logger.info("Sufficient augmented data found. Skipping preprocessing.")
            return

        self.logger.info("No/incomplete augmented data found. Starting parallel processing...")
        # Cast Paths to strings for external function, as its signature is unknown
        process_dataset_parallel(str(img_f),
                                 str(mask_f),
                                 str(self.cfg['aug_img']),
                                 str(self.cfg['aug_mask']),
                                 True,
                                 self.num_workers,
                                 self.logger)
        self.logger.info("Data preprocessing completed.\n")

    def _check_augmented_data_exists(self, img_f: Path) -> bool:
        """
        Check if augmented data exists and has the correct number of files.
        """
        aug_img_folder: Path = self.cfg['aug_img']
        aug_mask_folder: Path = self.cfg['aug_mask']

        # Check if both folders exist
        if not aug_img_folder.exists() or not aug_mask_folder.exists():
            self.logger.warning("Augmented data/mask folder not found.")
            return False

        # Count original files (using iterdir for efficiency if dirs are flat)
        try:
            orig_img_count = sum(1 for f in img_f.iterdir()
                                 if f.suffix.lower() in IMG_EXTENSIONS)
        except FileNotFoundError:
            self.logger.error(f"Original image folder not found: {img_f}")
            return False

        if orig_img_count == 0:
            self.logger.warning(f"No images found in {img_f}.")
            return False

        # Count augmented files
        aug_img_count = sum(1 for f in aug_img_folder.iterdir()
                            if f.suffix.lower() in IMG_EXTENSIONS)
        aug_mask_count = sum(1 for f in aug_mask_folder.iterdir()
                             if f.suffix.lower() in IMG_EXTENSIONS)

        expected_count = orig_img_count * AUGMENTATION_FACTOR

        # Check if augmented data has at least the expected number of files
        if aug_img_count >= expected_count and aug_mask_count >= expected_count:
            self.logger.info(f"Found {aug_img_count} aug images and {aug_mask_count} aug masks.")
            self.logger.info(f"(Expected >= {expected_count} based on {orig_img_count} originals)")
            return True

        self.logger.info(f"Found {aug_img_count} aug images, {aug_mask_count} aug masks.")
        self.logger.info(f"Expected {expected_count}. Preprocessing will run.")
        return False

    def prepare_data(self):
        """
        Creates and splits the dataset into train/validation DataLoaders.
        """
        self.logger.info("Preparing data loaders...")
        # Cast Paths to strings for external Dataset class
        ds = SegmentationDataset(str(self.cfg['aug_img']),
                                 str(self.cfg['aug_mask']),
                                 (256, 256))

        if len(ds) == 0:
            self.logger.error("Dataset is empty. Check augmented data folders.")
            raise ValueError("Cannot train on an empty dataset.")

        # Split dataset
        tr_sz = int(0.8 * len(ds))
        val_sz = len(ds) - tr_sz
        self.logger.info(f"Total dataset size: {len(ds)}. Splitting into {tr_sz} (train) / {val_sz} (val).")
        tr_ds, val_ds = random_split(ds, [tr_sz, val_sz])

        self.train_loader = DataLoader(tr_ds,
                                       self.cfg['bs'],
                                       shuffle=True,
                                       num_workers=self.num_workers,
                                       pin_memory=True)
        self.val_loader = DataLoader(val_ds,
                                     self.cfg['bs'],
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     pin_memory=True)
        self.logger.info("Done preparing data loaders.\n")

    def build_model(self):
        """
        Initializes the model, optimizer, scheduler, and loss function.
        Loads a checkpoint if one exists.
        """
        self.logger.info(f"Building model: {self.model_name}...")
        self.model = get_model(self.model_name,
                               self.cfg.get('use_mcnn', True),
                               True)

        self.model.to(self.device)

        self.opt = optim.Adam(self.model.parameters(), self.cfg['lr'])
        self.sch = optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'max', patience=5, factor=0.1)
        self.crit = CombinedLoss()

        # Load checkpoint if exists
        self._load_checkpoint_if_exists()

        self.logger.info(f"Model '{self.model_name}' built successfully.\n")

    def _load_checkpoint_if_exists(self):
        """
        Load the latest checkpoint. Prioritizes 'best.pth'.
        """
        if not self.ckpt_dir.exists():
            self.logger.info("Checkpoint directory not found. Starting from scratch.")
            return

        # Prioritize 'best.pth'
        if self.best_ckpt_path.exists():
            ckpt_path = self.best_ckpt_path
        else:
            # Find any other .pth file if 'best.pth' is missing
            try:
                ckpt_path = next(self.ckpt_dir.glob('*.pth'))
            except StopIteration:
                self.logger.info("No checkpoint files found. Starting from scratch.")
                return

        self.logger.info(f"Loading checkpoint from: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)

            # Handle dictionary-based checkpoints (common)
            if isinstance(checkpoint, dict):
                # Flexible key finding
                if 'model_state_dict' in checkpoint:
                    model_state = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    model_state = checkpoint['model']
                else:
                    model_state = checkpoint

                self.model.load_state_dict(model_state)

                # Load optimizer state if available
                opt_state = checkpoint.get('optimizer_state_dict') or checkpoint.get('optimizer')
                if opt_state:
                    self.opt.load_state_dict(opt_state)
                    self.logger.info("Optimizer state loaded.")

                # Load best IoU if available
                iou = checkpoint.get('best_val_iou') or checkpoint.get('iou')
                if iou is not None:
                    self.best_iou = iou
                    self.logger.info(f"Resuming from checkpoint with best IoU: {self.best_iou:.4f}")
            else:
                # Handle raw state_dict
                self.model.load_state_dict(checkpoint)

            self.logger.info("Checkpoint loaded successfully!")
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            self.logger.warning("Starting training from scratch...")

    def train(self):
        """
        Runs the main training and validation loop.
        """
        self.logger.info(f"--- Starting Training for {self.model_name} ---")
        self.logger.info(f"Epochs: {self.cfg['epochs']}")
        self.logger.info(f"Batch Size: {self.cfg['bs']}")
        self.logger.info(f"Learning Rate: {self.cfg['lr']}")
        self.logger.info(f"Device: {self.device}")

        for ep in range(1, self.cfg['epochs'] + 1):
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"[{self.model_name}] Ep {ep}/{self.cfg['epochs']} [Train]")
            for b in pbar:
                imgs = b['image'].to(self.device, non_blocking=True)
                masks = b['mask'].to(self.device, non_blocking=True)

                self.opt.zero_grad(set_to_none=True)

                # --- Standard Training ---
                preds = self.model(imgs)
                loss = self.crit(preds, masks)

                loss.backward()
                self.opt.step()

                # --- Optional: Mixed Precision Training ---
                # (Uncomment block below and comment block above)
                # with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                #    preds = self.model(imgs)
                #    loss = self.crit(preds, masks)
                #
                # self.scaler.scale(loss).backward()
                # self.scaler.step(self.opt)
                # self.scaler.update()
                # ------------------------------------------

                train_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = train_loss / len(self.train_loader)

            # --- Validation ---
            m = evaluate_model(self.model, self.val_loader, self.device)
            val_iou = m['iou']

            self.logger.info(
                f"[{self.model_name}] Epoch {ep:03d} | Train Loss: {avg_train_loss:.4f} | Val IoU: {val_iou:.4f}"
            )

            # --- Log metrics to CSV ---
            self._log_metrics(ep, avg_train_loss, m)

            # --- Scheduler & Checkpointing ---
            self.sch.step(val_iou)

            if val_iou > self.best_iou:
                self.best_iou = val_iou
                self.logger.info(f"✨ New best IoU: {self.best_iou:.4f}. Saving model to {self.best_ckpt_path}")

                # Save checkpoint with more info
                save_data = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.opt.state_dict(),
                    'best_val_iou': self.best_iou,
                    'epoch': ep,
                    'model_name': self.model_name,
                }
                torch.save(save_data, self.best_ckpt_path)

        self.logger.info(f"[{self.model_name}] Training completed. Best IoU achieved: {self.best_iou:.4f}\n")
        return self.best_iou

def main():
    """
    Segmentation Pipeline CLI

    Complete training pipeline for image segmentation with data augmentation,
    checkpoint management, and automatic preprocessing.

    Examples:
        # Basic training with defaults (combined model)
        python seg_pipeline.py

        # Train specific model
        python seg_pipeline.py --model deeplabv3plus

        # Train all models
        python seg_pipeline.py --model all

        # Track metrics to CSV
        python seg_pipeline.py --track_metrics

        # Custom paths and parameters
        python seg_pipeline.py \\
            --image_folder data/images \\
            --mask_folder data/masks \\
            --batch_size 16 \\
            --num_epochs 50 \\
            --model fat_net \\
            --track_metrics

        # Resume training from checkpoint
        python seg_pipeline.py \\
            --ckpt ./checkpoints \\
            --image_folder data/images \\
            --mask_folder data/masks \\
            --model combined

        # Force GPU training on specific device
        python seg_pipeline.py \\
            --force_device cuda \\
            --visible_cuda_devices 0,1
    """

    # --- Args Parser Section ---
    p = argparse.ArgumentParser(
        description='Skin lesion segmentation training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/Output Arguments
    p.add_argument(
        '--image_folder', type=str,
        default="datasets/ISIC2018_Task1-2_Training_Input",
        help='Path to original input images directory (default: %(default)s)'
    )
    p.add_argument(
        '--mask_folder', type=str,
        default="datasets/ISIC2018_Task1_Training_GroundTruth",
        help='Path to ground truth segmentation masks directory (default: %(default)s)'
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
        '--ckpt', type=str,
        default="./checkpoints",
        help='Checkpoint directory for saving/loading models (default: %(default)s)'
    )

    # Model Selection
    available_models = get_available_models()
    p.add_argument(
        '--model', type=str, default='combined',
        help=f'Model to train. Options: {", ".join(available_models)}, or "all" to train all models (default: %(default)s)'
    )

    # Training Hyperparameters
    p.add_argument(
        '--batch_size', type=int, default=8, metavar='BS',
        help='Batch size for training (default: %(default)s)'
    )
    p.add_argument(
        '--num_epochs', type=int, default=100, metavar='E',
        help='Number of training epochs (default: %(default)s)'
    )
    p.add_argument(
        '--learning_rate', type=float, default=1e-4, metavar='LR',
        help='Adam optimizer learning rate (default: %(default)s)'
    )

    # Metric Tracking
    p.add_argument(
        '--track_metrics', action='store_true',
        help='Track loss and IoU for each epoch in a CSV file under outputs/'
    )

    # Data Loading & Device Configuration
    p.add_argument(
        '--num_workers', type=int, default=64, metavar='W',
        help='Number of parallel data loading workers. Use -1 for all available CPUs (default: %(default)s)'
    )
    p.add_argument(
        '--visible_cuda_devices', type=str, default=None, metavar='DEVICES',
        help='GPU device indices to use (e.g., "0,1,2"). Only effective with --force_device cuda (default: all available)'
    )
    p.add_argument(
        '--force_device', type=str, default=None,
        metavar='DEVICE', choices=['cuda', 'cpu'],
        help='Force specific device: "cuda" for GPU, "cpu" for CPU. Default: auto-detect (CUDA if available)'
    )

    args = p.parse_args()

    if args.visible_cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cuda_devices

    # Determine which models to train
    available_models = get_available_models()
    if args.model == 'all':
        models_to_train = available_models
    elif args.model in available_models:
        models_to_train = [args.model]
    else:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {', '.join(available_models)}, or 'all'")
        return

    print(f"\n{'='*60}")
    print(f"Models to train: {', '.join(models_to_train)}")
    print(f"{'='*60}\n")

    # --- Config Dictionary ---
    # Convert string paths from argparse into Path objects for internal use
    cfg = {
        'aug_img': Path(args.aug_image_folder),
        'aug_mask': Path(args.aug_mask_folder),
        'image_folder': Path(args.image_folder),
        'mask_folder': Path(args.mask_folder),
        'ckpt': Path(args.ckpt),
        'bs': args.batch_size,
        'epochs': args.num_epochs,
        'lr': args.learning_rate,
        'num_workers': args.num_workers,
        'force_device': args.force_device,
        'track_metrics': args.track_metrics,
    }

    # --- Create necessary directories ---
    # Use pathlib's mkdir()
    try:
        cfg['aug_img'].mkdir(parents=True, exist_ok=True)
        cfg['aug_mask'].mkdir(parents=True, exist_ok=True)
        cfg['ckpt'].mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating directories: {e}")
        return

    # --- Main pipeline initiation ---
    results = {}
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training model: {model_name}")
        print(f"{'='*60}\n")
        
        try:
            pipeline = SegmentationPipeline(cfg, model_name=model_name)
            
            # Only preprocess once for all models
            if model_name == models_to_train[0]:
                pipeline.preprocess(cfg['image_folder'], cfg['mask_folder'])
                pipeline.prepare_data()
            else:
                # Reuse data loaders from first pipeline
                # For simplicity, recreate them (could be optimized)
                pipeline.prepare_data()
            
            pipeline.build_model()
            best_iou = pipeline.train()
            results[model_name] = best_iou
            
        except Exception as e:
            # Use a logger if available, otherwise print
            logger = getattr(pipeline, 'logger', None) if 'pipeline' in locals() else None
            if logger:
                logger.error(f"Pipeline failed for model '{model_name}': {e}", exc_info=True)
            else:
                print(f"An unexpected error occurred for model '{model_name}': {e}")
                import traceback
                traceback.print_exc()
            results[model_name] = None

    # --- Summary ---
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for model_name, best_iou in results.items():
        if best_iou is not None:
            print(f"{model_name:20s}: Best IoU = {best_iou:.4f}")
        else:
            print(f"{model_name:20s}: FAILED")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
