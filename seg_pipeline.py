import os, argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from preprocessing import process_dataset_parallel, SegmentationDataset
from seg_models import get_model
from evaluation import CombinedLoss, evaluate_model
from utils import setup_logger, get_num_workers

class SegmentationPipeline:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger or setup_logger(name='pipe')
        if cfg.get('force_device'):
            self.device = torch.device(cfg['force_device'])
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = get_num_workers(cfg.get('num_workers', 4))
        self.best_iou = 0.0
    
    def preprocess(self, img_f, mask_f):
        print("Starting data preprocessing...")
        
        # Check if augmented data already exists
        if self._check_augmented_data_exists(img_f, mask_f):
            print("Augmented images found, skipping augmentation process")
            return
        
        process_dataset_parallel(img_f, 
                                  mask_f, 
                                  self.cfg['aug_img'], 
                                  self.cfg['aug_mask'],
                                  True, 
                                  self.num_workers, 
                                  self.logger)
        print("Data preprocessing completed.\n")
    
    def _check_augmented_data_exists(self, img_f, mask_f):
        """Check if augmented data exists and has correct number of files"""
        aug_img_folder = self.cfg['aug_img']
        aug_mask_folder = self.cfg['aug_mask']
        
        # Check if both folders exist
        if not os.path.exists(aug_img_folder) or not os.path.exists(aug_mask_folder):
            return False
        
        # Count original files
        orig_img_files = [f for f in os.listdir(img_f) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))]
        orig_mask_files = [f for f in os.listdir(mask_f) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Count augmented files
        aug_img_files = [f for f in os.listdir(aug_img_folder) 
                        if f.endswith(('.png', '.jpg', '.jpeg'))]
        aug_mask_files = [f for f in os.listdir(aug_mask_folder) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Check if augmented data has 24x original files
        # (1 original + 7 rotations * 2 dilations + 2 dilations on original = 1 + 14 + 2 = 17)
        # Actually it should be: 1 original + 7 rotations (each with 2 dilations) + 2 dilations on original
        # = 1 + 7 + 14 + 2 = 24 per original image
        expected_count = len(orig_img_files) * 24
        
        if len(aug_img_files) >= expected_count and len(aug_mask_files) >= expected_count:
            print(f"Found {len(aug_img_files)} augmented images (expected {expected_count})")
            print(f"Found {len(aug_mask_files)} augmented masks (expected {expected_count})")
            return True
        
        return False

    def prepare_data(self):
        print("Preparing data loaders...")
        ds = SegmentationDataset(self.cfg['aug_img'], 
                                 self.cfg['aug_mask'], 
                                 (256, 256))
        
        # Split dataset
        tr_sz = int(0.8 * len(ds))
        tr_ds, val_ds = random_split(ds, [tr_sz, len(ds)-tr_sz])
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
        print("Done preparing data loaders.\n")
    
    def build_model(self):
        print("Building model...")
        self.model = get_model('combined', 
                               self.cfg.get('use_mcnn', True), 
                               True)
        
        self.model.to(self.device)
        
        self.opt = optim.Adam(self.model.parameters(), self.cfg['lr'])
        self.sch = optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'max')
        self.crit = CombinedLoss()
        
        # Load checkpoint if exists
        self._load_checkpoint_if_exists()
        
        print("Model built successfully.\n")
    
    def _load_checkpoint_if_exists(self):
        """Load checkpoint if checkpoint directory exists and is not empty"""
        ckpt_dir = self.cfg['ckpt']
        
        if not os.path.exists(ckpt_dir):
            return
        
        # Look for checkpoint files
        ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
        
        if not ckpt_files:
            return
        
        # Try to load best.pth first, otherwise load the first checkpoint found
        best_ckpt = os.path.join(ckpt_dir, 'best.pth')
        if os.path.exists(best_ckpt):
            ckpt_path = best_ckpt
        else:
            ckpt_path = os.path.join(ckpt_dir, ckpt_files[0])
        
        print(f"Loading checkpoint from: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                # Load optimizer state if available
                if 'optimizer_state_dict' in checkpoint or 'optimizer' in checkpoint:
                    opt_state = checkpoint.get('optimizer_state_dict') or checkpoint.get('optimizer')
                    self.opt.load_state_dict(opt_state)
                
                # Load best IoU if available
                if 'best_val_iou' in checkpoint or 'iou' in checkpoint:
                    self.best_iou = checkpoint.get('best_val_iou') or checkpoint.get('iou')
                    print(f"Resuming from checkpoint with best IoU: {self.best_iou:.4f}")
            else:
                self.model.load_state_dict(checkpoint)
            
            print("Checkpoint loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            print("Starting training from scratch...")
    
    def train(self):
        print("Starting training...")
        for ep in range(1, self.cfg['epochs']+1):
            self.model.train()
            for b in tqdm(self.train_loader, desc=f"Ep {ep}"):
                imgs, masks = b['image'].to(self.device), b['mask'].to(self.device)
                self.opt.zero_grad()
                loss = self.crit(self.model(imgs), masks)
                loss.backward()
                self.opt.step()
            m = evaluate_model(self.model, self.val_loader, self.device)
            print(f"Epoch {ep}/{self.cfg['epochs']} - Val IOU: {m['iou']:.4f}")
            self.sch.step(m['iou'])
            if m['iou'] > self.best_iou:
                self.best_iou = m['iou']
                torch.save(self.model.state_dict(), f"{self.cfg['ckpt']}/best.pth")
        print("Training completed.\n")
 
def main():
    """
    Segmentation Pipeline CLI
    
    Complete training pipeline for image segmentation with data augmentation,
    checkpoint management, and automatic preprocessing.
    
    Examples:
        # Basic training with defaults
        python seg_pipeline.py
        
        # Custom paths and parameters
        python seg_pipeline.py \\
            --image_folder data/images \\
            --mask_folder data/masks \\
            --batch_size 16 \\
            --num_epochs 50
        
        # Resume training from checkpoint
        python seg_pipeline.py \\
            --ckpt ./checkpoints \\
            --image_folder data/images \\
            --mask_folder data/masks
        
        # Force GPU training on specific device
        python seg_pipeline.py \\
            --force_device cuda \\
            --visible_cuda_devices 0,1
    """

    
    # Args Parser Section
    p = argparse.ArgumentParser(
        description='Skin lesion segmentation training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input/Output Arguments
    p.add_argument(
        '--image_folder',
        type=str,
        default="datasets/ISIC2018_Task1-2_Training_Input",
        help='Path to original input images directory (default: %(default)s)'
    )
    p.add_argument(
        '--mask_folder',
        type=str,
        default="datasets/ISIC2018_Task1_Training_GroundTruth",
        help='Path to ground truth segmentation masks directory (default: %(default)s)'
    )
    p.add_argument(
        '--aug_image_folder',
        type=str,
        default="./aug_img",
        help='Path to save augmented images (default: %(default)s)'
    )
    p.add_argument(
        '--aug_mask_folder',
        type=str,
        default="./aug_mask",
        help='Path to save augmented masks (default: %(default)s)'
    )
    p.add_argument(
        '--ckpt',
        type=str,
        default="./checkpoints",
        help='Checkpoint directory for saving/loading models (default: %(default)s)'
    )
    
    # Training Hyperparameters
    p.add_argument(
        '--batch_size',
        type=int,
        default=8,
        metavar='BS',
        help='Batch size for training (default: %(default)s)'
    )
    p.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        metavar='E',
        help='Number of training epochs (default: %(default)s)'
    )
    p.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        metavar='LR',
        help='Adam optimizer learning rate (default: %(default)s)'
    )
    
    # Data Loading & Device Configuration
    p.add_argument(
        '--num_workers',
        type=int,
        default=4,
        metavar='W',
        help='Number of parallel data loading workers. Use -1 for all available CPUs (default: %(default)s)'
    )
    p.add_argument(
        '--visible_cuda_devices',
        type=str,
        default=None,
        metavar='DEVICES',
        help='GPU device indices to use (e.g., "0,1,2"). Only effective with --force_device cuda (default: all available)'
    )
    p.add_argument(
        '--force_device',
        type=str,
        default=None,
        metavar='DEVICE',
        choices=['cuda', 'cpu'],
        help='Force specific device: "cuda" for GPU, "cpu" for CPU. Default: auto-detect (CUDA if available)'
    )
    a = p.parse_args()
    
    if a.visible_cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.visible_cuda_devices
        
    # Configs
    cfg = {'aug_img': a.aug_image_folder, 
           'aug_mask': a.aug_mask_folder, 
           'image_folder': a.image_folder,
           'mask_folder': a.mask_folder,
           'ckpt': a.ckpt,
           'bs': a.batch_size, 
           'epochs': a.num_epochs, 
           'lr': a.learning_rate, 
           'num_workers': a.num_workers,
           'force_device': a.force_device,
    }
    
    # Create necessary directories
    os.makedirs(cfg['aug_img'], exist_ok=True)
    os.makedirs(cfg['aug_mask'], exist_ok=True)
    os.makedirs(cfg['ckpt'], exist_ok=True)
    
    # Main pipeline initiation
    pipeline = SegmentationPipeline(cfg)
    pipeline.preprocess(a.image_folder, a.mask_folder)
    pipeline.prepare_data()
    pipeline.build_model()
    pipeline.train()

if __name__ == "__main__":
    main()
