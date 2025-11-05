"""Pipeline - uses modular components"""
import os, argparse, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from preprocessing import process_dataset_parallel, SegmentationDataset
from models import get_model
from evaluation import CombinedLoss, evaluate_model
from utils import setup_logger, get_num_workers

class SegmentationPipeline:
    def __init__(self, cfg, logger=None):
        self.cfg, self.logger = cfg, logger or setup_logger(name='pipe')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = get_num_workers(cfg.get('num_workers', 4))
        self.best_iou = 0.0
    
    def preprocess(self, img_f, mask_f):
        process_dataset_parallel(img_f, 
                                  mask_f, 
                                  self.cfg['aug_img'], 
                                  self.cfg['aug_mask'],
                                  True, 
                                  self.num_workers, 
                                  self.logger)

    def prepare_data(self):
        ds = SegmentationDataset(self.cfg['aug_img'], self.cfg['aug_mask'], (256, 256))
        tr_sz = int(0.8 * len(ds))
        tr_ds, val_ds = random_split(ds, [tr_sz, len(ds)-tr_sz])
        self.train_loader = DataLoader(tr_ds, self.cfg['bs'], True)
        self.val_loader = DataLoader(val_ds, self.cfg['bs'], False)
    
    def build_model(self):
        self.model = get_model('combined', self.cfg.get('use_mcnn', True), True).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), self.cfg['lr'])
        self.sch = optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'max')
        self.crit = CombinedLoss()
    
    def train(self):
        for ep in range(1, self.cfg['epochs']+1):
            self.model.train()
            for b in tqdm(self.train_loader, desc=f"Ep {ep}"):
                imgs, masks = b['image'].to(self.device), b['mask'].to(self.device)
                self.opt.zero_grad()
                loss = self.crit(self.model(imgs), masks)
                loss.backward()
                self.opt.step()
            m = evaluate_model(self.model, self.val_loader, self.device)
            self.sch.step(m['iou'])
            if m['iou'] > self.best_iou:
                self.best_iou = m['iou']
                torch.save(self.model.state_dict(), f"{self.cfg['ckpt']}/best.pth")

def main():
    # Args Parser Section
    p = argparse.ArgumentParser()
    p.add_argument('--image_folder', default="datasets/ISIC2018_Task1-2_Training_Input")
    p.add_argument('--mask_folder', default="datasets/ISIC2018_Task1_Training_GroundTruth")
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    a = p.parse_args()
    
    # Configs
    cfg = {'aug_img': './aug_img', 'aug_mask': './aug_mask', 'ckpt': './checkpoints',
           'bs': a.batch_size, 'epochs': a.num_epochs, 'lr': a.learning_rate, 'num_workers': a.num_workers}

    # Main pipeline initiation
    pipe = SegmentationPipeline(cfg)
    pipe.preprocess(a.image_folder, a.mask_folder)
    pipe.prepare_data()
    pipe.build_model()
    pipe.train()

if __name__ == "__main__":
    main()
