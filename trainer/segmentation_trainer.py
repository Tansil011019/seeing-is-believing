import torch
from tqdm import tqdm
import logging
from evaluation import evaluate_model
from base.trainer import Trainer

import torch.nn as nn

logger = logging.getLogger(__name__)

class SegmentationTrainer(Trainer):
    def __init__(
        self, 
        model,
        model_name,
        optimizer,
        criterion,
        train_loader,
        val_loader, 
        device,
        epoch,
        scheduler = None,
        early_stopping = False,
        patience = -1,
        min_delta = 0,
        save_path = None,
        fold_index = None,
        scheduler_step_at_epoch_end = True
    ):
        super().__init__(
            model,
            model_name,
            optimizer,
            criterion,
            train_loader,
            val_loader,
            device,
            epoch,
            scheduler,
            early_stopping,
            patience,
            min_delta,
            save_path,
            fold_index,
            scheduler_step_at_epoch_end
        )
    
    def _train_one_epoch(self):
        self.model.train()
        train_loss = 0.0
        
        pbar = tqdm(
            self.train_loader,
        )
        
        for b in pbar:
            imgs = b['image'].to(self.device, non_blocking=True)
            masks = b['mask'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            preds = self.model(imgs)
            loss = self.criterion(preds, masks)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = train_loss / len(self.train_loader)
        
        # Validation
        m = evaluate_model(self.model, self.train_loader, self.device)
        val_iou = m['iou']
        return avg_train_loss, val_iou
    
    
    def _validate_one_epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for b in self.val_loader:
                imgs = b['image'].to(self.device, non_blocking=True)
                masks = b['mask'].to(self.device, non_blocking=True)
                
                preds = self.model(imgs)
                loss = self.crit(preds, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
            
        m = evaluate_model(self.model, self.val_loader, self.device)
        val_iou = m['iou']
        
        return avg_val_loss, val_iou
        
        

    