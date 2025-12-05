import torch
from tqdm import tqdm
import logging
from evaluation.attr_metrics import evaluate_attr_model
from base.trainer import Trainer

import torch.nn as nn

logger = logging.getLogger(__name__)

class AttributeTrainer(Trainer):
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
            labels = b['labels'].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            logits = self.model(imgs)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = train_loss / len(self.train_loader)
        
        # Validation metrics
        m = evaluate_attr_model(self.model, self.val_loader, self.criterion, self.device)
        val_f1_macro = m['f1_macro']
        
        return avg_train_loss, val_f1_macro
    
    
    def _validate_one_epoch(self):
        self.model.eval()
        m = evaluate_attr_model(self.model, self.val_loader, self.criterion, self.device)
        val_loss = m['loss']
        val_f1_macro = m['f1_macro']
        
        return val_loss, val_f1_macro
