from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import copy
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class Trainer:
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
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epoch = epoch
        self.save_path = save_path
        self.scheduler = scheduler
        self.scheduler_step_at_epoch_end = scheduler_step_at_epoch_end
        self.fold_index = fold_index
        if early_stopping:
            self.early_stopping = early_stopping
            assert patience > 0, "Patience must be a positive integer greater than 0."
            self.patience = patience
            self.min_delta = min_delta
            self._patience_counter = 0
            self._best_val_loss = float('inf')
            self._best_model_state = None
    
    def _train_one_epoch(self):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs.logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if not self.scheduler_step_at_epoch_end and self.scheduler is not None:
                self.scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss_avg = train_loss / len(self.train_loader)
        train_acc = 100 * correct / total

        return train_loss_avg, train_acc

    def _validate_one_epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs.logits, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        return val_loss / len(self.val_loader), val_acc

    def run(self):
        logger.info(f"Starting training for {self.epoch} epochs")
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        best_epoch = -1
        for epoch in range(self.epoch):
            logger.info(f"Epoch {epoch+1}/{self.epoch}")
            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._validate_one_epoch()
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            logger.info(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )

            if self.scheduler_step_at_epoch_end and self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if val_loss < self._best_val_loss - self.min_delta:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                self._best_model_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                logger.info(f"Validation loss improved. Saving model.")
            else:
                self._patience_counter += 1
                logger.info(f"No improvement. Patience counter: {self._patience_counter}/{self.patience}")

            if self._patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs.")
                logger.info(f"Loading best model state (Val Loss: {self._best_val_loss:.4f})")

                if self._best_model_state:
                    self.model.load_state_dict(self._best_model_state) 

                break

        logger.info("Training Complete")
        
        history = {
            "training_loss": train_losses,
            "training_accuracy": train_accs,
            "val_loss": val_losses,
            "val_accuracy": val_accs
        }

        if self.save_path:
            folder_path = os.path.join(self.save_path, self.model_name)
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"Saving best model to: {folder_path}")
            
            datastamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            fold_part = f"fold_{self.fold_index}_" if self.fold_index is not None else ""
            file_name = f"{self.model_name}_best_{datastamp}_{fold_part}{best_epoch}.pt"
            full_save_path = os.path.join(folder_path, file_name)

            torch.save(self.model.state_dict(), full_save_path)

        return history