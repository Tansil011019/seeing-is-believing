from tqdm import tqdm
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

# TODO: Early Stopping
class Trainer:
    def __init__(
        self, 
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader, 
        device,
        epoch,
        early_stopping = False
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epoch = epoch
        self.early_stopping = early_stopping
    
    def _train_one_epoch(self):
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.items()
            _, predicted = torch.max(outputs.data, 1)
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
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
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

        logger.info("Training Complete")
        
        history = {
            "training_loss": train_losses,
            "training_accuracy": train_accs,
            "val_loss": val_losses,
            "val_accuracy": val_accs
        }

        return history