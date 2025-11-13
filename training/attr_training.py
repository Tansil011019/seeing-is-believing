"""
Training utilities for attribute classification pipeline
"""
import torch
from tqdm import tqdm
from typing import Dict


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    use_amp: bool = True,
    scaler: torch.cuda.amp.GradScaler = None,
    model_name: str = '',
    epoch: int = 1,
    total_epochs: int = 1
) -> float:
    """
    Train model for one epoch
    
    Args:
        model: Model to train
        train_loader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device for computation
        use_amp: Use automatic mixed precision
        scaler: GradScaler for AMP
        model_name: Model name for logging
        epoch: Current epoch number
        total_epochs: Total number of epochs
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    train_loss = 0.0
    
    pbar = tqdm(
        train_loader,
        desc=f"[{model_name}] Ep {epoch}/{total_epochs} [Train]"
    )
    
    for batch in pbar:
        imgs = batch['image'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
        
        # Backward pass with gradient scaling
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return train_loss / len(train_loader)
