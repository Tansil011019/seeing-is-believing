"""
Training entrypoint for ISIC2018 Task 2: Attribute Detection
Supports multi-label classification for skin lesion attributes using Hydra
"""
import os
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from hydra.utils import instantiate

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HYDRA_FULL_ERROR"] = "1"
logger = logging.getLogger(__name__)

batch_scheduler = ["OneCycleLR"]

@hydra.main(version_base="1.1", config_path="config", config_name="task2_config")
def train_task2(cfg: DictConfig):
    logger.info("Configuration: ")
    logger.info(f"\t{OmegaConf.to_yaml(cfg)}") 

    torch.manual_seed(cfg.seed)
    
    # Data loaders
    train_dataset = instantiate(cfg.dataset.train_dataset)
    val_dataset = instantiate(cfg.dataset.val_dataset)
    train_loader = instantiate(cfg.dataloader.train_loader, dataset=train_dataset)
    val_loader = instantiate(cfg.dataloader.val_loader, dataset=val_dataset)
    
    # Pretrained model
    device = torch.device(cfg.model.training.device)
    model = instantiate(cfg.model.params)
    model.to(device)
    
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    criterion = instantiate(cfg.loss)

    scheduler = None
    scheduler_step_at_epoch_end = True

    if cfg.scheduler._target_ is not None:
        if batch_scheduler.__contains__(cfg.scheduler._target_.split('.')[-1]):
            scheduler = instantiate(
                cfg.scheduler, 
                optimizer=optimizer, 
                steps_per_epoch=len(train_loader),
                epochs=cfg.model.training.epochs
            )
            scheduler_step_at_epoch_end = False
        else:
            scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
            
    
    # Trainer
    datastamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trainer = instantiate(
        cfg.training,
        model=model,
        model_name=cfg.model.name,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epoch=cfg.model.training.epochs,
        scheduler=scheduler,
        early_stopping=cfg.model.training.early_stopping,
        patience=cfg.model.training.patience,
        min_delta=cfg.model.training.min_delta,
        save_path=cfg.model.training.save_path,
        fold_index=None,
        scheduler_step_at_epoch_end=scheduler_step_at_epoch_end
    )
    
    history = trainer.run()

    with open(f"{cfg.paths.results_dir}/{cfg.model.name}/{datastamp}/history_{cfg.model.name}.csv", 'w') as f:
        pd.DataFrame(history).to_csv(f, index=False)
    
    logger.info("Task 2 Training Completed.")
    
    

if __name__ == "__main__":
    train_task2()