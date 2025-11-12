import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from hydra.utils import instantiate
import pandas as pd
from torch.utils.data import Subset
import torch
from base import Trainer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
logger = logging.getLogger(__name__)

batch_scheduler = ["OneCycleLR"]

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    logger.info("Configuration: ")
    logger.info(f"\t{OmegaConf.to_yaml(cfg)}") 

    torch.manual_seed(cfg.seed)

    df = pd.read_csv(cfg.paths.csv_file)
    unique_labels = df['label'].unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    train_dataset = instantiate(cfg.dataset.train_dataset, label_map=label_mapping)
    val_dataset = instantiate(cfg.dataset.val_dataset, label_map=label_mapping)

    histories = []
    skf = instantiate(cfg.strategy)
    for fold_index, (train_indices, val_indices) in enumerate(skf.split(df, df['label'].values)):
        logger.info(f"Fold {fold_index + 1}/{cfg.strategy.n_splits}")
        
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

        train_loader = instantiate(cfg.dataloader.train_loader, dataset=train_subset)
        val_loader = instantiate(cfg.dataloader.val_loader, dataset=val_subset)
        
        device = torch.device(cfg.model.training.device)
        model = instantiate(cfg.model.params)
        model = model.to(device)

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
                    epochs=cfg.model.traning.epochs
                )
                scheduler_step_at_epoch_end = False
            else:
                scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

        trainer = Trainer(
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
            save_path=f"{cfg.paths.results_dir}",
            fold_index=fold_index,
            scheduler_step_at_epoch_end=scheduler_step_at_epoch_end
        )

        history = trainer.run()
        histories.append(history)

        os.makedirs(cfg.paths.results_dir, exist_ok=True)

    with open(f"{cfg.paths.results_dir}/{cfg.model.name}/history_{cfg.model.name}.csv", 'w') as f:
        pd.DataFrame(histories).to_csv(f, index=False)
    
    logger.info("Training Complete")   

if __name__ == '__main__':
    train()