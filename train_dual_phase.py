import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from hydra.utils import instantiate
import pandas as pd
from torch.utils.data import Subset
import torch
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6, 7"
logging.getLogger("PIL").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

batch_scheduler = ["OneCycleLR"]

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    logger.info(f"Token loaded: {'HF_TOKEN' in os.environ}")
    logger.info("Configuration: ")
    logger.info(f"\t{OmegaConf.to_yaml(cfg)}") 

    torch.manual_seed(cfg.seed)

    df = pd.read_csv(cfg.paths.csv_file)
    unique_labels = df['label'].unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    df_original = df[df['type'] == 'original'].reset_index(drop=True)
    df_augmented = df[df['type'] != 'original'].reset_index(drop=True)

    histories_phase_1 = []
    histories_phase_2 = []
    skf = instantiate(cfg.strategy)
    datastamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for fold_index, (train_indices, val_indices) in enumerate(skf.split(df_original, df_original['label'].values)):
        logger.info(f"Fold {fold_index + 1}/{cfg.strategy.n_splits}")

        val_dataset = instantiate(cfg.dataset.val_dataset, meta_df=df_original, label_map=label_mapping)
        val_subset = Subset(val_dataset, val_indices)
        val_loader = instantiate(cfg.dataloader.val_loader, dataset=val_subset)

        # Phase 1: Train on merged dataset
        logger.info(f"Starting Phase 1 Training for Fold {fold_index + 1}/{cfg.strategy.n_splits}")
        meta_df_train = df_original.iloc[train_indices].copy()
        major_class = meta_df_train['label'].value_counts().idxmax()
        major_class_count = meta_df_train['label'].value_counts().max()

        for label in unique_labels:
            if label == major_class:
                continue
            label_count = meta_df_train['label'].value_counts()[label]
            needed_count = major_class_count - label_count
            augmentations = df_augmented[df_augmented['label'] == label]
            if not augmentations.empty:
                sampled_augmentations = augmentations.sample(n=needed_count, replace=False, random_state=cfg.seed)
                meta_df_train = pd.concat([meta_df_train, sampled_augmentations], ignore_index=True)

        train_dataset = instantiate(cfg.dataset.train_dataset, meta_df=meta_df_train, label_map=label_mapping)
        train_loader = instantiate(cfg.dataloader.train_loader, dataset=train_dataset)
        
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
        
        freeze_ratio = cfg.model.training.get("freeze_ratio", 0.0)
        print("Class count weighting:", cfg.model.training.get("class_count", False))
        class_counts = None
        if cfg.model.training.get("class_count"):
            class_counts = df['label'].value_counts().sort_index().values.tolist()
        trainer_phase_1 = instantiate(
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
            freeze_ratio=freeze_ratio,
            class_count=class_counts if class_counts else None,
            early_stopping=cfg.model.training.early_stopping,
            patience=cfg.model.training.patience,
            min_delta=cfg.model.training.min_delta,
            save_path=f"{cfg.paths.results_dir}/{cfg.model.name}/phase_1/{datastamp}",
            fold_index=fold_index,
            scheduler_step_at_epoch_end=scheduler_step_at_epoch_end
        )

        history_phase_1 = trainer_phase_1.run()
        histories_phase_1.append(history_phase_1)

        # Phase 2: Fine-tune on original dataset
        logger.info(f"Starting Phase 2 Fine-tuning for Fold {fold_index + 1}/{cfg.strategy.n_splits}")
        train_dataset_phase_2 = instantiate(cfg.dataset.train_dataset, meta_df=df_original.iloc[train_indices].copy(), label_map=label_mapping)
        train_loader_phase_2 = instantiate(cfg.dataloader.train_loader, dataset=train_dataset_phase_2)

        optimizer_phase_2 = instantiate(cfg.optimizer, lr=cfg.optimizer.lr / 100, params=model.parameters())
        criterion_phase_2 = instantiate(cfg.loss)

        scheduler_phase_2 = None
        scheduler_step_at_epoch_end = True

        if cfg.scheduler._target_ is not None:
            if batch_scheduler.__contains__(cfg.scheduler._target_.split('.')[-1]):
                scheduler_phase_2 = instantiate(
                    cfg.scheduler, 
                    optimizer=optimizer_phase_2, 
                    steps_per_epoch=len(train_loader_phase_2),
                    epochs=cfg.model.training.epochs
                )
                scheduler_step_at_epoch_end = False
            else:
                scheduler_phase_2 = instantiate(cfg.scheduler, optimizer=optimizer_phase_2)
        
        trainer_phase_2 = instantiate(
            cfg.training,
            model=model,
            model_name=cfg.model.name,
            optimizer=optimizer_phase_2,
            criterion=criterion_phase_2,
            train_loader=train_loader_phase_2,
            val_loader=val_loader,
            device=device,
            epoch=cfg.model.training.epochs,
            scheduler=scheduler_phase_2,
            freeze_ratio=0.0, 
            class_count=class_counts if class_counts else None,
            early_stopping=cfg.model.training.early_stopping,
            patience=cfg.model.training.patience,
            min_delta=cfg.model.training.min_delta,
            save_path=f"{cfg.paths.results_dir}/{cfg.model.name}/phase_2/{datastamp}",
            fold_index=fold_index,
            scheduler_step_at_epoch_end=scheduler_step_at_epoch_end
        )

        history_phase_2 = trainer_phase_2.run()
        histories_phase_2.append(history_phase_2)

    os.makedirs(f"{cfg.paths.results_dir}/{cfg.model.name}//phase_1/{datastamp}", exist_ok=True)
    with open(f"{cfg.paths.results_dir}/{cfg.model.name}//phase_1/{datastamp}/history_{cfg.model.name}.csv", 'w') as f:
        pd.DataFrame(histories_phase_1).to_csv(f, index=False)

    os.makedirs(f"{cfg.paths.results_dir}/{cfg.model.name}//phase_2/{datastamp}", exist_ok=True)
    with open(f"{cfg.paths.results_dir}/{cfg.model.name}//phase_2/{datastamp}/history_{cfg.model.name}.csv", 'w') as f:
        pd.DataFrame(histories_phase_2).to_csv(f, index=False)
    
    logger.info("Training Complete")   

if __name__ == '__main__':
    train()