import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import os
from hydra.utils import to_absolute_path, instantiate
import pandas as pd
from datetime import datetime
from torch.utils.data import Subset
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="generate_oof", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"Token loaded: {'HF_TOKEN' in os.environ}")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")

    model_folder = cfg.model_folder
    save_path = cfg.save_path

    if not os.path.isdir(model_folder):
        logger.error(f"Model folder {model_folder} does not exist.")
        return
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Created save path directory at {save_path}")

    torch.manual_seed(cfg.seed)

    df = pd.read_csv(to_absolute_path(cfg.paths.csv_file))
    unique_labels = df['label'].unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    skf = instantiate(cfg.strategy)
    val_dataset = instantiate(cfg.dataset.train_dataset, label_map=label_mapping)

    model_list = [model for model in os.listdir(model_folder) if model.endswith(".pt")]
    logger.info(f"Found {len(model_list)} models in {model_folder}")
    logger.info(
        "Model list:\n - " + "\n - ".join(model_list)
    )
    datastamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    final_probs = []
    final_labels = []
    final_images = []
    final_folds = []
    fold_metrics = []

    for fold_index, (_, val_indices) in enumerate(skf.split(df, df['label'].values)):
        logger.info(f"Fold {fold_index + 1}/{cfg.strategy.n_splits}")

        val_subset = Subset(val_dataset, val_indices)
        val_loader = instantiate(cfg.dataloader.val_loader, dataset=val_subset)

        device = torch.device(cfg.device)
        model_name = model_list[fold_index]

        logger.info(f"Processing model: {model_name}")
        model_path = os.path.join(model_folder, model_name)
        model = instantiate(cfg.model.params)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            logger.error(f"Model file {model_path} does not exist.")
            return
        model = model.to(device)
        model.eval()

        validator = instantiate(
            cfg.validation,
            model_name=cfg.model.name,
            model=model,
            dataloader=val_loader,
            device=device,
        )

        probs, labels, images, accuracy, loss = validator.run()
        logger.info(f"Fold {fold_index} | Accuracy: {accuracy:.4f} | Loss: {loss:.4f}")

        final_probs.append(probs)
        final_labels.append(labels)
        final_images.extend(images)

        final_folds.extend([fold_index] * len(images))
        fold_metrics.append({
            'fold': fold_index, 
            'model_name': cfg.model.name,
            'accuracy': accuracy, 
            'loss': loss
        })
    
    all_probs = np.vstack(final_probs)
    all_targets = np.concatenate(final_labels)

    cols = [f"class_{i}_prob" for i in range(all_probs.shape[1])]
    results_df = pd.DataFrame(all_probs, columns=cols)
    results_df['target'] = all_targets
    results_df['image'] = final_images
    results_df['fold'] = final_folds

    os.makedirs(f"{save_path}/{cfg.model.name}", exist_ok=True)
    
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(f"{save_path}/{cfg.model.name}/fold_metrics_{datastamp}.csv", index=False)
    logger.info(f"Saved fold metrics to {save_path}/{cfg.model.name}/fold_metrics_{datastamp}.csv")

    results_df.to_csv(f"{save_path}/{cfg.model.name}/oof_predictions_{datastamp}.csv", index=False)
    logger.info(f"Saved OOF predictions to {save_path}/{cfg.model.name}/oof_predictions_{datastamp}.csv")

if __name__ == "__main__":
    main()