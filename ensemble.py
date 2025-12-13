import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from hydra.utils import instantiate, to_absolute_path
import logging
import pandas as pd
from dotenv import load_dotenv
from helpers import prior_adjustment

load_dotenv()
logger = logging.getLogger(__name__)

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

@hydra.main(config_path="config", config_name="ensemble_config", version_base=None)
def main(cfg: DictConfig):
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)

    df = pd.read_csv(cfg.paths.csv_file)
    unique_labels = df['label'].unique()
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    test_dataset = instantiate(cfg.dataset.test_dataset, label_map=label_mapping)
    test_loader = instantiate(cfg.dataloader.test_loader, dataset=test_dataset)

    ensemble_model = instantiate(cfg.ensemble, ensemble_config=cfg.model_defs, device=cfg.device)
    predictions, probabilities, meta_features = ensemble_model.predict(test_loader) 

    if cfg.get('use_prior_adjustment', True):
        logger.info("Applying prior adjustment to probabilities")

        TRAIN_COUNTS = [1113, 6705, 514, 327, 1099, 115, 142]
        probabilities = prior_adjustment(probabilities, TRAIN_COUNTS)

        final_preds = probabilities.argmax(axis=1)
        logger.info("Prior Adjustment Complete.")

    feature_cols = []
    for model_name in cfg.ensemble.model_order:
        for class_name in CLASS_NAMES:
            feature_cols.append(f"{model_name}_{class_name}")
    
    meta_df = pd.DataFrame(meta_features, columns=feature_cols)
    meta_df.insert(0, 'image', df['image'])

    meta_save_path = f"{cfg.paths.results_dir}/meta_features.csv"
    meta_df.to_csv(meta_save_path, index=False)
    logger.info(f"Saved meta-features to {meta_save_path}")

    if probabilities.ndim > 1:
        probabilities = probabilities.max(axis=1)

    results_df = pd.DataFrame({
        'image': df['image'],
        'prediction': final_preds,
        'probability': probabilities
    })
    results_save_path = f"{cfg.paths.results_dir}/ensemble_predictions.csv"
    results_df.to_csv(results_save_path, index=False)
    logger.info(f"Saved final predictions to {results_save_path}")

if __name__ == "__main__":
    main()  