import hydra
from omegaconf import DictConfig, OmegaConf 
from hydra.utils import instantiate, to_absolute_path
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from helpers import prior_adjustment
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config_meta", version_base=None)
def main(cfg: DictConfig):
    logger.info("Configuration: ")
    logger.info(f"\t{OmegaConf.to_yaml(cfg)}") 

    input_path = to_absolute_path(cfg.paths.input_path)
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
        logger.info(f"Loaded data from {input_path} with shape {df.shape}")
    elif input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
        logger.info(f"Loaded data from {input_path} with shape {df.shape}")
    
    model = instantiate(cfg.model.params)
    ignore_columns = cfg.ignore_columns
    feature_columns = [c for c in df.columns if c not in ignore_columns]

    logger.info(f"Using {len(feature_columns)} features for training.")
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    splitter = instantiate(cfg.strategy)
    X_train, X_val, y_train, y_val = splitter.split(df, feature_columns, cfg.target_column)

    trainer = instantiate(
        cfg.training,
        model=model,
        use_eval_set=cfg.model.training.use_eval_set,
        verbose=cfg.model.training.verbose
    )

    trainer.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val
    )

    val_pred_raw = trainer.predict_proba(X_val)
    acc_raw = accuracy_score(y_val, val_pred_raw.argmax(axis=1))
    f1_raw = f1_score(y_val, val_pred_raw.argmax(axis=1), average='weighted')
    logger.info(f"Validation Accuracy: {acc_raw}")
    logger.info(f"Validation F1 Score: {f1_raw}")

    if cfg.use_prior_adjustment: 
        train_counts = y_train.value_counts().sort_index().tolist()
        val_pred_adjusted = prior_adjustment(val_pred_raw, train_counts)
        acc_adjusted = accuracy_score(y_val, val_pred_adjusted.argmax(axis=1))
        f1_adjusted = f1_score(y_val, val_pred_adjusted.argmax(axis=1), average='weighted')
        logger.info(f"Validation Accuracy after Prior Adjustment: {acc_adjusted}")
        logger.info(f"Validation F1 Score after Prior Adjustment: {f1_adjusted}")
    
    results_dir = to_absolute_path(f"{cfg.paths.results_dir}/{cfg.model.name}/{time_stamp}")
    os.makedirs(results_dir, exist_ok=True)
    model_save_path = os.path.join(results_dir, f"{cfg.model.name}_{time_stamp}.pkl")
    trainer.save(model_save_path)
    logger.info(f"Model saved at {model_save_path}")


if __name__ == "__main__":
    main()