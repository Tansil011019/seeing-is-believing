import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import pandas as pd
import os
from datetime import datetime

logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="generate_meta_dataset", version_base=None)
def main(cfg: DictConfig):
    logger.info("Configuration: ")
    logger.info(f"\t{OmegaConf.to_yaml(cfg)}") 

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_cols = ['image', 'target']
    master = False
    df_master = None
    for key, values in cfg.files.items():
        if not master:
            df_original = pd.read_csv(values)
            df_master = df_original[meta_cols].copy()
            master = True
        model = key
        logger.info(f"Processing model: {model}")
        df_model = pd.read_csv(values)
        prob_columns = [col for col in df_model.columns if col.endswith('_prob')]
        df_subset = df_model[['image'] + prob_columns].copy()
        new_names = {col: f"{model}_{col}" for col in prob_columns}
        df_subset.rename(columns=new_names, inplace=True)
        df_master = df_master.merge(df_subset, on='image', how='left')

    df_meta = df_master
    logger.info(f"Generated meta-dataset with shape: {df_meta.shape}")

    output_folder = cfg.get('output_meta_dataset_folder', '.')
    os.makedirs(output_folder, exist_ok=True)
    output_file_csv = os.path.join(output_folder, f'{time_stamp}_meta_dataset.csv')
    output_file_parquet = os.path.join(output_folder, f'{time_stamp}_meta_dataset.parquet')
    df_meta.to_csv(output_file_csv, index=False)
    df_meta.to_parquet(output_file_parquet, index=False)
    logger.info(f"Meta-dataset saved to {output_file_csv} and {output_file_parquet}")

if __name__ == "__main__":
    main()