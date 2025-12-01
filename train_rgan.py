import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from hydra.utils import instantiate
import torch
import pandas as pd
from datetime import datetime
import os

logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6, 7"

@hydra.main(version_base=None, config_path="config", config_name="augmentation_config")
def train(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")

    df = pd.read_csv(cfg.paths.csv_file)
    label_mapping = cfg.label_map

    filtered_df = df[df['label'].isin(cfg.target_labels)]

    for label in cfg.target_labels:
        device = torch.device(cfg.model.training.device)
        label_df = filtered_df[filtered_df['label'] == label]

        rgan_dataset = instantiate(cfg.dataset.dataset_aug, meta_df=label_df, label_map=label_mapping)
        rgan_loader = instantiate(cfg.dataloader.train_loader, dataset=rgan_dataset)

        generator = instantiate(cfg.model.generator)
        discriminator = instantiate(cfg.model.discriminator)

        datastamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = f"{cfg.model.training.checkpoints_dir}/{label}/{datastamp}"

        save_path = f"{cfg.paths.results_dir}/{cfg.model.name}/{label}/{datastamp}"

        rgan_trainer = instantiate(
            cfg.training,
            generator=generator,
            discriminator=discriminator,
            dataloader=rgan_loader,
            device=device,
            epoch=cfg.model.training.epochs,
            z_dim=cfg.model.generator.z_dim,
            checkpoint_dir=checkpoint_dir,
            save_path=save_path,
            resume_path=cfg.model.training.resume_path,
            lambda_gp=cfg.model.training.lambda_gp,
            lambda_sim=cfg.model.training.lambda_sim,
            generator_lr=cfg.model.training.generator_lr,
            discriminator_lr=cfg.model.training.discriminator_lr,
            generator_training_times=cfg.model.training.generator_training_times,
            discriminator_training_times=cfg.model.training.discriminator_training_times,
            checkpoint_interval=cfg.model.training.checkpoint_interval,
        )

        history = rgan_trainer.train()
        logger.info(f"Finished training R-GAN for label: {label}")

        with open(f"{save_path}/training_history_{label}.csv", "w") as f:
            f.write("epoch,discriminator_loss,generator_loss\n")
            for epoch in range(len(history['discriminator_losses'])):
                f.write(f"{epoch+1},{history['discriminator_losses'][epoch]},{history['generator_losses'][epoch]}\n")

if __name__ == "__main__":
    train()