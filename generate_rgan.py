import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
from tqdm import tqdm
from hydra.utils import instantiate, to_absolute_path
import logging
import torchvision

logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="generate_rgan")
def main(cfg: DictConfig):
    logger.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")

    device = torch.device(cfg.device)
    generator = instantiate(cfg.model.generator)
    generator = generator.to(device)
    generator.eval()

    target_labels = cfg.target_labels
    tasks = cfg.tasks

    for class_name in target_labels:
        if class_name not in tasks:
            logger.warning(f"Class {class_name} not found in tasks. Skipping.")
            continue

        params = tasks[class_name]
        logger.info(f"Generating samples for class: {class_name}")
        logger.info(f"Parameters: {params}")

        path = to_absolute_path(params.path)

        if not os.path.exists(path):
            logger.error(f"Path {path} does not exist. Skipping.")
            continue

        try:
            checkpoint = torch.load(path, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            logger.info(f"Loaded checkpoint from {path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {path}: {e}")
            continue
            
        save_folder = os.path.join(cfg.save_path, class_name)
        os.makedirs(save_folder, exist_ok=True)

        generated_count = 0
        progress_bar = tqdm(total=params.target_counts, desc=f"Generating {class_name}")

        while generated_count < params.target_counts:
            current_batch_size = min(cfg.batch_size, params.target_counts - generated_count)

            with torch.no_grad():
                z = torch.randn(current_batch_size, cfg.model.generator.z_dim).to(device)
                generated_imgs = generator(z)
                generated_imgs = (generated_imgs + 1) / 2.0 

                for i, img in enumerate(generated_imgs):
                    file_number = generated_count + i + 1
                    img_path = os.path.join(save_folder, f"{class_name}_{file_number}.png")
                    torchvision.utils.save_image(img, img_path)
                
            generated_count += current_batch_size
            progress_bar.update(current_batch_size)
        progress_bar.close()

if __name__ == "__main__":
    main()