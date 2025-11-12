import hydra
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    logger.info("Configuration: ")
    logger.info(f"\t{OmegaConf.to_yaml(cfg)}") 

    logger.info("Phase 1: Dataloader")
    dataloader_config = cfg.dataloader
    train_loader = hydra.utils.instantiate(dataloader_config.train_loader)
    val_loader = hydra.utils.instantiate(dataloader_config.val_loader)
    logger.info("Checking DataLoaders...")
    try:
        # 1. Check training loader
        logger.info(f"Train dataset size: {len(train_loader.dataset)}")
        images, labels = next(iter(train_loader))
        logger.info("Train loader check PASS. Batch details:")
        logger.info(f"  Images shape: {images.shape}, dtype: {images.dtype}")
        logger.info(f"  Labels shape: {len(labels)}")
        logger.info(f"  Images min: {images.min():.2f}, max: {images.max():.2f}")

        # 2. Check validation loader
        logger.info(f"Val dataset size: {len(val_loader.dataset)}")
        images, labels = next(iter(val_loader))
        logger.info("Val loader check PASS. Batch details:")
        logger.info(f"  Images shape: {images.shape}, dtype: {images.dtype}")
        logger.info(f"  Labels shape: {len(labels)}")
        logger.info(f"  Images min: {images.min():.2f}, max: {images.max():.2f}")        
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return 
    logger.info("Dataloader Loaded")

    logger.info("Phase 2: Training")
    
    logger.info("Training Complete")   

if __name__ == '__main__':
    train()