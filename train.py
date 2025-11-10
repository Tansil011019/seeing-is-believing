import hydra
from omegaconf import DictConfig, OmegaConf
import logging

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:
    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
    pass

if __name__ == '__main__':
    train()