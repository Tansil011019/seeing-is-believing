from omegaconf import DictConfig
from hydra.utils import instantiate
from fusion_xai import EnsembleXAI
from hydra import initialize, compose

def build_ensemble_xai(cfg: DictConfig) -> EnsembleXAI:
    """
    Build EnsembleXAI from a fully composed Hydra config
    """
    ensemble_wrapper = instantiate(
        cfg.ensemble,
        ensemble_config=cfg.model_defs,
        device=cfg.device
    )

    xai = EnsembleXAI(ensemble_wrapper)
    return xai



def get_xai():
    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name="ensemble_config")

    return build_ensemble_xai(cfg)
