import joblib
import logging
from hydra.utils import instantiate, to_absolute_path
from glob import glob
import os
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import ListConfig

logger = logging.getLogger(__name__)

class StackingEnsemble:
    def __init__(self, device, model_order, meta_model_path: str, base_model_paths: list, ensemble_config: dict):
        self.device = device
        self.base_model_paths = base_model_paths
        self.model_defs = ensemble_config
        self.model_order = model_order

        logger.info(f"Loading meta-model from {meta_model_path}")
        self.meta_model = joblib.load(to_absolute_path(meta_model_path))
        if hasattr(self.meta_model, 'set_params'):
            self.meta_model.set_params(device=device)
    
    def _find_weights(self, dir_paths):
        if isinstance(dir_paths, str):
            dir_paths = [dir_paths]

        weights_files = []
        for dir_path in dir_paths:
            logger.info(f"Searching for weight files in {dir_path}")
            abs_path = to_absolute_path(dir_path)
            weights_files.extend(glob(os.path.join(abs_path, "**", "*.pt"), recursive=True))

        return sorted(list(set(weights_files)))
    
    def _predict_single_model(self, model, data_loader):
        model.eval()
        all_outputs = []

        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Predicting", leave=False)
            for images, _, _ in progress_bar:
                inputs = images.to(self.device)
                outputs = model(inputs)
                outputs = getattr(outputs, 'logits', outputs)
                
                probs = torch.softmax(outputs, dim=1)
                all_outputs.append(probs.cpu().numpy())

        return np.concatenate(all_outputs, axis=0)

    def predict(self, data_loader):
        meta_features = []
    
        for model_key in self.model_order:
            logger.info(f"Processing model: {model_key}")

            paths_to_use = []

            if model_key in self.base_model_paths:
                paths = self.base_model_paths[model_key]
                if isinstance(paths, (list, ListConfig)):
                    paths_to_use = paths
                else:
                    paths_to_use = [paths]
            else:
                try:
                    base_name, suffix = model_key.rsplit('_', 1)
                    idx = int(suffix) - 1
                    if base_name in self.base_model_paths:
                        full_list = self.base_model_paths[base_name]
                        if 0 <= idx < len(full_list):
                            paths_to_use = [full_list[idx]]
                        else:
                            raise IndexError(f"Index {idx} out of range for {base_name}")
                except (ValueError, IndexError):
                    pass

            if not paths_to_use:
                raise KeyError(f"Could not resolve path for '{model_key}'. Check config spelling or indices.")
        
            model_path = paths_to_use
            logger.info(f"Processing: {model_key} from {model_path}")

            def_key = model_key
            if model_key not in self.model_defs:
                candidate = model_key.split('_')[0]
                if candidate in self.model_defs:
                    def_key = candidate
            
            if def_key not in self.model_defs:
                logger.error(f"Model definition for {model_key} not found in ensemble config.")
                continue

            model_cfg = self.model_defs[def_key]
            logger.info(f"Using model config for: {def_key}, {model_cfg.keys()}")
            folds = self._find_weights(model_path)
            if not folds:
                logger.error(f"No weight files found for model {model_key} in path {model_path}.")
                continue

            fold_predictions = []
            for ckpt_path in folds:
                model = model_cfg['params']
                model = model.to(self.device)
                logger.info(f"Loading weights from {ckpt_path}")
                model.load_state_dict(torch.load(ckpt_path, map_location=self.device))

                preds = self._predict_single_model(model, data_loader)
                fold_predictions.append(preds)

                del model
                torch.cuda.empty_cache()

            avg_predictions = sum(fold_predictions) / len(fold_predictions)
            meta_features.append(avg_predictions)
        
        X_meta_numpy = np.hstack(meta_features)
        X_meta = torch.from_numpy(X_meta_numpy).float().to(self.device)
        logger.info(f"Meta Input Shape: {X_meta.shape}")

        final_preds = self.meta_model.predict(X_meta)
        final_probs = self.meta_model.predict_proba(X_meta)

        return final_preds, final_probs, X_meta_numpy