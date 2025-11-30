from base import Trainer
import logging
import torch

logger = logging.getLogger(__name__)

class TimmTrainer(Trainer):
    def __init__(
        self,
        model, 
        freeze_ratio=0.0,
        class_count=None,
        **kwargs
    ):
        if class_count:
            logger.info("Using class-weighted CrossEntropyLoss.")
            device = kwargs.get('device', 'cpu')
            weights = 1.0 / torch.tensor(class_count, dtype=torch.float32).to(device)
            weights = weights / weights.sum()

            criterion = torch.nn.CrossEntropyLoss(weight=weights)
            kwargs['criterion'] = criterion

        super().__init__(
            model = model, 
            **kwargs
        )

        if freeze_ratio > 0.0:
            self._freeze_layers(freeze_ratio)

    def _freeze_layers(self, freeze_ratio):
        params = list(self.model.parameters())
        freeze_count = int(len(params) * freeze_ratio)

        logger.info(f"Freezing {freeze_count} out of {len(params)} layers.")

        for i, param in enumerate(params):
            if i < freeze_count:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def _get_logits_from_outputs(self, outputs):
        return outputs