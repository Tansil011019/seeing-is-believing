from base import Trainer
import logging

logger = logging.getLogger(__name__)

class TransformerTrainer(Trainer):
    def __init__(
        self,
        model,
        freeze_ratio=0.0,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        if freeze_ratio > 0.0:
            self.__freeze_layers(freeze_ratio)

    def __freeze_layers(self, freeze_ratio):
        params = list(self.model.parameters())
        freeze_count = int(len(params) * freeze_ratio)

        logger.info(f"Freezing {freeze_count} out of {len(params)} layers.")

        for i, param in enumerate(params):
            if i < freeze_count:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def _get_logits_from_outputs(self, outputs):
        return outputs.logits