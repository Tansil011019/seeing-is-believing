from base import Trainer

class TimmTrainer(Trainer):
    def _get_logits_from_outputs(self, outputs):
        return outputs