from base import Trainer

class TransformerTrainer(Trainer):
    def _get_logits_from_outputs(self, outputs):
        return outputs.logits