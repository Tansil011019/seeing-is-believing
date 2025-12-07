from base import Validator

class TransformerValidator(Validator):
    def _get_logits_from_outputs(self, outputs):
        return outputs.logits