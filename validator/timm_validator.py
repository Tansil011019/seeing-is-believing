from base import Validator

class TimmValidator(Validator):
    def _get_logits_from_outputs(self, outputs):
        return outputs