from tqdm import tqdm
import torch
import logging
from datetime import datetime
from torch import softmax
from sklearn.metrics import accuracy_score, log_loss

logger = logging.getLogger(__name__)

class Validator:
    def __init__(
        self,
        model,
        model_name,
        dataloader,
        device,
        **kwargs
    ):
        self.model = model
        self.model_name = model_name
        self.dataloader = dataloader
        self.device = device
    
        if kwargs:
            logging.warning(f"Unused Validator parameters: {kwargs.keys()}")

    def _get_logits_from_outputs(self, outputs):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def _validate(self):
        self.model.eval()
        all_probs = []
        all_labels = []
        all_images = []
        with torch.no_grad():
            for images, labels, ids in tqdm(self.dataloader, desc=f"Validating {self.model_name}"):
                inputs = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                logits = self._get_logits_from_outputs(outputs)
                probs = softmax(logits, dim=1)

                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
                all_images.extend(ids)

        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)

        all_probs = all_probs.numpy()
        all_labels = all_labels.numpy()
        all_preds = torch.argmax(torch.tensor(all_probs), dim=1).numpy()

        accuracy = accuracy_score(all_labels, all_preds)

        num_classes = all_probs.shape[1]
        if num_classes == 2:
            loss = log_loss(all_labels, all_probs[:, 1])
        else:
            loss = log_loss(all_labels, all_probs)

        return all_probs, all_labels, all_images, accuracy, loss
    
    def run(self):
        return self._validate() 
    