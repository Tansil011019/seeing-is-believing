import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from PIL import Image
import cv2
import pandas as pd
from utils.hydra_factory.ensemble_xai import get_xai

class ClassificationInference:
    """
    Classification inference with gradient-based visualization (Grad-CAM)
    """
    
    # Class mapping for melanoma attributes (ISIC Task 2)
    CLASS_MAPPING = {
        0: 'Melanoma',
        1: 'Melanocytic nevus',
        2: 'Basal cell carcinoma',
        3: 'Actinic keratosis / Bowen’s disease (intraepithelial carcinoma)',
        4: 'Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)',
        5: 'Dermatofibroma',
        6: 'Vascular lesion',
    }
    
    def __init__(self, 
                 model_name: str, 
                 checkpoint_path: str, 
                 num_classes: int = 7, 
                 device: str = 'cuda'):
        """
        Initialize classification inference pipeline
        
        Args:
            model_name: Name of the classification model (e.g., 'resnet50', 'efficientnet_b0')
            checkpoint_path: Path to the model checkpoint
            num_classes: Number of classification classes
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # Load integrated xai ensemble wrapper
        self.xai = get_xai()
        
    def infer(self, 
              image: torch.Tensor, 
              image_path: str = None) -> Dict[str, any]:
        """
        Run classification inference and compute gradient visualization
        
        Args:
            image: Input image tensor (C, H, W) or (B, C, H, W)
            
        Returns:
            Dictionary containing:
                - predictions: Class predictions as text list
                - probabilities: Prediction probabilities
                - gradient_heatmap: Gradient-based heatmap (H, W)
                - gradient_heatmap_raw: Raw gradient values
        """
        # TODO : CHANGE LATER : PLACEHOLDER FOR CLASSIFICATION INFERENCE
        df_pred = pd.read_csv('outputs/ensemble_predictions_old.csv')
        
        if image_path is not None:
            img_id = image_path.split('/')[-1].split('.')[0]
            print("TEST IMG ID:", img_id)
            pred_row = df_pred[df_pred['image'] == img_id]
            if not pred_row.empty:
                pred_class = pred_row.iloc[0]['prediction']
                predictions_text = self.CLASS_MAPPING[pred_class]
            else:
                predictions_text = self.CLASS_MAPPING[0]
        else:
            predictions_text = self.CLASS_MAPPING[0]
        
        probabilities = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.025, 0.025])  # Dummy probabilities
        
        heatmap_path = "outputs/heatmap/ISIC_0000000.png" # Initnya gradient_heatmap harus PIL.Image
        gradient_heatmap = np.array(Image.open(heatmap_path).convert("RGB"))
        
        
        return {
            'prediction_text': predictions_text,
            'gradient_heatmap': gradient_heatmap,
        }