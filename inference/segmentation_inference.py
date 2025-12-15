import torch
import cv2
import numpy as np
from scipy import ndimage
from typing import Dict, Tuple
from models.seg_models import get_model

class SegmentationInference:
    
    def __init__(self, 
                 model_name: str, 
                 checkpoint_path: str, 
                 device: str = 'cuda'):
        """
        Initialize segmentation inference pipeline
        
        Args:
            model_name: Name of the segmentation model (e.g., 'segformer_b0')
            checkpoint_path: Path to the model checkpoint
            device: Device to run inference on
            num_classes: Number of segmentation classes
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = get_model(model_name)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        

    def infer(self, image: torch.Tensor) -> Dict[str, float]:
        """
        Run segmentation inference and calculate ABCD metrics
        
        Args:
            image: Input image tensor (B, C, H, W) or (C, H, W)
            
        Returns:
            Dictionary containing segmentation metrics (asymmetry, border irregularity)
        """
        
        # Run segmentation
        self.model.eval()
        predicted_mask = None
        with torch.no_grad():
            image = image.to(self.device)
            image = image.unsqueeze(0)
            output = self.model(image)
            
            # Get predicted mask
            predicted_mask = (torch.sigmoid(output) > 0.5).long().squeeze(1)
        
        if predicted_mask is None :
            raise ValueError("Segmentation inference failed, predicted_mask is None")
        
        # Convert predicted mask to numpy array, must be just (H, W)
        predicted_mask = predicted_mask.squeeze(0).cpu().numpy().astype(np.uint8)
        seg_dict = {}
        seg_dict['asymmetry_index'] = self.calculate_asymmetry(predicted_mask)
        seg_dict['border_irregularity'] = self.calculate_border_irregularity(predicted_mask)
        
        return seg_dict
    
    def calculate_asymmetry(self, mask: np.ndarray) -> float:
        """
        Calculate asymmetry index using PCA and folding technique
        
        Formula: AI = Area(L ⊕ R') / Area(M)
        Where L is left half, R is right half, R' is flipped R, ⊕ is XOR
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Asymmetry index (0 = symmetric, higher = more asymmetric)
        """
        if mask.sum() == 0:
            return 0.0
        
        # Find centroid using image moments
        moments = cv2.moments(mask)

        if moments['m00'] == 0: # Area  is zero
            return 0.0
            
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        # Calculate orientation using PCA
        # Get coordinates of foreground pixels
        y_coords, x_coords = np.where(mask > 0)
        coords = np.column_stack([x_coords, y_coords]) # Coordinates of foreground pixels
        coords_centered = coords - [cx, cy] # Centered coordinatres
        
        if len(coords_centered) < 2:
            return 0.0
        
        # Compute covariance matrix
        cov_matrix = np.cov(coords_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Principal axis angle
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
        theta = np.arctan2(principal_axis[1], principal_axis[0])
        
        # Rotate mask to align principal axis with vertical
        rotation_matrix = cv2.getRotationMatrix2D((cx, cy), -np.degrees(theta), 1.0)
        h, w = mask.shape
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        
        # Split into left and right halves along vertical axis
        left_half = rotated_mask[:, :w//2].copy()
        right_half = rotated_mask[:, w//2:].copy()
        
        # Flip right half horizontally
        right_half_flipped = cv2.flip(right_half, 1)
        
        # Resize to match dimensions if needed
        if left_half.shape[1] != right_half_flipped.shape[1]:
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
        
        # Calculate XOR (non-overlapping area)
        xor_area = np.sum(np.logical_xor(left_half, right_half_flipped))
        total_area = np.sum(rotated_mask > 0)
        
        if total_area == 0:
            return 0.0
        
        # Asymmetry index
        asymmetry_index = xor_area / total_area
        
        return float(asymmetry_index)
        
    def calculate_border_irregularity(self, mask: np.ndarray) -> float:
        """
        Calculate border irregularity using compactness score
        
        Formula: C = P² / (4π × A)
        Where P is perimeter, A is area
        Perfect circle has C = 1, irregular shapes have C >> 1
        
        Args:
            mask: Binary mask (H, W)
            
        Returns:
            Compactness score (1 = smooth circle, higher = more irregular)
        """
        if mask.sum() == 0:
            return 1.0
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) == 0:
            return 1.0
        
        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate perimeter and area
        perimeter = cv2.arcLength(contour, closed=True)
        area = cv2.contourArea(contour)
        
        if area == 0:
            return 1.0
        
        # Compactness score (isoperimetric quotient)
        compactness = (perimeter ** 2) / (4 * np.pi * area)
        
        return float(compactness)
    
    
        
        