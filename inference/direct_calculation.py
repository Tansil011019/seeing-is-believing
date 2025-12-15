import cv2
import numpy as np
from typing import Dict, Union
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


class DirectCalculation:
    """
    Direct calculation of ABCD rule features from input images
    C: Color Variegation
    D: Differential Structures (Texture)
    """
    
    def __init__(self):
        """Initialize direct calculation pipeline"""
        pass
    
    def calculate_color_variegation(self, 
                                    image: Union[np.ndarray, Image.Image]) -> Dict[str, Union[float, int]]:
        """
        Calculate color variegation using K-means clustering in CIELAB color space
        
        Process:
        1. Convert to CIELAB color space
        2. Apply K-means clustering (K=6 for diagnostic colors)
        3. Filter clusters < 5% of lesion area
        4. Count valid clusters and calculate color deviation
        
        Args:
            image: Input image (H, W, C) numpy array or PIL Image
            mask: Optional binary mask to focus on lesion region (H, W)
            
        Returns:
            Dictionary with:
                - n_colors: Number of distinct color clusters
                - color_std: Standard deviation of colors from mean skin tone
                - dominant_colors: List of dominant color cluster centers
        """
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB format
        if image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
        
        h, w, c = image.shape
        
        
        # If no mask, use center region as lesion approximation
        center_h, center_w = h // 4, w // 4
        lesion_pixels = image[center_h:3*center_h, center_w:3*center_w].reshape(-1, 3)
        
        if len(lesion_pixels) == 0:
            return {
                'n_colors': 0,
                'color_std': 0.0,
                'dominant_colors': []
            }
        
        # Convert to CIELAB color space (perceptually uniform)
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lesion_pixels_lab = image_lab[center_h:3*center_h, center_w:3*center_w].reshape(-1, 3)
        
        # K-means clustering (K=6 for typical diagnostic colors)
        K = 6
        lesion_pixels_lab_float = lesion_pixels_lab.astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            lesion_pixels_lab_float, 
            K, 
            None, 
            criteria, 
            10, 
            cv2.KMEANS_PP_CENTERS
        )
        
        # Count pixels in each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_pixels = len(labels)
        
        # Filter clusters: keep only those with >= 5% of total lesion pixels
        threshold = 0.05 * total_pixels
        valid_clusters = []
        valid_centers = []
        
        for label, count in zip(unique_labels, counts):
            if count >= threshold:
                valid_clusters.append(label)
                valid_centers.append(centers[label])
        
        n_colors = len(valid_clusters)
        
        # Calculate color standard deviation from mean skin color
        # Mean skin color in LAB: approximately [120, 140, 150] (varies by ethnicity)
        mean_skin_lab = np.array([120.0, 140.0, 150.0])
        
        if len(valid_centers) > 0:
            valid_centers_array = np.array(valid_centers)
            color_deviations = np.linalg.norm(valid_centers_array - mean_skin_lab, axis=1)
            color_std = float(np.std(color_deviations))
        else:
            color_std = 0.0
        
        return {
            'n_colors': n_colors,
            'color_std': color_std,
            'dominant_colors': [center.tolist() for center in valid_centers]
        }
    
    def calculate_differential_structures(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, float]:
        """
        Calculate differential structures (texture) using GLCM and LBP
        
        Process:
        1. Compute GLCM (Gray Level Co-occurrence Matrix)
        2. Extract Haralick features: Contrast, Homogeneity, Energy, Correlation
        3. Compute LBP (Local Binary Patterns) for additional texture info
        
        Args:
            image: Input image (H, W, C) numpy array or PIL Image
            mask: Optional binary mask to focus on lesion region (H, W)
            
        Returns:
            Dictionary with:
                - contrast: GLCM contrast (high = network-like structures)
                - homogeneity: GLCM homogeneity (low = disorganized growth)
                - energy: GLCM energy
                - correlation: GLCM correlation
                - lbp_variance: Variance of LBP (roughness indicator)
        """
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        center_h, center_w = h // 4, w // 4
        masked_gray = gray[center_h:3*center_h, center_w:3*center_w]
        
        # Compute GLCM (Gray Level Co-occurrence Matrix)
        # Distances and angles for GLCM computation
        distances = [1, 2]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        # Quantize gray levels to reduce computation (256 -> 16 levels)
        gray_quantized = (gray // 16).astype(np.uint8)
        
        try:
            glcm = graycomatrix(
                gray_quantized,
                distances=distances,
                angles=angles,
                levels=16,
                symmetric=True,
                normed=True
            )
            
            # Extract Haralick features from GLCM
            contrast = float(np.mean(graycoprops(glcm, 'contrast')))
            homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
            energy = float(np.mean(graycoprops(glcm, 'energy')))
            correlation = float(np.mean(graycoprops(glcm, 'correlation')))
            
        except Exception as e:
            # Fallback if GLCM computation fails
            contrast = 0.0
            homogeneity = 0.0
            energy = 0.0
            correlation = 0.0
        
        # Compute Local Binary Patterns (LBP) for texture analysis
        # LBP parameters: radius=1, n_points=8
        radius = 1
        n_points = 8 * radius
        
        try:
            lbp = local_binary_pattern(masked_gray, n_points, radius, method='uniform')
            
            # Calculate LBP variance (indicates roughness/pattern complexity)
            if mask is not None:
                lbp_values = lbp[mask > 0]
            else:
                lbp_values = lbp.flatten()
            
            if len(lbp_values) > 0:
                lbp_variance = float(np.var(lbp_values))
            else:
                lbp_variance = 0.0
                
        except Exception as e:
            lbp_variance = 0.0
        
        return {
            'contrast': contrast,
            'homogeneity': homogeneity,
            'energy': energy,
            'correlation': correlation,
            'lbp_variance': lbp_variance
        }