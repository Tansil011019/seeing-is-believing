import torch
import numpy as np
import cv2
from typing import Dict, Union, Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms

from .segmentation_inference import SegmentationInference
from .classification_inference import ClassificationInference
from .direct_calculation import DirectCalculation
from .text_generation_inference import TextGenerationInference



class InferenceSystem:
    """
    Comprehensive inference system that orchestrates all inference pipelines
    Generates complete medical reports with visualizations
    """
    
    def __init__(
        self,
        # Segmentation parameters
        seg_model_name: str,
        seg_checkpoint_path: str,
        # Classification parameters
        cls_model_name: str,
        cls_checkpoint_path: str,
        # Text generation parameters
        text_model_name: str = "google/medgemma-4b-it",
        text_system_prompt: Optional[str] = None,
        # General parameters
        device: str = "cuda",
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the complete inference system
        
        Args:
            seg_model_name: Segmentation model name
            seg_checkpoint_path: Path to segmentation checkpoint
            cls_model_name: Classification model name
            cls_checkpoint_path: Path to classification checkpoint
            cls_num_classes: Number of classification classes
            text_model_name: Text generation model name
            text_system_prompt: Custom system prompt for text generation
            device: Device to run inference on
            input_size: Input size for models (H, W)
        """
        self.device = device
        self.input_size = input_size
        
        # Initialize all inference components
        print("Loading segmentation model...")
        self.segmentation_inference = SegmentationInference(
            model_name=seg_model_name,
            checkpoint_path=seg_checkpoint_path,
            device=device
        )
        
        print("Loading classification model...")
        self.classification_inference = ClassificationInference(
            model_name=cls_model_name,
            checkpoint_path=cls_checkpoint_path,
            device=device
        )
        
        print("Initializing direct calculation...")
        self.direct_calculation = DirectCalculation()
        
        print("Loading text generation model...")
        self.text_generation_inference = TextGenerationInference(
            model_name=text_model_name,
            system_prompt=text_system_prompt,
            device=device
        )
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(input_size)
        ])
        self.transform_tensor = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])
        
        print("Inference system initialized successfully!")
    
    def infer(self, image_path: str) -> Dict:
        """
        Complete inference pipeline: from image to medical report
        
        Args:
            image_path: Path to image file, numpy array, or PIL Image
            
        Returns:
            Dictionary containing:
                - report: Generated medical report text
                - visualization: Side-by-side image with heatmap overlay
                - metrics: All calculated metrics
                - classification: Classification results
                - segmentation: Segmentation results
        """
        # Load and preprocess image
        original_image, processed_image_pil, processed_image_tensor = self._load_and_preprocess_image(image_path)
        
        # Step 1: Classification Inference
        print("\n[1/5] Running classification inference...")
        cls_results = self.classification_inference.infer(processed_image_pil, image_path=image_path)
        
        
        # Step 2: Segmentation Inference
        print("\n[2/5] Running segmentation inference...")
        seg_results = self.segmentation_inference.infer(processed_image_tensor)
        
        
        # Step 3: Direct Calculation on Original Image
        print("\n[3/5] Computing color and texture features...")
        
        color_results = self.direct_calculation.calculate_color_variegation(np.array(original_image))
        differential_results = self.direct_calculation.calculate_differential_structures(np.array(original_image))
        # Step 4: Create Visualization (Heatmap + Original Image)
        print("\n[4/5] Creating visualization...")
        visualization = self._create_visualization(
            original_image, 
            cls_results['gradient_heatmap']
        )
        
        # Step 5: Generate Text Report
        print("\n[5/5] Generating medical report...")
        description_text = self._create_description_text(
            cls_results, seg_results, color_results, differential_results
        )
        
        # Convert visualization to PIL Image for text generation
        viz_image = Image.fromarray(visualization)
        
        report = self.text_generation_inference.infer(
            input_prompt=description_text,
            input_image=viz_image
        )
        
        
        print(f"TEST Detected: {cls_results['prediction_text']}")
        print(f"TEST Asymmetry Index: {seg_results['asymmetry_index']:.3f}")
        print(f"TEST Border Irregularity: {seg_results['border_irregularity']:.3f}")
        print(f"TEST Color Variegation - N Colors: {color_results['n_colors']}, Color Std: {color_results['color_std']:.2f}")
        print(f"TEST Differential Structures - Homogeneity: {differential_results['homogeneity']:.3f}, Contrast: {differential_results['contrast']:.2f}")
        
        # Compile all results
        return {
            'report': report,
            'visualization': visualization,
            'description': description_text,
            'metrics': {
                'asymmetry_index': seg_results['asymmetry_index'],
                'border_irregularity': seg_results['border_irregularity'],
                'color': color_results,
                'differential': differential_results
            },
            'prediction_text': cls_results['prediction_text'],
            'heatmap': cls_results['gradient_heatmap'],
            'original_image': original_image,
            'mask': seg_results.get('mask', None)
        }
    
    def _load_and_preprocess_image(self, 
                                   image_input: str) -> Tuple[Image.Image, Image.Image, torch.Tensor]:
        """
        Load and preprocess image for inference
        
        Args:
            image_input: Path to image, numpy array, or PIL Image
            
        Returns:
            Tuple of (original_image, processed_tensor)
        """
        # Load image
        if isinstance(image_input, str):
            original_image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            original_image = Image.fromarray(image_input.astype('uint8')).convert('RGB')
        elif isinstance(image_input, Image.Image):
            original_image = image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Preprocess for model
        processed_image_pil = self.transform(original_image)
        processed_image_tensor = self.transform_tensor(original_image)
        
        return original_image, processed_image_pil, processed_image_tensor
    
    def _create_visualization(self, 
                              original_image: Image.Image, 
                              heatmap: np.ndarray) -> np.ndarray:
        """
        Create side-by-side visualization: Original + Heatmap Overlay
        
        Args:
            original_image: Original RGB image (PIL Image)
            heatmap: Gradient heatmap (H, W)
            
        Returns:
            Side-by-side visualization image (H, W*2, 3)
        """
        # Convert PIL Image to numpy array
        original_array = np.array(original_image)
        h, w = original_array.shape[:2]
        
        # Ensure original image is RGB
        if len(original_array.shape) == 2:
            original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
        elif original_array.shape[2] == 4:
            original_array = cv2.cvtColor(original_array, cv2.COLOR_RGBA2RGB)
        
        # Resize heatmap to match original image size
        if heatmap.shape != (h, w):
            heatmap = cv2.resize(heatmap, (w, h))
        
        # Normalize heatmap to 0-255
        heatmap_normalized = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7) * 255).astype(np.uint8)
        
        # Apply colormap (jet colormap for heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image with transparency
        alpha = 0.5
        overlay = cv2.addWeighted(original_array, 1-alpha, heatmap_colored, alpha, 0)
        
        # Create side-by-side visualization
        visualization = np.hstack([original_array, overlay])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(visualization, 'Original', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(visualization, 'Attention Map', (w + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        return visualization
    
    def _create_description_text(self, cls_results: Dict, seg_results: Dict, 
                                color_results: Dict, texture_results: Dict) -> str:
        """
        Create structured description text from all inference results
        
        Args:
            cls_results: Classification inference results
            seg_results: Segmentation inference results
            color_results: Color variegation results
            texture_results: Texture analysis results
            
        Returns:
            Formatted description text
        """
        
        # ABCD rule assessment, add "Assessment : XXX " for each section if enabled
        # asymmetry_assessment = self._assess_asymmetry(seg_results['asymmetry_index'])
        # border_assessment = self._assess_border(seg_results['border_irregularity'])
        # color_assessment = self._assess_color(color_results['n_colors'])
        # texture_assessment = self._assess_texture(texture_results['homogeneity'], texture_results['contrast'])
        
        
        
        description = f"""
LESION ANALYSIS SUMMARY
=======================

DEEP LEARNING CLASSIFICATION:
{cls_results['prediction_text']}

ABCD RULE ANALYSIS:

A - ASYMMETRY:
  Index: {seg_results['asymmetry_index']:.3f}
  (Reference: 0.0 = perfectly symmetric, >0.35 = high asymmetry)

B - BORDER IRREGULARITY:
  Compactness: {seg_results['border_irregularity']:.3f}
  (Reference: 1.0 = smooth circle, >4.0 = irregular border)

C - COLOR VARIEGATION:
  Distinct Colors: {color_results['n_colors']}
  Color Deviation: {color_results['color_std']:.2f}
  (Reference: 1-2 colors = uniform, ≥4 colors = multicolor)

D - DIFFERENTIAL STRUCTURES (TEXTURE):
  Homogeneity: {texture_results['homogeneity']:.3f}
  Contrast: {texture_results['contrast']:.2f}
  Energy: {texture_results['energy']:.3f}
  Correlation: {texture_results['correlation']:.3f}
  (Low homogeneity + High contrast = disorganized growth pattern)
"""
        return description.strip()
    
    # def _assess_asymmetry(self, asymmetry_index: float) -> str:
    #     """Assess asymmetry level"""
    #     if asymmetry_index < 0.15:
    #         return "Low asymmetry (symmetric lesion)"
    #     elif asymmetry_index < 0.35:
    #         return "Moderate asymmetry"
    #     else:
    #         return "High asymmetry (concerning feature)"
    
    # def _assess_border(self, compactness: float) -> str:
    #     """Assess border irregularity"""
    #     if compactness < 2.0:
    #         return "Smooth, well-defined borders"
    #     elif compactness < 4.0:
    #         return "Moderately irregular borders"
    #     else:
    #         return "Highly irregular, jagged borders (concerning)"
    
    # def _assess_color(self, n_colors: int) -> str:
    #     """Assess color variegation"""
    #     if n_colors <= 1:
    #         return "Uniform color (single shade)"
    #     elif n_colors <= 2:
    #         return "Minimal color variation (2 shades)"
    #     elif n_colors == 3:
    #         return "Moderate color variation (3 shades)"
    #     else:
    #         return f"High color variation ({n_colors} distinct colors - concerning)"
    
    # def _assess_texture(self, homogeneity: float, contrast: float) -> str:
    #     """Assess texture characteristics"""
    #     if homogeneity > 0.5 and contrast < 10:
    #         return "Uniform texture pattern"
    #     elif homogeneity > 0.3 and contrast < 20:
    #         return "Moderate texture variation"
    #     else:
    #         return "Heterogeneous texture with high contrast (atypical network pattern)"
    
    # def _calculate_risk_summary(self, seg_results: Dict, color_results: Dict, texture_results: Dict) -> str:
    #     """Calculate overall risk summary"""
    #     risk_factors = []
        
    #     if seg_results['asymmetry_index'] > 0.35:
    #         risk_factors.append("• High asymmetry detected")
        
    #     if seg_results['border_irregularity'] > 4.0:
    #         risk_factors.append("• Irregular borders present")
        
    #     if color_results['n_colors'] >= 4:
    #         risk_factors.append("• Multiple color variations present")
        
    #     if texture_results['homogeneity'] < 0.3 and texture_results['contrast'] > 20:
    #         risk_factors.append("• Atypical texture pattern observed")
        
    #     if len(risk_factors) == 0:
    #         return "No significant risk indicators detected in automated analysis."
    #     elif len(risk_factors) <= 2:
    #         return "Moderate risk indicators present:\n" + "\n".join(risk_factors)
    #     else:
    #         return "Multiple risk indicators detected (requires clinical evaluation):\n" + "\n".join(risk_factors)


def create_inference_system(
    seg_model_name: str = "segformer_b0",
    seg_checkpoint_path: str = "./checkpoints/segmentation/best_model.pth",
    cls_model_name: str = "resnet50",
    cls_checkpoint_path: str = "./checkpoints/classification/best_model.pth",
    **kwargs
) -> InferenceSystem:
    """
    Convenience function to create inference system with default parameters
    
    Args:
        seg_model_name: Segmentation model name
        seg_checkpoint_path: Path to segmentation checkpoint
        cls_model_name: Classification model name
        cls_checkpoint_path: Path to classification checkpoint
        **kwargs: Additional parameters for InferenceSystem
        
    Returns:
        Initialized InferenceSystem instance
    """
    return InferenceSystem(
        seg_model_name=seg_model_name,
        seg_checkpoint_path=seg_checkpoint_path,
        cls_model_name=cls_model_name,
        cls_checkpoint_path=cls_checkpoint_path,
        **kwargs
    )
