"""
Inference package for comprehensive skin lesion analysis
Includes segmentation, classification, direct calculation, and report generation
"""

from .segmentation_inference import SegmentationInference
from .classification_inference import ClassificationInference
from .direct_calculation import DirectCalculation
from .text_generation_inference import TextGenerationInference
from .inference_system import InferenceSystem, create_inference_system

__all__ = [
    'SegmentationInference',
    'ClassificationInference',
    'DirectCalculation',
    'TextGenerationInference',
    'InferenceSystem',
    'create_inference_system'
]
