import torch
from typing import Optional, Union
from PIL import Image
import numpy as np

from transformers import pipeline
import torch

AVAILABLE_MODELS = ['medgemma-4b-it', 'llama-3']


from dotenv import load_dotenv
import os

load_dotenv()  # loads variables from .env into the environment

class TextGenerationInference:
    """
    Text generation inference for creating medical reports
    Uses vision-language model to generate diagnostic reports
    """
    
    # Default system prompt for medical report generation
    DEFAULT_SYSTEM_PROMPT = """You are an expert dermatologist AI assistant specializing in melanoma detection and skin lesion analysis. Your role is to generate comprehensive, clinically accurate diagnostic reports based on the ABCD rule (Asymmetry, Border irregularity, Color variegation, Differential structures/Diameter) and deep learning analysis.

When generating reports:
1. Start with a brief overview of the lesion characteristics
2. Analyze each ABCD component with specific measurements
3. Interpret the neural network's classification results
4. Provide a risk assessment based on the combined metrics
5. Recommend next steps (e.g., biopsy, monitoring, no action needed)
6. Use clear, professional medical terminology
7. Be objective and evidence-based
8. Highlight any concerning features that warrant further investigation

Format the report in a structured manner with clear sections for clinical use."""
    
    def __init__(self, 
                 model_name: str = "google/medgemma-4b-it", 
                 system_prompt: Optional[str] = None, 
                 device: str = "cuda"):
        """
        Initialize text generation model
        
        Args:
            model_name: Name of the text generation model
            system_prompt: Custom system prompt for report generation
            device: Device to run inference on
        """
        self.device = device
        self.system_prompt = system_prompt if system_prompt else self.DEFAULT_SYSTEM_PROMPT
        self.model_name = model_name.lower()
        
        # Initialize model based on model name
        self._load_model()
    
    def _load_model(self):
        """Load the appropriate text generation model"""
        try:
            self.pipe = pipeline(
                "image-text-to-text",
                model=self.model_name,
                torch_dtype=torch.bfloat16,
                model_kwargs={"load_in_8bit": True},
            )
            
            self.model = self.pipe.model
            self.tokenizer = self.pipe.tokenizer
            # The processor might be in different attributes depending on the pipeline
            self.processor = getattr(self.pipe, 'image_processor', 
                              getattr(self.pipe, 'feature_extractor',
                              getattr(self.pipe, 'processor', self.pipe.tokenizer)))
            self.model_type = 'medgemma'

        except Exception as e:
            print(f"Warning: Could not load {self.model_name} Error: {e}")
            raise ValueError(f"Failed to load model {self.model_name}.")
    
    def infer(self, 
              input_prompt: str, 
              input_image: Optional[Union[Image.Image, np.ndarray]] = None, 
              max_length: int = 1024, 
              temperature: float = 0.7) -> str:
        """
        Generate medical report text from input prompt and optional image
        
        Args:
            input_prompt: Text prompt containing analysis results and metrics
            input_image: Optional input image for vision-language models
            max_length: Maximum length of generated text
            temperature: Sampling temperature for generation (higher = more creative)
            
        Returns:
            Generated medical report text
        """
        
        report = self._generate_vision_language(
                prompt=input_prompt,
                image=input_image,
                max_length=max_length,
                temperature=temperature
            )
        
        return report
    
    def _generate_vision_language(self, 
                                  prompt: str, 
                                  image: Optional[Image.Image], 
                                  max_length: int, 
                                  temperature: float) -> str:
        """Generate text using vision-language model (Medgemma, etc.)"""
        self.model.eval()
        
        
        
        # For MedGemma and similar models, we need to include <image> token
        if image is not None:
            messages = [
                {"role": "system", "content": 
                    [{"type": "text", "text" : ""}]}, # REINSERT THE SYSTEM PROMPT
                {"role": "user", 
                 "content": 
                    [{"type": "text", "text" : f"<image>\n\n\n{prompt}\n\nGenerate a diagnostic report:"},
                     {"type": "image", "image": image}]}
           ]
        else:
            messages = [
                {"role": "system", "content": 
                    [{"type": "text", "text" : self.system_prompt}]},
                {"role": "user", "content": 
                    [{"type": "text", "text" : f"Analysis Data:\n{prompt}\n\nGenerate a diagnostic report:"}]}
            ]
        
        outputs = self.pipe(
            messages,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
        )
        
        # Extract generated text from pipeline output
        generated_text = outputs[0]['generated_text'][-1]['content']
        
        return generated_text.strip()