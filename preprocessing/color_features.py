"""
Advanced color preprocessing features for skin lesion images
"""
import cv2
import numpy as np
from typing import Optional


def add_shade_of_gray(image: np.ndarray, p: int = 6) -> np.ndarray:
    """
    Apply Shade of Gray color constancy algorithm
    
    Shade of Gray is a variant of Gray World that uses Minkowski p-norm
    for illuminant estimation. It's particularly useful for dermatology images
    to normalize lighting conditions.
    
    Args:
        image: Input RGB image (H, W, 3)
        p: Minkowski norm parameter (default=6). Higher values approximate max-RGB
    
    Returns:
        Color-corrected RGB image
    """
    if image.dtype != np.float32:
        image_float = image.astype(np.float32) / 255.0
    else:
        image_float = image.copy()
    
    # Compute Minkowski p-norm for each channel
    epsilon = 1e-10
    power = 1.0 / p
    
    # Calculate illuminant estimate using p-norm
    illuminant = np.power(
        np.mean(np.power(image_float + epsilon, p), axis=(0, 1)),
        power
    )
    
    # Avoid division by zero
    illuminant = np.maximum(illuminant, epsilon)
    
    # Normalize by the illuminant
    corrected = image_float / illuminant
    
    # Scale to match original illuminant magnitude
    scale = np.sqrt(np.sum(illuminant ** 2))
    corrected = corrected * scale
    
    # Clip and convert back to uint8
    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    
    return corrected


def add_clahe_enhancement(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    CLAHE enhances local contrast and is useful for bringing out details
    in skin lesions while avoiding over-amplification of noise.
    
    Args:
        image: Input RGB image (H, W, 3)
        clip_limit: Threshold for contrast limiting (default=2.0)
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Contrast-enhanced RGB image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    
    # Merge back and convert to RGB
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced


def add_white_balance(image: np.ndarray, method: str = 'gray_world') -> np.ndarray:
    """
    Apply white balance correction
    
    White balancing corrects color casts caused by different lighting conditions,
    which is crucial for consistent skin lesion analysis.
    
    Args:
        image: Input RGB image (H, W, 3)
        method: 'gray_world' or 'retinex' (default='gray_world')
    
    Returns:
        White-balanced RGB image
    """
    if method == 'gray_world':
        # Gray World assumption: average color should be gray
        image_float = image.astype(np.float32)
        
        # Calculate mean for each channel
        avg_r = np.mean(image_float[:, :, 0])
        avg_g = np.mean(image_float[:, :, 1])
        avg_b = np.mean(image_float[:, :, 2])
        
        # Calculate gray point (average of all channels)
        gray = (avg_r + avg_g + avg_b) / 3.0
        
        # Scale each channel
        if avg_r > 0 and avg_g > 0 and avg_b > 0:
            scale_r = gray / avg_r
            scale_g = gray / avg_g
            scale_b = gray / avg_b
            
            balanced = image_float.copy()
            balanced[:, :, 0] = np.clip(balanced[:, :, 0] * scale_r, 0, 255)
            balanced[:, :, 1] = np.clip(balanced[:, :, 1] * scale_g, 0, 255)
            balanced[:, :, 2] = np.clip(balanced[:, :, 2] * scale_b, 0, 255)
            
            return balanced.astype(np.uint8)
    
    return image


def add_color_temperature_adjustment(
    image: np.ndarray,
    temperature: float = 0.0
) -> np.ndarray:
    """
    Adjust color temperature (warm/cool)
    
    Simulates different lighting temperatures, useful for augmentation
    and handling images captured under various lighting conditions.
    
    Args:
        image: Input RGB image (H, W, 3)
        temperature: Temperature shift in range [-1, 1]
                    Negative = cooler (more blue)
                    Positive = warmer (more yellow/orange)
    
    Returns:
        Temperature-adjusted RGB image
    """
    if temperature == 0.0:
        return image
    
    image_float = image.astype(np.float32)
    
    # Temperature adjustment using color matrix
    if temperature > 0:  # Warmer
        # Increase red, decrease blue
        image_float[:, :, 0] = np.clip(image_float[:, :, 0] * (1 + temperature * 0.3), 0, 255)
        image_float[:, :, 2] = np.clip(image_float[:, :, 2] * (1 - temperature * 0.3), 0, 255)
    else:  # Cooler
        # Decrease red, increase blue
        abs_temp = abs(temperature)
        image_float[:, :, 0] = np.clip(image_float[:, :, 0] * (1 - abs_temp * 0.3), 0, 255)
        image_float[:, :, 2] = np.clip(image_float[:, :, 2] * (1 + abs_temp * 0.3), 0, 255)
    
    return image_float.astype(np.uint8)


def add_hair_removal_filter(
    image: np.ndarray,
    kernel_size: int = 17,
    threshold: int = 10
) -> np.ndarray:
    """
    Apply morphological hair removal filter
    
    Removes dark hair artifacts common in dermoscopy images using
    morphological operations and inpainting.
    
    Args:
        image: Input RGB image (H, W, 3)
        kernel_size: Size of morphological kernel (default=17)
        threshold: Threshold for hair detection (default=10)
    
    Returns:
        Hair-removed RGB image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply blackhat morphological operation to detect dark hair
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold to create hair mask
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    
    # Inpaint the detected hair regions
    result = cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return result


def add_color_jitter(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0
) -> np.ndarray:
    """
    Apply color jittering for augmentation
    
    Randomly adjusts brightness, contrast, and saturation to make models
    robust to color variations.
    
    Args:
        image: Input RGB image (H, W, 3)
        brightness: Brightness adjustment in range [-0.5, 0.5] (default=0.0)
        contrast: Contrast multiplier in range [0.5, 1.5] (default=1.0)
        saturation: Saturation multiplier in range [0.5, 1.5] (default=1.0)
    
    Returns:
        Color-jittered RGB image
    """
    # Convert to HSV for saturation adjustment
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Adjust saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    
    # Convert back to RGB
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    
    # Adjust brightness and contrast
    result = result * contrast + brightness * 255
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result
