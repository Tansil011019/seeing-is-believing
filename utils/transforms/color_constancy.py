# Shades of Gray
import torch

class ShadesOfGrayTransform:
    """
    Apply the Shades of Gray color constancy algorithm using the Minkowski norm.

    This method estimates the scene's illuminant and corrects color casts caused by non-neutral lighting conditions. It generalizes the traditional Gray-World (L1-norm) and Max-RGB (L∞-norm) algorithms by using a variable Minkowski norm (Lp).

    Steps:
        1. Raise each pixel intensity to the power `p`.
        2. Compute the mean of these powered values for each channel (R, G, B).
        3. Take the p-th root of these means to estimate the illuminant per channel.
        4. Scale each channel by the ratio between the mean illuminant intensity and its estimated value (brightness-preserving normalization).
        5. Clip the result to [0, 255] and return an 8-bit corrected image.

    Args:
        images (numpy.ndarray): Input image in RGB format.
        p (int or float, optional): Minkowski norm degree.
    
    Returns:
        numpy.ndarray: Color-corrected image

    Raises:
        ValueError: If the input image is None.

    References:
        Finlayson, G. D., & Trezzi, E. (2004). Shades of gray and colour constancy. In Proceedings of the IS&T/SID Twelfth Color Imaging Conference (pp. 37–41). Society for Imaging Science and Technology.
    """

    def __init__(self, p=6):
        self.p = p
    
    def __call__(self, image):
        if image is None: 
            raise ValueError("Image not loaded. Check the file path")

        img_powered = image.pow(self.p)
        mean_powered = img_powered.mean(dim=(1,2))
        illuminant_estimates = mean_powered.pow(1.0 / self.p)

        # Brightness preserving
        mean_illumninant = illuminant_estimates.mean()
        corrected_images = (image / illuminant_estimates[:, None, None]) * mean_illumninant

        # Clipping
        corrected_images = torch.clamp(corrected_images, 0, 1)
        return corrected_images

class MaxRGBTransform:
    def __call__(self, img):
        max_per_channel = img.amax(dim=(1, 2))
        mean_illuminant = max_per_channel.mean()
        corrected = (img / max_per_channel[:, None, None]) * mean_illuminant
        corrected = torch.clamp(corrected, 0, 1)
        return corrected