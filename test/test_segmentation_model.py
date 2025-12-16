from models.seg_models import get_model, get_available_models
import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing.segmentation_preprocessing.preprocessing import preprocess_segmentation_dataset_parallel
from evaluation.metrics import calculate_dice, calculate_iou
import logging

class SegmentationTester :
    
    def __init__(self,
                 model_name: str,
                 checkpoint_path: str,
                 input_folder: str = "datasets/ISIC2018_Task1-2_Test_Input",
                 ground_truth_folder: str = "datasets/ISIC2018_Task1_Test_GroundTruth",
                 prediction_output_folder: str = "datasets/pred_out/Task1",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 batch_size: int = 8,
                 num_workers: int = 4) :
        
        if not model_name in get_available_models() :
            raise Exception("Error: Model not Found")
        
        # Load model and checkpoint
        self.model = get_model(model_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(device)
        self.model.eval()
        
        self.input_folder =  input_folder
        self.ground_truth_folder = ground_truth_folder
        self.prediction_output_folder = prediction_output_folder
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        os.makedirs(prediction_output_folder, exist_ok=True)
        
    def test(self) :
        """
        Run the complete testing pipeline:
        1. Load images and masks
        2. Preprocess images
        3. Run inference
        4. Evaluate with Dice and IoU metrics
        """
        self.logger.info("Starting testing pipeline...")
        
        # Step 1: Preprocess the dataset
        preprocessed_image_folder = os.path.join(self.prediction_output_folder, "preprocessed_images")
        preprocessed_mask_folder = os.path.join(self.prediction_output_folder, "preprocessed_masks")
        
        self.logger.info("Step 1: Preprocessing images...")
        num_processed = preprocess_segmentation_dataset_parallel(
            image_folder=self.input_folder,
            mask_folder=self.ground_truth_folder,
            output_image_folder=preprocessed_image_folder,
            output_mask_folder=preprocessed_mask_folder,
            apply_augmentation=False,
            num_workers=self.num_workers,
            logger=self.logger,
            output_size=(512, 512)
        )
        self.logger.info(f"Preprocessed {num_processed} images")
        
        # Step 2: Load preprocessed images and masks
        self.logger.info("Step 2: Loading preprocessed data...")
        image_files = sorted([
            f for f in os.listdir(preprocessed_image_folder)
            if f.endswith('.png') or f.endswith('.jpg')
        ])
        
        # Step 3: Run inference
        self.logger.info("Step 3: Running inference...")
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for image_file in tqdm(image_files, desc="Processing images"):
                # Load image
                image_path = os.path.join(preprocessed_image_folder, image_file)
                image = self._load_image(image_path)
                
                # Load corresponding mask
                mask_path = os.path.join(preprocessed_mask_folder, image_file)
                mask = self._load_mask(mask_path)
                
                # Convert to tensors
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                # Run inference
                output = self.model(image_tensor)
                output = torch.sigmoid(output)
                
                # Convert to binary prediction
                pred_mask = (output > 0.5).float()
                
                # Save prediction
                pred_np = pred_mask.squeeze().cpu().numpy()
                output_path = os.path.join(self.prediction_output_folder, image_file)
                self._save_prediction(pred_np, output_path)
                
                # Store for evaluation
                predictions.append(pred_mask.cpu())
                ground_truths.append(torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() / 255.0)
        
        # Step 4: Evaluate metrics
        self.logger.info("Step 4: Evaluating metrics...")
        predictions_batch = torch.cat(predictions, dim=0)
        ground_truths_batch = torch.cat(ground_truths, dim=0)
        
        dice_score = calculate_dice(predictions_batch, ground_truths_batch)
        iou_score = calculate_iou(predictions_batch, ground_truths_batch)
        
        # Print results
        print(f"\n{'='*50}")
        print(f"Testing Results:")
        print(f"{'='*50}")
        print(f"Number of test images: {len(image_files)}")
        print(f"Dice Score: {dice_score:.4f}")
        print(f"IoU Score: {iou_score:.4f}")
        print(f"{'='*50}\n")
        
        self.logger.info(f"Testing complete. Dice: {dice_score:.4f}, IoU: {iou_score:.4f}")
        
        return {
            'dice': dice_score,
            'iou': iou_score,
            'num_images': len(image_files)
        }
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and convert image to RGB"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load and binarize mask"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask
    
    def _save_prediction(self, pred_mask: np.ndarray, output_path: str) -> None:
        """Save prediction mask as image"""
        pred_mask = (pred_mask * 255).astype(np.uint8)
        cv2.imwrite(output_path, pred_mask)
        
        
if __name__ == "__main__":
    tester = SegmentationTester(
        model_name='segformer_b0',
        checkpoint_path='checkpoints/transfer_segformer_b0/best.pth',
        input_folder='datasets/ISIC2018_Task1-2_Test_Input',
        ground_truth_folder='datasets/ISIC2018_Task1_Test_GroundTruth',
        prediction_output_folder='datasets/pred_out/Task1',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=8,
        num_workers=4
    )
    tester.test()
        