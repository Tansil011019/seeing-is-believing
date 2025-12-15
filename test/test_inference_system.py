import torch
import cv2
import numpy as np
from inference.inference_system import InferenceSystem
import matplotlib.pyplot as plt
import os
import json

# Configuration
SEG_MODEL_NAME = 'segformer_b0'
SEG_CHECKPOINT_PATH = 'checkpoints/transfer_segformer_b0/best.pth' 
CLS_MODEL_NAME = 'PLACEHOLDER_MODEL_NAME'
CLS_CHECKPOINT_PATH = 'checkpoints/PLACEHOLDER_CHECKPOINT.pth'
IMAGE_PATH = 'datasets/ISIC2018_Task1-2_Training_Input/ISIC_0000001.jpg'
DEVICE = 'cpu'
VIZ_OUTPUT_PATH = 'output/inference_result.jpg'
JSON_OUTPUT_PATH = 'output/test/inference_test_result.json'

def main():
    # Initialize complete inference system
    print(f"\n[INITIALIZATION]")
    print(f"  Segmentation Model: {SEG_MODEL_NAME}")
    print(f"  Segmentation Checkpoint: {SEG_CHECKPOINT_PATH}")
    print(f"  Classification Model: {CLS_MODEL_NAME}")
    print(f"  Classification Checkpoint: {CLS_CHECKPOINT_PATH}")
    print(f"  Device: {DEVICE}")
    print(f"  Input Image: {IMAGE_PATH}")
    print()
    
    inference_system = InferenceSystem(
        seg_model_name=SEG_MODEL_NAME,
        seg_checkpoint_path=SEG_CHECKPOINT_PATH,
        cls_model_name=CLS_MODEL_NAME,
        cls_checkpoint_path=CLS_CHECKPOINT_PATH,
        device=DEVICE,
        input_size=(224, 224)
    )
    
    # Run complete inference pipeline
    results = inference_system.infer(IMAGE_PATH)
    
    out = {}
    out["metrics"] = results['metrics']
    out["preduction_text"] = results['prediction_text']
    out["report"] = results['report']
    out["description"] = results['description']

    
    # Save as json
    json_output_path =  JSON_OUTPUT_PATH
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

    with open(json_output_path, 'w') as f:
        json.dump(out, f, indent=4)
        
    # Display results
    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULTS")
    print("=" * 80)
    print(f"Prediction: {results['prediction_text']}")
    # print(f"\nProbabilities:")
    # for attr, prob in cls_results['probabilities'].items():
    #     print(f"  {attr}: {prob:.4f}")
    
    print("\n" + "=" * 80)
    print("ABCD Metrics")
    print("=" * 80)
    print(f"Asymmetry Index:       {results['metrics']['asymmetry_index']:.6f}")
    print(f"Border Irregularity:   {results['metrics']['border_irregularity']:.6f}")
    color_results = results['metrics']['color']
    print(f"Number of Colors:      {color_results['n_colors']}")
    print(f"Color Std Deviation:   {color_results['color_std']:.6f}")
    print(f"Dominant Colors:       {len(color_results['dominant_colors'])} clusters")
    texture_results = results['metrics']['differential']
    print(f"Contrast:              {texture_results['contrast']:.6f}")
    print(f"Homogeneity:           {texture_results['homogeneity']:.6f}")
    print(f"Energy:                {texture_results['energy']:.6f}")
    print(f"Correlation:           {texture_results['correlation']:.6f}")
    print(f"LBP Variance:          {texture_results['lbp_variance']:.6f}")
    
    print("\n" + "=" * 80)
    print("GENERATED MEDICAL REPORT")
    print("=" * 80)
    print(results['report'])
    
    print("\n" + "=" * 80)
    print("DESCRIPTION TEXT")
    print("=" * 80)
    print(results['description'])
    
    visualization = results['visualization']
    visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(VIZ_OUTPUT_PATH, visualization_bgr)
    print(f"  Visualization saved to: {VIZ_OUTPUT_PATH}")
    
    
    
    # Display visualization using matplotlib
    try:
        plt.figure(figsize=(15, 7))
        plt.imshow(visualization)
        plt.axis('off')
        plt.title('Inference System Output: Original vs Attention Map', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig('output/inference_result_display.png', dpi=150, bbox_inches='tight')
        print(f"  Display image saved to: output/inference_result_display.png")
        plt.close()
    except Exception as e:
        print(f"  Could not create display image: {e}")
    
    return results

if __name__ == "__main__":
    results = main()
