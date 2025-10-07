#!/usr/bin/env python3
"""
Example usage of the skin disease classification utilities.
This script demonstrates how to use the various models and evaluation functions.
"""
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from utils import (
        calculate_heatmap_iou,
        display_heatmap_intersection,
        display_heatmap_union,
        train_and_evaluate_model,
        instantiate_model_hi_mvit,
        instantiate_model_medkan,
        instantiate_model_vgg16,
        get_best_resnet
    )
    print("✓ Successfully imported all utilities")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)


def demo_heatmap_evaluation():
    """Demonstrate heatmap IoU evaluation."""
    print("\n=== Heatmap IoU Evaluation Demo ===")
    
    # Create sample heatmaps
    np.random.seed(42)
    heatmap1 = np.random.rand(50, 50)
    heatmap2 = np.random.rand(50, 50)
    
    # Add some correlation
    heatmap2 = 0.6 * heatmap1 + 0.4 * heatmap2
    
    # Calculate IoU
    iou = calculate_heatmap_iou(heatmap1, heatmap2, threshold=0.5)
    print(f"IoU between heatmaps: {iou:.3f}")
    
    # Display visualizations
    print("Displaying heatmap intersection...")
    display_heatmap_intersection(heatmap1, heatmap2, threshold=0.5)
    
    print("Displaying heatmap union...")
    display_heatmap_union(heatmap1, heatmap2, threshold=0.5)


def demo_model_instantiation():
    """Demonstrate model instantiation."""
    print("\n=== Model Instantiation Demo ===")
    
    input_shape = (224, 224, 3)
    num_classes = 7  # Typical for skin disease classification
    
    models_to_test = [
        ("HI_MViT", lambda: instantiate_model_hi_mvit(input_shape, num_classes)),
        ("MedKAN", lambda: instantiate_model_medkan(input_shape, num_classes)),
        ("VGG16", lambda: instantiate_model_vgg16(input_shape, num_classes, weights=None)),
        ("ResNet152", lambda: get_best_resnet(input_shape=input_shape, num_classes=num_classes, weights=None))
    ]
    
    for model_name, model_func in models_to_test:
        try:
            print(f"\nInstantiating {model_name}...")
            model = model_func()
            total_params = model.count_params()
            trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
            
            print(f"✓ {model_name} created successfully")
            print(f"  - Total parameters: {total_params:,}")
            print(f"  - Trainable parameters: {trainable_params:,}")
            print(f"  - Model input shape: {model.input_shape}")
            print(f"  - Model output shape: {model.output_shape}")
            
            # Clean up memory
            del model
            
        except Exception as e:
            print(f"✗ Failed to create {model_name}: {e}")


def demo_synthetic_training():
    """Demonstrate training with synthetic data."""
    print("\n=== Synthetic Training Demo ===")
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    num_samples = 100
    image_size = (64, 64)  # Smaller size for demo
    num_classes = 3
    
    # Generate synthetic images
    np.random.seed(42)
    images = []
    labels = []
    
    for i in range(num_samples):
        # Create synthetic RGB image
        img_array = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)
        
        # Random label
        labels.append(np.random.randint(0, num_classes))
    
    labels = np.array(labels)
    
    print(f"Created {len(images)} synthetic images with {num_classes} classes")
    print(f"Class distribution: {np.bincount(labels)}")
    
    # Test with a simple model (VGG16 with no pretrained weights for speed)
    print("\nTraining small VGG16 model...")
    try:
        model = instantiate_model_vgg16(
            input_shape=(*image_size, 3),
            num_classes=num_classes,
            weights=None  # No pretrained weights for speed
        )
        
        # Train model
        trained_model, cm, metrics = train_and_evaluate_model(
            X=images,
            y=labels,
            model=model,
            test_ratio=0.3,
            epochs=2,  # Just 2 epochs for demo
            batch_size=16,
            target_size=image_size,
            verbose=1
        )
        
        print("\n✓ Training completed successfully!")
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        print(f"\nConfusion Matrix:\n{cm}")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")


def main():
    """Run all demonstrations."""
    print("Skin Disease Classification Utils Demo")
    print("=====================================")
    
    # Run demos
    try:
        demo_heatmap_evaluation()
    except Exception as e:
        print(f"Heatmap demo failed: {e}")
    
    try:
        demo_model_instantiation()
    except Exception as e:
        print(f"Model instantiation demo failed: {e}")
    
    try:
        demo_synthetic_training()
    except Exception as e:
        print(f"Training demo failed: {e}")
    
    print("\n=== Demo Complete ===")
    print("Next steps:")
    print("1. Run cmd/download_dataset.sh to download real data")
    print("2. Use the utilities in your own training scripts")
    print("3. Experiment with different model architectures")


if __name__ == "__main__":
    main()