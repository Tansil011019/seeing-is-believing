#!/usr/bin/env python3
"""
Model Visualization Demo

Demonstrates the model visualization utilities on available segmentation models.

Usage:
    python test/demo_model_visualization.py
    python test/demo_model_visualization.py --models combined deeplabv3p
    python test/demo_model_visualization.py --save-dir outputs/model_visualizations
"""

import sys
import argparse
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from seg_models import get_model
from utils.model_visualization import (
    print_model_summary,
    visualize_model_architecture,
    compare_models
)


def demo_single_model(model_name: str, save_dir: Path):
    """
    Demonstrate visualization for a single model
    
    Args:
        model_name: Name of the model to visualize
        save_dir: Directory to save visualizations
    """
    print(f"\n{'='*80}")
    print(f"Visualizing: {model_name.upper()}")
    print(f"{'='*80}")
    
    try:
        # Get model
        if model_name == 'combined':
            model = get_model('combined', use_mask_cnn=True, pretrained=False)
        elif model_name == 'segmentation':
            model = get_model('segmentation', use_mask_cnn=False, pretrained=False)
        elif model_name in ['deeplabv3p', 'deeplab', 'deeplabv3+']:
            from seg_models.deeplab_v3p_torch import DeepLabV3Plus
            model = DeepLabV3Plus(num_classes=1, backbone='resnet50')
        elif model_name in ['fatnet', 'fat_net', 'fat-net']:
            from seg_models.fat_net import FAT_Net
            model = FAT_Net(num_classes=1)
        else:
            print(f"Unknown model: {model_name}")
            return None
        
        # Print text summary
        print_model_summary(model, input_size=(1, 3, 256, 256), model_name=model_name)
        
        # Create visualization
        if save_dir:
            save_path = save_dir / f"{model_name}_architecture.png"
            visualize_model_architecture(
                model,
                input_size=(1, 3, 256, 256),
                model_name=model_name.upper(),
                save_path=str(save_path),
                show_parameters=True
            )
        else:
            visualize_model_architecture(
                model,
                input_size=(1, 3, 256, 256),
                model_name=model_name.upper(),
                show_parameters=True
            )
        
        return model
        
    except Exception as e:
        print(f"Error visualizing {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def demo_model_comparison(model_names: list, save_dir: Path):
    """
    Compare multiple models
    
    Args:
        model_names: List of model names to compare
        save_dir: Directory to save comparison
    """
    print(f"\n{'='*80}")
    print(f"COMPARING MODELS: {', '.join(model_names)}")
    print(f"{'='*80}")
    
    models = {}
    
    for name in model_names:
        try:
            if name == 'combined':
                model = get_model('combined', use_mcnn=True, pretrained=False)
            elif name == 'segmentation':
                model = get_model('segmentation', use_mcnn=False, pretrained=False)
            elif name in ['deeplabv3p', 'deeplab', 'deeplabv3+']:
                from seg_models.deeplab_v3p_torch import DeepLabV3Plus
                model = DeepLabV3Plus(num_classes=1, backbone='resnet50')
                name = 'DeepLabV3+'
            elif name in ['fatnet', 'fat_net', 'fat-net']:
                from seg_models.fat_net import FAT_Net
                model = FAT_Net(num_classes=1)
                name = 'FAT-Net'
            else:
                print(f"Skipping unknown model: {name}")
                continue
            
            models[name] = model
            print(f"✓ Loaded {name}")
            
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
    
    if len(models) > 1:
        save_path = None
        if save_dir:
            save_path = str(save_dir / "model_comparison.png")
        
        compare_models(models, save_path=save_path)
    else:
        print("Not enough models loaded for comparison")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize segmentation model architectures',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['combined'],
        help='Models to visualize (options: combined, segmentation, deeplabv3p, fatnet)'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default=None,
        help='Directory to save visualizations (default: display only)'
    )
    
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Create comparison visualization of all models'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Visualize all available models'
    )
    
    args = parser.parse_args()
    
    # Setup save directory
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {save_dir}")
    
    # Determine which models to visualize
    if args.all:
        model_names = ['combined', 'segmentation', 'deeplabv3p', 'fatnet']
    else:
        model_names = args.models
    
    print("="*80)
    print("MODEL ARCHITECTURE VISUALIZATION DEMO")
    print("="*80)
    print(f"Models to visualize: {', '.join(model_names)}")
    
    # Visualize each model
    loaded_models = []
    for name in model_names:
        model = demo_single_model(name, save_dir)
        if model is not None:
            loaded_models.append(name)
    
    # Create comparison if requested
    if args.compare and len(loaded_models) > 1:
        demo_model_comparison(loaded_models, save_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    if save_dir:
        print(f"Files saved to: {save_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
