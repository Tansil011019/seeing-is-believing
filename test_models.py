#!/usr/bin/env python3
"""
Test script to verify that all models can be instantiated and run forward passes
"""
import torch
from seg_models import get_available_models, get_model


def test_model(model_name, input_shape=(2, 3, 256, 256)):
    """
    Test a model by creating it and running a forward pass
    
    Args:
        model_name: Name of the model to test
        input_shape: Shape of input tensor (B, C, H, W)
    
    Returns:
        True if test passes, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Create model
        print(f"Creating model '{model_name}'...")
        model = get_model(model_name, use_mask_cnn=False, pretrained=False)
        print(f"✓ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Set to eval mode
        model.eval()
        
        # Create dummy input
        print(f"Creating dummy input with shape {input_shape}...")
        dummy_input = torch.randn(input_shape)
        
        # Forward pass
        print(f"Running forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
        
        # Check output shape
        print(f"  Output shape: {output.shape}")
        
        # Verify output shape
        expected_shape = (input_shape[0], 1, input_shape[2], input_shape[3])
        if output.shape == expected_shape:
            print(f"✓ Output shape matches expected {expected_shape}")
        else:
            print(f"✗ Output shape mismatch! Expected {expected_shape}, got {output.shape}")
            return False
        
        # Check for NaN or Inf
        if torch.isnan(output).any():
            print(f"✗ Output contains NaN values!")
            return False
        if torch.isinf(output).any():
            print(f"✗ Output contains Inf values!")
            return False
        print(f"✓ Output contains no NaN or Inf values")
        
        print(f"\n✅ Model '{model_name}' test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Model '{model_name}' test FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all available models"""
    print("\n" + "="*60)
    print("TESTING ALL MODELS")
    print("="*60)
    
    available_models = get_available_models()
    print(f"\nAvailable models: {', '.join(available_models)}")
    
    results = {}
    for model_name in available_models:
        passed = test_model(model_name)
        results[model_name] = passed
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed_count = sum(results.values())
    total_count = len(results)
    
    for model_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{model_name:20s}: {status}")
    
    print(f"\n{passed_count}/{total_count} models passed")
    print("="*60)
    
    return passed_count == total_count


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
