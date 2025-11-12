"""
Quick test to verify transfer learning pipeline setup
Run this to ensure everything is configured correctly
"""
import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers not installed. Run: pip install transformers>=4.30.0")
        return False
    
    try:
        from seg_models.transfer_learning_models import get_available_transfer_models
        models = get_available_transfer_models()
        print(f"✓ Transfer learning models available: {len(models)}")
        for model in models:
            print(f"  - {model}")
    except Exception as e:
        print(f"✗ Failed to import transfer learning models: {e}")
        return False
    
    return True


def test_model_creation():
    """Test that models can be created"""
    print("\nTesting model creation...")
    
    try:
        from seg_models.transfer_learning_models import get_transfer_model
        
        # Test creating a small model
        print("Creating Segformer model...")
        model = get_transfer_model('segformer', freeze_encoder=True)
        print(f"✓ Segformer model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        print("\nNote: First run will download pretrained weights from HuggingFace.")
        print("This requires internet connection and may take a few minutes.")
        return False


def test_pipeline_structure():
    """Test that pipeline files exist"""
    print("\nTesting pipeline structure...")
    
    files_to_check = [
        'seg_transfer_learning_pipeline.py',
        'seg_models/transfer_learning_models.py',
        'example_transfer_learning.py',
        'scripts/setup_transfer_learning.sh',
        'outputs/transfer_learning_guide.txt',
    ]
    
    all_exist = True
    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} not found")
            all_exist = False
    
    return all_exist


def main():
    print("="*60)
    print("Transfer Learning Pipeline - Setup Verification")
    print("="*60)
    
    # Run tests
    imports_ok = test_imports()
    structure_ok = test_pipeline_structure()
    
    if imports_ok:
        models_ok = test_model_creation()
    else:
        models_ok = False
        print("\n✗ Skipping model creation test due to import failures")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if imports_ok and structure_ok and models_ok:
        print("✓ All tests passed!")
        print("\nYou can now use the transfer learning pipeline:")
        print("  python seg_transfer_learning_pipeline.py --model segformer --freeze_encoder")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        
        if not imports_ok:
            print("\nTo fix import issues:")
            print("  bash scripts/setup_transfer_learning.sh")
            print("  Or: pip install transformers>=4.30.0")
    
    print("="*60)


if __name__ == "__main__":
    main()
