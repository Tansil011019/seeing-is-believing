"""
Validation script for Attribute Transfer Learning Pipeline

Checks that all required files and components are properly set up.
"""

import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists"""
    path = Path(filepath)
    if path.exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False


def check_import(module_name, package_name=None):
    """Check if a Python module can be imported"""
    try:
        __import__(module_name)
        display_name = package_name or module_name
        print(f"✓ {display_name} installed")
        return True
    except ImportError:
        display_name = package_name or module_name
        print(f"✗ {display_name} NOT installed")
        return False


def main():
    print("="*60)
    print("Attribute Transfer Learning Pipeline - Setup Validation")
    print("="*60)
    
    all_checks = []
    
    # Check main pipeline file
    print("\n[1/7] Main Pipeline File")
    all_checks.append(check_file_exists(
        "attr_transfer_learning_pipeline.py",
        "Main pipeline"
    ))
    
    # Check preprocessing files
    print("\n[2/7] Preprocessing Files")
    all_checks.append(check_file_exists(
        "preprocessing/attr_preprocess.py",
        "Preprocessing script"
    ))
    all_checks.append(check_file_exists(
        "preprocessing/attr_dataset.py",
        "Dataset loader"
    ))
    
    # Check model files
    print("\n[3/7] Model Files")
    all_checks.append(check_file_exists(
        "models/attr_model_factory.py",
        "Model factory"
    ))
    all_checks.append(check_file_exists(
        "models/resnet18_attr.py",
        "ResNet-18 model"
    ))
    all_checks.append(check_file_exists(
        "models/resnet34_attr.py",
        "ResNet-34 model"
    ))
    all_checks.append(check_file_exists(
        "models/resnet50_attr.py",
        "ResNet-50 model"
    ))
    all_checks.append(check_file_exists(
        "models/efficientvim_attr.py",
        "EfficientViM model"
    ))
    all_checks.append(check_file_exists(
        "models/ecvit_attr.py",
        "ECViT model"
    ))
    
    # Check evaluation files
    print("\n[4/7] Evaluation Files")
    all_checks.append(check_file_exists(
        "evaluation/attr_metrics.py",
        "Metrics module"
    ))
    
    # Check documentation
    print("\n[5/7] Documentation Files")
    all_checks.append(check_file_exists(
        "ATTR_PIPELINE_README.md",
        "Main README"
    ))
    all_checks.append(check_file_exists(
        "QUICKSTART_ATTR.md",
        "Quick start guide"
    ))
    
    # Check dataset directories
    print("\n[6/7] Dataset Directories")
    dataset_checks = []
    dataset_checks.append(check_file_exists(
        "datasets/ISIC2018_Task1-2_Training_Input",
        "Training images"
    ))
    dataset_checks.append(check_file_exists(
        "datasets/ISIC2018_Task1-2_Validation_Input",
        "Validation images"
    ))
    dataset_checks.append(check_file_exists(
        "datasets/ISIC2018_Task2_Training_GroundTruth_v3",
        "Training ground truth"
    ))
    dataset_checks.append(check_file_exists(
        "datasets/ISIC2018_Task2_Validation_GroundTruth",
        "Validation ground truth"
    ))
    
    if not all(dataset_checks):
        print("\n  ⚠️  Dataset not found. Download ISIC 2018 Task 2 dataset.")
    
    # Check Python dependencies
    print("\n[7/7] Python Dependencies")
    dep_checks = []
    dep_checks.append(check_import("torch", "PyTorch"))
    dep_checks.append(check_import("torchvision", "TorchVision"))
    dep_checks.append(check_import("timm", "timm (PyTorch Image Models)"))
    dep_checks.append(check_import("PIL", "Pillow"))
    dep_checks.append(check_import("cv2", "OpenCV"))
    dep_checks.append(check_import("pandas", "Pandas"))
    dep_checks.append(check_import("numpy", "NumPy"))
    dep_checks.append(check_import("sklearn", "scikit-learn"))
    
    if not all(dep_checks):
        print("\n  ⚠️  Missing dependencies. Install with: pip install -r requirements.txt")
    
    # Summary
    print("\n" + "="*60)
    if all(all_checks):
        print("✅ ALL CHECKS PASSED - Pipeline is ready!")
        print("\nNext steps:")
        print("  1. Download dataset (if not already done)")
        print("  2. Run preprocessing: python attr_transfer_learning_pipeline.py --preprocess")
        print("  3. Train model: python attr_transfer_learning_pipeline.py --model resnet18")
    else:
        print("❌ SOME CHECKS FAILED - Please fix the issues above")
        print("\nCommon issues:")
        print("  • Missing files: Check that all files were created correctly")
        print("  • Missing dependencies: Run 'pip install -r requirements.txt'")
        print("  • Missing dataset: Download ISIC 2018 Task 2 from official website")
    print("="*60)
    
    return 0 if all(all_checks) else 1


if __name__ == "__main__":
    sys.exit(main())
