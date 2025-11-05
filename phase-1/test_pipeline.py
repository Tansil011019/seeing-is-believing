"""
Test script to verify all components are working correctly
"""
import sys
import torch
import numpy as np

print("=" * 80)
print("TESTING SEGMENTATION PIPELINE COMPONENTS")
print("=" * 80)

# Test 1: Import modules
print("\n1. Testing module imports...")
try:
    from seg_preprocess import (
        rotate_image_and_mask,
        dilate_image_and_mask,
        augment_image_and_mask,
        SegmentationDataset
    )
    from seg_models import (
        MaskCNN,
        AttentionBlock,
        MultiscaleLayer,
        SegmentationModel,
        CombinedModel,
        get_model
    )
    from seg_evaluation import (
        DiceLoss,
        CombinedLoss,
        IoULoss,
        BBoxLoss,
        calculate_iou,
        calculate_dice,
        calculate_pixel_accuracy,
        extract_bbox_from_mask
    )
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Test preprocessing functions
print("\n2. Testing preprocessing functions...")
try:
    # Create dummy image and mask
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    dummy_mask = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
    
    # Test rotation
    rotated_img, rotated_mask = rotate_image_and_mask(dummy_image, dummy_mask, 45)
    assert rotated_img.shape == dummy_image.shape, "Rotation changed image shape"
    assert rotated_mask.shape == dummy_mask.shape, "Rotation changed mask shape"
    
    # Test dilation
    dilated_img, dilated_mask = dilate_image_and_mask(dummy_image, dummy_mask, 1.5, 1.0)
    assert dilated_img.shape[1] == int(dummy_image.shape[1] * 1.5), "Dilation incorrect"
    
    # Test augmentation
    augmented = augment_image_and_mask(dummy_image, dummy_mask)
    assert len(augmented) > 1, "Augmentation didn't create multiple samples"
    
    print(f"✓ Preprocessing functions working (created {len(augmented)} augmented samples)")
except Exception as e:
    print(f"✗ Preprocessing test failed: {e}")
    sys.exit(1)

# Test 3: Test MaskCNN
print("\n3. Testing MaskCNN...")
try:
    mask_cnn = MaskCNN(backbone='resnet18', pretrained=False)
    dummy_input = torch.randn(2, 3, 256, 256)
    bbox = mask_cnn(dummy_input)
    
    assert bbox.shape == (2, 4), f"Expected bbox shape (2, 4), got {bbox.shape}"
    assert (bbox >= 0).all() and (bbox <= 1).all(), "BBox values not in [0, 1]"
    
    # Test cropping
    cropped = mask_cnn.crop_image(dummy_input, bbox)
    assert cropped.shape == dummy_input.shape, "Cropped image has wrong shape"
    
    print(f"✓ MaskCNN working (output shape: {bbox.shape})")
except Exception as e:
    print(f"✗ MaskCNN test failed: {e}")
    sys.exit(1)

# Test 4: Test AttentionBlock
print("\n4. Testing AttentionBlock...")
try:
    attention = AttentionBlock(64, 128)
    dummy_input = torch.randn(2, 64, 32, 32)
    output = attention(dummy_input)
    
    assert output.shape == (2, 128, 32, 32), f"Expected shape (2, 128, 32, 32), got {output.shape}"
    
    print(f"✓ AttentionBlock working (output shape: {output.shape})")
except Exception as e:
    print(f"✗ AttentionBlock test failed: {e}")
    sys.exit(1)

# Test 5: Test MultiscaleLayer
print("\n5. Testing MultiscaleLayer...")
try:
    multiscale = MultiscaleLayer(128, 256)
    dummy_input = torch.randn(2, 128, 16, 16)
    output = multiscale(dummy_input)
    
    assert output.shape == (2, 256, 16, 16), f"Expected shape (2, 256, 16, 16), got {output.shape}"
    
    print(f"✓ MultiscaleLayer working (output shape: {output.shape})")
except Exception as e:
    print(f"✗ MultiscaleLayer test failed: {e}")
    sys.exit(1)

# Test 6: Test SegmentationModel
print("\n6. Testing SegmentationModel...")
try:
    seg_model = SegmentationModel(in_channels=3, num_classes=1, base_channels=64)
    dummy_input = torch.randn(2, 3, 256, 256)
    output = seg_model(dummy_input)
    
    assert output.shape == (2, 1, 256, 256), f"Expected shape (2, 1, 256, 256), got {output.shape}"
    
    print(f"✓ SegmentationModel working (output shape: {output.shape})")
except Exception as e:
    print(f"✗ SegmentationModel test failed: {e}")
    sys.exit(1)

# Test 7: Test CombinedModel
print("\n7. Testing CombinedModel...")
try:
    combined = CombinedModel(use_mask_cnn=True, mask_cnn_pretrained=False)
    dummy_input = torch.randn(2, 3, 256, 256)
    
    # Test without bbox
    output = combined(dummy_input, return_bbox=False)
    assert output.shape == (2, 1, 256, 256), f"Expected shape (2, 1, 256, 256), got {output.shape}"
    
    # Test with bbox
    output, bbox = combined(dummy_input, return_bbox=True)
    assert output.shape == (2, 1, 256, 256), "Output shape incorrect"
    assert bbox.shape == (2, 4), "BBox shape incorrect"
    
    print(f"✓ CombinedModel working (output: {output.shape}, bbox: {bbox.shape})")
except Exception as e:
    print(f"✗ CombinedModel test failed: {e}")
    sys.exit(1)

# Test 8: Test loss functions
print("\n8. Testing loss functions...")
try:
    pred = torch.randn(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    # Test DiceLoss
    dice_loss = DiceLoss()
    loss_val = dice_loss(pred, target)
    assert loss_val.item() >= 0, "Dice loss should be non-negative"
    
    # Test CombinedLoss
    combined_loss = CombinedLoss()
    loss_val = combined_loss(pred, target)
    assert loss_val.item() >= 0, "Combined loss should be non-negative"
    
    # Test IoULoss
    iou_loss = IoULoss()
    loss_val = iou_loss(pred, target)
    assert loss_val.item() >= 0, "IoU loss should be non-negative"
    
    # Test BBoxLoss
    bbox_loss = BBoxLoss()
    pred_bbox = torch.randn(2, 4)
    target_bbox = torch.randn(2, 4)
    loss_val = bbox_loss(pred_bbox, target_bbox)
    assert loss_val.item() >= 0, "BBox loss should be non-negative"
    
    print("✓ All loss functions working")
except Exception as e:
    print(f"✗ Loss function test failed: {e}")
    sys.exit(1)

# Test 9: Test evaluation metrics
print("\n9. Testing evaluation metrics...")
try:
    pred = torch.randn(4, 1, 128, 128)
    target = torch.randint(0, 2, (4, 1, 128, 128)).float()
    
    # Test IoU
    iou = calculate_iou(pred, target)
    assert 0 <= iou <= 1, f"IoU should be in [0, 1], got {iou}"
    
    # Test Dice
    dice = calculate_dice(pred, target)
    assert 0 <= dice <= 1, f"Dice should be in [0, 1], got {dice}"
    
    # Test Accuracy
    accuracy = calculate_pixel_accuracy(pred, target)
    assert 0 <= accuracy <= 1, f"Accuracy should be in [0, 1], got {accuracy}"
    
    # Test bbox extraction
    bbox = extract_bbox_from_mask(target)
    assert bbox.shape == (4, 4), f"Expected bbox shape (4, 4), got {bbox.shape}"
    
    print(f"✓ Evaluation metrics working (IoU: {iou:.4f}, Dice: {dice:.4f}, Acc: {accuracy:.4f})")
except Exception as e:
    print(f"✗ Evaluation metrics test failed: {e}")
    sys.exit(1)

# Test 10: Test model factory
print("\n10. Testing model factory...")
try:
    # Test different model types
    mask_cnn_model = get_model('mask_cnn', pretrained=False)
    seg_only_model = get_model('segmentation', pretrained=False)
    combined_model = get_model('combined', use_mask_cnn=True, pretrained=False)
    
    print("✓ Model factory working")
except Exception as e:
    print(f"✗ Model factory test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("ALL TESTS PASSED! ✓")
print("=" * 80)
print("\nThe pipeline is ready to use. You can now run:")
print("  python seg_pipeline.py --verbose --logfile seg_log.log")
print("\nFor more options, see:")
print("  python seg_pipeline.py --help")
print("=" * 80)
