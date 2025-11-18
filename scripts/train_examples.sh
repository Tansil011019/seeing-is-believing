#!/bin/bash
# Example training commands for ISIC2018 unified pipeline

echo "=== ISIC2018 Training Examples ==="
echo ""

# Task 1: Segmentation Examples
echo "--- Task 1: Segmentation Training ---"
echo ""

echo "1. Basic training with SegFormer (default):"
echo "   python seg_attr_train.py --config-name hydra-task1/segmentation"
echo ""

echo "2. Train with BEiT encoder:"
echo "   python seg_attr_train.py --config-name hydra-task1/segmentation model.name=beit"
echo ""

echo "3. Train with custom batch size and learning rate:"
echo "   python seg_attr_train.py --config-name hydra-task1/segmentation \\"
echo "       training.batch_size=16 training.learning_rate=2e-4"
echo ""

echo "4. Train for longer with larger batches:"
echo "   python seg_attr_train.py --config-name hydra-task1/segmentation \\"
echo "       training.epochs=150 training.batch_size=12"
echo ""

# Task 2: Attribute Detection Examples
echo "--- Task 2: Attribute Detection Training ---"
echo ""

echo "5. Basic training with ViT-Base (default):"
echo "   python seg_attr_train.py --config-name hydra-task2/attribute"
echo ""

echo "6. Train with Swin Transformer:"
echo "   python seg_attr_train.py --config-name hydra-task2/attribute model.name=swin-base"
echo ""

echo "7. Train with ConvNeXt:"
echo "   python seg_attr_train.py --config-name hydra-task2/attribute model.name=convnext-base"
echo ""

echo "8. Train with larger hidden dimension and less dropout:"
echo "   python seg_attr_train.py --config-name hydra-task2/attribute \\"
echo "       model.hidden_dim=512 model.dropout=0.2"
echo ""

echo "9. Train with adjusted threshold and batch size:"
echo "   python seg_attr_train.py --config-name hydra-task2/attribute \\"
echo "       data.feat_threshold=0.05 training.batch_size=32"
echo ""

# Multi-run examples
echo "--- Multi-Run Experiments ---"
echo ""

echo "10. Compare all Task 2 models:"
echo "   python seg_attr_train.py --config-name hydra-task2/attribute -m \\"
echo "       model.name=vit-base,swin-base,convnext-base"
echo ""

echo "11. Learning rate sweep for Task 1:"
echo "   python seg_attr_train.py --config-name hydra-task1/segmentation -m \\"
echo "       training.learning_rate=1e-3,1e-4,1e-5"
echo ""

echo "12. Grid search over batch sizes and dropout (Task 2):"
echo "   python seg_attr_train.py --config-name hydra-task2/attribute -m \\"
echo "       training.batch_size=16,32 model.dropout=0.2,0.3,0.5"
echo ""

echo "=== Notes ==="
echo "- Add 'training.mixed_precision=false' to disable AMP (slower but more stable)"
echo "- Use 'data.num_workers=8' for faster data loading"
echo "- Check outputs in: outputs/plots/task1/ and outputs/plots/task2/"
echo "- Checkpoints saved in: checkpoints/task{1,2}_{model_name}/"
