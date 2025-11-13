"""
Model Visualization Utilities

Provides functions to visualize PyTorch model architectures in a paper-ready format.
Generates both text-based summaries and graphical representations suitable for
academic publications.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_layer_info(model: nn.Module) -> List[Dict[str, Any]]:
    """
    Extract detailed layer information from model
    
    Args:
        model: PyTorch model
        
    Returns:
        List of dictionaries containing layer information
    """
    layers_info = []
    
    for name, module in model.named_modules():
        # Skip the root module and containers without parameters
        if name == '' or isinstance(module, (nn.Sequential, nn.ModuleList)):
            continue
        
        # Get parameter count for this layer
        params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # Get layer type
        layer_type = module.__class__.__name__
        
        # Get output shape if available (requires forward pass)
        output_shape = "N/A"
        
        # Get specific layer properties
        properties = {}
        if isinstance(module, nn.Conv2d):
            properties = {
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'stride': module.stride,
                'padding': module.padding,
            }
        elif isinstance(module, nn.Linear):
            properties = {
                'in_features': module.in_features,
                'out_features': module.out_features,
            }
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            properties = {
                'num_features': module.num_features,
            }
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
            properties = {
                'kernel_size': module.kernel_size,
                'stride': module.stride,
            }
        
        layers_info.append({
            'name': name,
            'type': layer_type,
            'params': params,
            'trainable_params': trainable_params,
            'properties': properties,
        })
    
    return layers_info


def format_number(num: int) -> str:
    """Format large numbers with K/M/B suffixes"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def create_model_summary_table(model: nn.Module, model_name: str = "Model") -> str:
    """
    Create a LaTeX-style table summarizing the model architecture
    
    Args:
        model: PyTorch model
        model_name: Name of the model for the table caption
        
    Returns:
        Formatted string representation of the table
    """
    layers_info = get_layer_info(model)
    total_params, trainable_params = count_parameters(model)
    
    # Create table header
    table = f"\n{'='*80}\n"
    table += f"{model_name.upper()} ARCHITECTURE SUMMARY\n"
    table += f"{'='*80}\n\n"
    
    # Layer details
    table += f"{'Layer Name':<40} {'Type':<20} {'Parameters':<15}\n"
    table += f"{'-'*80}\n"
    
    for layer in layers_info:
        if layer['params'] > 0:  # Only show layers with parameters
            param_str = format_number(layer['params'])
            table += f"{layer['name']:<40} {layer['type']:<20} {param_str:<15}\n"
            
            # Add properties if available
            if layer['properties']:
                props_str = ", ".join(f"{k}={v}" for k, v in layer['properties'].items())
                if len(props_str) < 70:
                    table += f"  └─ {props_str}\n"
    
    table += f"{'-'*80}\n"
    table += f"{'TOTAL PARAMETERS':<40} {'':<20} {format_number(total_params):<15}\n"
    table += f"{'TRAINABLE PARAMETERS':<40} {'':<20} {format_number(trainable_params):<15}\n"
    table += f"{'NON-TRAINABLE PARAMETERS':<40} {'':<20} {format_number(total_params - trainable_params):<15}\n"
    table += f"{'='*80}\n"
    
    return table


def visualize_model_architecture(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 256, 256),
    model_name: str = "Model",
    save_path: Optional[str] = None,
    show_parameters: bool = True
) -> None:
    """
    Create a publication-ready visualization of model architecture
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)
        model_name: Name of the model
        save_path: Path to save the figure (if None, displays only)
        show_parameters: Whether to show parameter counts
    """
    layers_info = get_layer_info(model)
    
    # Filter to major layers for visualization
    major_layers = [
        layer for layer in layers_info
        if layer['params'] > 0 and layer['type'] in [
            'Conv2d', 'Linear', 'BatchNorm2d', 'MaxPool2d', 'AvgPool2d',
            'ConvTranspose2d', 'Dropout', 'ReLU', 'Sigmoid', 'Tanh'
        ]
    ]
    
    if not major_layers:
        major_layers = [layer for layer in layers_info if layer['params'] > 0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(major_layers) * 0.5)))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(major_layers) + 2)
    ax.axis('off')
    
    # Title
    ax.text(5, len(major_layers) + 1.5, f"{model_name} Architecture",
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Color mapping for layer types
    color_map = {
        'Conv2d': '#3498db',
        'ConvTranspose2d': '#9b59b6',
        'Linear': '#e74c3c',
        'BatchNorm2d': '#2ecc71',
        'MaxPool2d': '#f39c12',
        'AvgPool2d': '#f39c12',
        'Dropout': '#95a5a6',
        'ReLU': '#1abc9c',
        'Sigmoid': '#16a085',
        'Tanh': '#d35400',
    }
    
    # Draw layers
    y_pos = len(major_layers)
    for idx, layer in enumerate(major_layers):
        color = color_map.get(layer['type'], '#34495e')
        
        # Draw rectangle for layer
        rect = mpatches.FancyBboxPatch(
            (1, y_pos - 0.4), 8, 0.8,
            boxstyle="round,pad=0.05",
            edgecolor='black',
            facecolor=color,
            alpha=0.7,
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Layer name and type
        layer_text = f"{layer['type']}"
        if show_parameters and layer['params'] > 0:
            layer_text += f"\n{format_number(layer['params'])} params"
        
        ax.text(5, y_pos, layer_text,
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Properties on the side
        if layer['properties']:
            props_str = ", ".join(f"{k}={v}" for k, v in list(layer['properties'].items())[:2])
            ax.text(9.5, y_pos, props_str, ha='left', va='center', fontsize=7, style='italic')
        
        # Draw arrow to next layer
        if idx < len(major_layers) - 1:
            ax.arrow(5, y_pos - 0.5, 0, -0.4, head_width=0.2, head_length=0.1,
                    fc='gray', ec='gray', alpha=0.5)
        
        y_pos -= 1
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=color, edgecolor='black', label=layer_type, alpha=0.7)
        for layer_type, color in color_map.items()
        if any(layer['type'] == layer_type for layer in major_layers)
    ]
    
    if legend_elements:
        ax.legend(handles=legend_elements[:6], loc='upper left', bbox_to_anchor=(0, 0),
                 fontsize=8, framealpha=0.9)
    
    # Parameter count summary
    total_params, trainable_params = count_parameters(model)
    summary_text = f"Total Parameters: {format_number(total_params)}\n"
    summary_text += f"Trainable: {format_number(trainable_params)}"
    
    ax.text(5, -0.5, summary_text, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model visualization saved to: {save_path}")
    
    plt.show()


def print_model_summary(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 3, 256, 256),
    model_name: str = "Model"
) -> None:
    """
    Print a comprehensive model summary to console
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        model_name: Name of the model
    """
    print(create_model_summary_table(model, model_name))
    
    # Try to get output shape with a forward pass
    try:
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(input_size)
            output = model(dummy_input)
            
            if isinstance(output, torch.Tensor):
                print(f"\nInput Shape:  {list(input_size)}")
                print(f"Output Shape: {list(output.shape)}")
            elif isinstance(output, (list, tuple)):
                print(f"\nInput Shape:  {list(input_size)}")
                print(f"Output Shapes:")
                for idx, out in enumerate(output):
                    print(f"  Output {idx}: {list(out.shape)}")
    except Exception as e:
        print(f"\nCould not determine output shape: {e}")


def compare_models(
    models: Dict[str, nn.Module],
    save_path: Optional[str] = None
) -> None:
    """
    Create a comparison table of multiple models
    
    Args:
        models: Dictionary of {model_name: model}
        save_path: Path to save the comparison table
    """
    comparison = []
    
    for name, model in models.items():
        total_params, trainable_params = count_parameters(model)
        layers_info = get_layer_info(model)
        
        # Count layer types
        conv_layers = sum(1 for l in layers_info if 'Conv' in l['type'])
        linear_layers = sum(1 for l in layers_info if 'Linear' in l['type'])
        
        comparison.append({
            'Model': name,
            'Total Params': format_number(total_params),
            'Trainable Params': format_number(trainable_params),
            'Conv Layers': conv_layers,
            'Linear Layers': linear_layers,
        })
    
    # Print comparison table
    print("\n" + "="*90)
    print("MODEL COMPARISON")
    print("="*90)
    print(f"{'Model':<25} {'Total Params':<15} {'Trainable':<15} {'Conv':<10} {'Linear':<10}")
    print("-"*90)
    
    for row in comparison:
        print(f"{row['Model']:<25} {row['Total Params']:<15} {row['Trainable Params']:<15} "
              f"{row['Conv Layers']:<10} {row['Linear Layers']:<10}")
    
    print("="*90)
    
    if save_path:
        # Create a visual comparison
        fig, ax = plt.subplots(figsize=(12, max(6, len(models) * 0.8)))
        
        model_names = [c['Model'] for c in comparison]
        params = [float(c['Total Params'].rstrip('MKB').replace(',', '')) for c in comparison]
        
        # Determine unit from first model
        unit = 'M' if 'M' in comparison[0]['Total Params'] else 'K' if 'K' in comparison[0]['Total Params'] else ''
        
        y_pos = range(len(model_names))
        bars = ax.barh(y_pos, params, color='#3498db', alpha=0.7, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names)
        ax.set_xlabel(f'Parameters ({unit})', fontsize=12, fontweight='bold')
        ax.set_title('Model Parameter Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, params)):
            ax.text(val, bar.get_y() + bar.get_height()/2, f' {comparison[i]["Total Params"]}',
                   va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison visualization saved to: {save_path}")
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("Model Visualization Utilities")
    print("Import this module to use visualization functions")
    print("\nExample:")
    print("  from utils.model_visualization import print_model_summary, visualize_model_architecture")
    print("  model = YourModel()")
    print("  print_model_summary(model, input_size=(1, 3, 256, 256), model_name='YourModel')")
    print("  visualize_model_architecture(model, save_path='model_arch.png')")
