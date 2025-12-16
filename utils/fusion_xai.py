import os
import glob
import re
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# --- 1. UTILITIES & HELPERS ---

def get_target_layer(model):
    """
    Intelligently selects the last convolutional layer for Grad-CAM.
    Handles standard torchvision models and your peer's custom wrappers.
    """
    name = model.__class__.__name__
    
    # Custom Wrapper (ResNetForImageClassification)
    if name == "ResNetForImageClassification":
        try:
            return model.resnet.encoder.stages[-1]
        except AttributeError:
            pass 

    # SqueezeNet / Custom
    if "SqueezeNet" in name:
        if hasattr(model, 'features'): return model.features[-1]
        elif hasattr(model, 'squeezenet'): return model.squeezenet.features[-1]

    # Standard Architectures
    if "ResNet" in name or "SeResNet" in name or "SeResNext" in name:
        return model.layer4[-1]
    elif "DenseNet" in name:
        return model.features.denseblock4
    elif "Inception" in name:
        return model.Mixed_7c
    elif "DPN" in name:
        return model.features[-1]

    # Generic Fallback: Search backwards for last Conv2d
    layers = list(model.modules())
    for i in range(len(layers) - 1, -1, -1):
        if isinstance(layers[i], torch.nn.Conv2d):
            return layers[i]
            
    raise ValueError(f"Could not find a target layer for {name}")

def find_smart_key(target_name, available_keys):
    """
    Matches specific model names (resnet34_1) to generic keys (resnet34).
    """
    # Exact match
    if target_name in available_keys:
        return target_name, 0

    # Regex for numbered suffixes
    match = re.search(r'(.+)_(\d+)$', target_name)
    if match:
        base_name = match.group(1)
        idx = int(match.group(2)) - 1
        if base_name in available_keys:
            return base_name, idx
        search_name = base_name
    else:
        search_name = target_name
        idx = 0

    # Longest Prefix Match
    sorted_keys = sorted(list(available_keys), key=len, reverse=True)
    for key in sorted_keys:
        if search_name.startswith(key):
            return key, idx
            
    return None, 0

# --- 2. CORE GRAD-CAM LOGIC ---

class SingleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_out):
        self.gradients = grad_out[0] 

    def __call__(self, x):
        # Forward
        output = self.model(x)
        
        # Handle HuggingFace/Tuple outputs
        if hasattr(output, 'logits'): logits = output.logits
        elif isinstance(output, tuple): logits = output[0]
        else: logits = output
            
        probs = F.softmax(logits, dim=1)
        class_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, class_idx].item()

        # Backward
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1
        logits.backward(gradient=one_hot, retain_graph=True)

        # Map Generation
        if self.gradients is None or self.activations is None:
            return np.zeros((x.shape[2], x.shape[3])), class_idx, 0.0

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activation = self.activations[0]
        
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU
        
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap, class_idx, confidence

class FusionGradCAM:
    def __init__(self, model_pairs):
        self.extractors = []
        for model, layer in model_pairs:
            try:
                self.extractors.append(SingleGradCAM(model, layer))
            except Exception as e:
                print(f"[Warning] Hook registration failed: {e}")

    def __call__(self, input_tensor, input_image_size=(224, 224)):
        votes, confidences, heatmaps = [], [], []

        for extractor in self.extractors:
            try:
                hmap, pred, conf = extractor(input_tensor)
                votes.append(pred)
                confidences.append(conf)
                heatmaps.append(hmap)
            except Exception:
                # Silently fail for individual models to keep ensemble running
                votes.append(-1)
                confidences.append(0)
                heatmaps.append(np.zeros(input_image_size))

        valid_votes = [v for v in votes if v != -1]
        if not valid_votes:
            return np.zeros(input_image_size), -1

        # Majority Vote
        votes_tensor = torch.tensor(valid_votes)
        majority_class = torch.mode(votes_tensor).values.item()

        # Weighted Fusion
        fused_heatmap = np.zeros(input_image_size)
        total_weight = 0

        for i in range(len(votes)):
            if votes[i] == majority_class and confidences[i] > 0:
                hmap_resized = cv2.resize(heatmaps[i], input_image_size)
                weight = confidences[i]
                fused_heatmap += hmap_resized * weight
                total_weight += weight

        if total_weight > 0:
            fused_heatmap /= total_weight

        return fused_heatmap, majority_class

# --- 3. MAIN WRAPPER CLASS ---
class EnsembleXAI:
    """
    Main class for the peer.
    Usage:
        xai = EnsembleXAI(ensemble_wrapper, device='cuda')
        heatmap, pred_label = xai.explain_image("path/to/image.jpg")
    """
    def __init__(self, ensemble_wrapper, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_mapping = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        
        # Preprocessing (Standard ImageNet)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        print("Initializing Ensemble XAI...")
        self.models = self._load_models(ensemble_wrapper)
        
        print("Hooking Grad-CAM layers...")
        self.model_pairs = []
        for m in self.models:
            try:
                target = get_target_layer(m)
                self.model_pairs.append((m, target))
            except Exception as e:
                print(f"  [Skip] {type(m).__name__}: {e}")
                
        self.engine = FusionGradCAM(self.model_pairs)
        print(f"Ready! Loaded {len(self.model_pairs)} models.")

    def _load_models(self, wrapper):
        """ Reconstructs models from the Hydra wrapper using Smart Matching """
        model_order = wrapper.model_order
        model_defs = wrapper.model_defs
        model_paths = wrapper.base_model_paths
        loaded_models = []

        for name in model_order:
            # 1. Architecture
            def_key, _ = find_smart_key(name, model_defs.keys())
            if not def_key: continue
            
            model = copy.deepcopy(model_defs[def_key]['params'])

            # 2. Weights
            path_key, path_idx = find_smart_key(name, model_paths.keys())
            if path_key:
                path_list = model_paths[path_key]
                if path_idx >= len(path_list): path_idx = 0
                
                model_dir = path_list[path_idx]
                if not os.path.exists(model_dir): model_dir = os.path.abspath(model_dir)
                
                ckpt_files = glob.glob(os.path.join(model_dir, "*.pt"))
                if ckpt_files:
                    try:
                        ckpt = torch.load(ckpt_files[0], map_location=self.device, weights_only=False)
                        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
                        
                        # Fix prefixes
                        if list(sd.keys())[0].startswith('model.') and not list(model.state_dict().keys())[0].startswith('model.'):
                            sd = {k.replace("model.", ""): v for k, v in sd.items()}
                        
                        model.load_state_dict(sd, strict=False)
                    except Exception:
                        pass # Ignore loading errors, use random weights if must

            model.to(self.device)
            model.eval()
            # Fix inplace ReLU for Grad-CAM
            for m in model.modules():
                if isinstance(m, nn.ReLU): m.inplace = False
            
            loaded_models.append(model)
            
        return loaded_models

    def explain_image(self, image_source):
        """
        Input: 
            image_source: path to jpg (str) OR PIL Image object
        Returns:
            heatmap (numpy 224x224), predicted_label (str)
        """
        # 1. Load & Preprocess
        if isinstance(image_source, str):
            image = Image.open(image_source).convert('RGB')
        else:
            image = image_source.convert('RGB')

        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        input_tensor.requires_grad = True

        # 2. Run Fusion
        heatmap, pred_idx = self.engine(input_tensor)
        
        # 3. Decode Label
        label_name = self.idx_to_label.get(pred_idx, "Unknown")
        
        return heatmap, label_name
