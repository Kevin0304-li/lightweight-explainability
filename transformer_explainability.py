import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy.ndimage import zoom
import gc

def simplify_attention_map(attention_map, threshold_percent=10):
    """Simplify a transformer attention map by keeping only the top % most important connections
    
    Args:
        attention_map: Numpy array of attention weights
        threshold_percent: Percentage threshold (1-100)
        
    Returns:
        simplified: Thresholded attention map
    """
    # Handle multi-head attention
    if len(attention_map.shape) > 2:
        # Average across heads for simplicity
        attention_map = np.mean(attention_map, axis=0)
    
    # Apply percentile-based thresholding
    threshold = np.percentile(attention_map, 100 - threshold_percent)
    simplified = np.zeros_like(attention_map)
    simplified[attention_map > threshold] = attention_map[attention_map > threshold]
    
    return simplified

def attention_to_heatmap(attention, patch_size, img_shape):
    """Convert patch-based attention to pixel-space heatmap
    
    Args:
        attention: Attention weights
        patch_size: Size of image patches in ViT
        img_shape: Shape of original image
        
    Returns:
        heatmap: Attention heatmap resized to match image dimensions
    """
    # Extract CLS token attention to all patches (first token's attention)
    try:
        cls_attention = attention[0, 0, 1:]  # Skip attention to CLS token itself
        
        # Determine number of patches
        num_patches = cls_attention.shape[0]
        
        # Dynamically calculate grid dimensions based on number of patches
        grid_size = int(np.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            # Not a perfect square, create a default square grid
            print(f"Warning: Cannot reshape {num_patches} patches into a square grid. Using default heatmap.")
            # Create a default heatmap of the expected size
            if len(img_shape) >= 3:
                heatmap = np.ones((img_shape[-2], img_shape[-1]))
            else:
                heatmap = np.ones((224, 224))
            return heatmap
        
        # Reshape to patch grid
        attention_grid = cls_attention.reshape(grid_size, grid_size)
        
        # Upsample to original image size
        if len(img_shape) >= 3:
            scale_h = img_shape[-2] / attention_grid.shape[0]
            scale_w = img_shape[-1] / attention_grid.shape[1]
        else:
            scale_h = 224 / attention_grid.shape[0]
            scale_w = 224 / attention_grid.shape[1]
        
        return zoom(attention_grid, (scale_h, scale_w), order=1)
    except Exception as e:
        print(f"Error in attention_to_heatmap: {e}")
        # Return a default heatmap in case of error
        if len(img_shape) >= 3:
            return np.ones((img_shape[-2], img_shape[-1]))
        else:
            return np.ones((224, 224))

# New function to analyze contributions of each attention head
def analyze_attention_heads(attention_maps, num_top_heads=3):
    """Analyze the contributions of different attention heads
    
    Args:
        attention_maps: List of attention tensors from different layers
        num_top_heads: Number of top heads to highlight
        
    Returns:
        head_importance: Dictionary with head importance metrics
    """
    head_scores = {}
    
    for layer_idx, attn in enumerate(attention_maps):
        # Skip if not a proper attention tensor
        if not isinstance(attn, torch.Tensor) or len(attn.shape) < 4:
            continue
            
        # Get number of heads
        num_heads = attn.shape[1]
        
        for head_idx in range(num_heads):
            # Extract attention for this head
            head_attn = attn[0, head_idx].detach().cpu().numpy()
            
            # Calculate concentration score (higher = more focused attention)
            # We use Gini coefficient as a measure of concentration
            head_attn_flat = head_attn.flatten()
            head_attn_flat.sort()
            n = len(head_attn_flat)
            index = np.arange(1, n+1)
            gini = 2 * np.sum(index * head_attn_flat) / (n * np.sum(head_attn_flat)) - (n + 1) / n
            
            # Calculate entropy (lower = more focused attention)
            # Normalize attention to probabilities
            head_attn_prob = head_attn / np.sum(head_attn)
            entropy = -np.sum(head_attn_prob * np.log2(head_attn_prob + 1e-9))
            
            # Store scores
            head_scores[f"layer{layer_idx}_head{head_idx}"] = {
                "concentration": gini,
                "entropy": entropy,
                "layer": layer_idx,
                "head": head_idx
            }
    
    # Find top heads by concentration
    if head_scores:
        top_heads = sorted(head_scores.items(), key=lambda x: x[1]["concentration"], reverse=True)[:num_top_heads]
        top_head_ids = [{"layer": v["layer"], "head": v["head"]} for k, v in top_heads]
    else:
        top_head_ids = []
        
    return {
        "head_scores": head_scores,
        "top_heads": top_head_ids
    }

# New function to support text transformers (like BERT)
def analyze_text_attention(attention_maps, tokens, num_tokens=None):
    """Analyze attention patterns in text transformers
    
    Args:
        attention_maps: List of attention tensors from different layers
        tokens: List of input tokens
        num_tokens: Number of tokens to include (None for all)
        
    Returns:
        token_importance: Dictionary with token importance metrics
    """
    if not attention_maps or not tokens:
        return {"error": "No attention maps or tokens provided"}
    
    # Take the last layer's attention by default
    if isinstance(attention_maps[-1], torch.Tensor):
        last_layer_attn = attention_maps[-1].detach().cpu().numpy()
    else:
        return {"error": "Unsupported attention format"}
    
    # Average across heads
    if len(last_layer_attn.shape) >= 4:
        avg_attn = np.mean(last_layer_attn[0], axis=0)  # [seq_len, seq_len]
    else:
        avg_attn = last_layer_attn
    
    # Calculate token importance (sum of attention received by each token)
    token_importance = np.sum(avg_attn, axis=0)
    
    # Limit number of tokens if specified
    num_tokens = min(len(tokens), len(token_importance)) if num_tokens is None else min(num_tokens, len(tokens))
    
    # Create token-importance pairs
    token_scores = [(tokens[i], token_importance[i]) for i in range(min(len(tokens), len(token_importance)))]
    
    # Sort by importance
    sorted_tokens = sorted(token_scores, key=lambda x: x[1], reverse=True)[:num_tokens]
    
    return {
        "token_importance": token_importance,
        "top_tokens": sorted_tokens
    }

class TransformerExplainer:
    """Class for explaining Vision Transformer models"""
    
    def __init__(self, model, patch_size=16, model_family="vit"):
        """Initialize the explainer
        
        Args:
            model: Vision Transformer model
            patch_size: Size of image patches used in the model
            model_family: Model architecture family ('vit', 'deit', 'swin', 'beit', 'bert', 'roberta', etc.)
        """
        self.model = model
        self.model.eval()
        self.patch_size = patch_size
        self.model_family = model_family.lower()
        self.attention_maps = []
        self.hook_handles = []
        self.last_memory_usage = 0
        self.last_processing_time = 0
        
        # Set model-specific parameters
        self.is_vision_transformer = model_family.lower() in ["vit", "deit", "swin", "beit", "clip"]
        self.is_text_transformer = model_family.lower() in ["bert", "roberta", "gpt", "t5", "bart"]
        
    def _register_hooks(self):
        """Register hooks to capture attention maps"""
        self.attention_maps = []
        
        def attention_hook(module, input, output):
            # Extract attention weights from output
            # Different transformer libraries store attention differently
            if hasattr(output, "attentions"):
                self.attention_maps.append(output.attentions)
            elif isinstance(output, tuple) and len(output) > 1 and hasattr(output[1], "attentions"):
                self.attention_maps.append(output[1].attentions)
            elif isinstance(output, dict) and "attentions" in output:
                self.attention_maps.append(output["attentions"])
            elif isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], tuple) and len(output[1]) > 0:
                # Handle HuggingFace pattern where attention is in output tuple
                attn_matrices = [tensor for tensor in output[1] if isinstance(tensor, torch.Tensor) and len(tensor.shape) == 4]
                if attn_matrices:
                    self.attention_maps.append(attn_matrices[0])
            else:
                # Fallback for custom models
                # This assumes attention is computed within the module and passed along
                # May need adjustment for specific transformer implementations
                attn_output = output
                if isinstance(output, tuple):
                    attn_output = output[0]
                self.attention_maps.append(attn_output)
        
        # Register hooks for all attention layers
        for name, module in self.model.named_modules():
            # Different architectures use different naming conventions
            attention_keywords = ["attn", "attention", "mha", "multihead"]
            
            # Add model-specific keywords
            if self.model_family == "vit":
                attention_keywords.extend(["attnblock", "msa"])
            elif self.model_family == "swin":
                attention_keywords.extend(["windowattention", "w_msa"])
            elif self.model_family in ["bert", "roberta"]:
                attention_keywords.extend(["selfattention", "bertattention"])
            
            if any(x in name.lower() for x in attention_keywords):
                handle = module.register_forward_hook(attention_hook)
                self.hook_handles.append(handle)
                
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
    def explain(self, input_data, target_class=None, threshold_percent=10, layer_idx=-1, head_idx=None, analyze_heads=False):
        """Generate explanation for Transformer model
        
        Args:
            input_data: Input tensor (image or text)
            target_class: Target class (None for predicted class)
            threshold_percent: Threshold for simplification
            layer_idx: Which transformer layer to use for explanation (-1 for last)
            head_idx: Which attention head to use (None for average of all heads)
            analyze_heads: Whether to return head analysis
            
        Returns:
            dict: Explanation results
        """
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
            
        start_time = time.time()
        
        # Register hooks
        self._register_hooks()
        
        # Handle different input types
        tokens = None
        if isinstance(input_data, tuple) and len(input_data) == 2 and self.is_text_transformer:
            # Handle text transformer input format (inputs, tokens)
            input_tensor, tokens = input_data
        else:
            input_tensor = input_data
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
        # Get predicted class if not specified
        if target_class is None:
            if hasattr(outputs, "logits"):
                predicted = outputs.logits.argmax(-1).item()
            else:
                predicted = outputs.argmax(-1).item()
            target_class = predicted
        
        # Process explanation based on model type
        explanation_result = {}
        
        if self.is_vision_transformer:
            try:
                # Vision transformer processing
                if self.attention_maps:
                    # Get specified layer or default to last layer
                    attention = self.attention_maps[layer_idx] 
                    
                    # Convert tensor to numpy if needed
                    if isinstance(attention, torch.Tensor):
                        attention = attention.detach().cpu().numpy()
                        
                    # Select specific head if requested
                    if head_idx is not None and len(attention.shape) > 2:
                        attention_slice = attention[0, head_idx]
                        attention_for_heatmap = np.expand_dims(attention_slice, axis=0)
                    else:
                        attention_for_heatmap = attention
                    
                    # Convert attention to heatmap
                    heatmap = attention_to_heatmap(attention_for_heatmap, self.patch_size, input_tensor.shape)
                    
                    # Apply simplification
                    simplified_heatmap = simplify_attention_map(heatmap, threshold_percent)
                    
                    # Add to result
                    explanation_result.update({
                        "heatmap": heatmap,
                        "simplified_heatmap": simplified_heatmap,
                    })
                    
                    # Add head analysis if requested
                    if analyze_heads:
                        head_analysis = analyze_attention_heads(self.attention_maps)
                        explanation_result["head_analysis"] = head_analysis
                else:
                    # Fallback if no attention maps were captured
                    print("Warning: No attention maps captured")
                    heatmap = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
                    explanation_result.update({
                        "heatmap": heatmap,
                        "simplified_heatmap": heatmap,
                    })
            except Exception as e:
                print(f"Error generating vision transformer explanation: {e}")
                heatmap = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
                explanation_result.update({
                    "heatmap": heatmap,
                    "simplified_heatmap": heatmap,
                })
        
        elif self.is_text_transformer and tokens:
            try:
                # Text transformer processing
                token_analysis = analyze_text_attention(self.attention_maps, tokens)
                explanation_result["token_analysis"] = token_analysis
            except Exception as e:
                print(f"Error generating text transformer explanation: {e}")
                explanation_result["error"] = str(e)
        
        # Remove hooks
        self._remove_hooks()
        
        # Memory tracking end
        if torch.cuda.is_available():
            end_mem = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()
            self.last_memory_usage = peak_mem - start_mem
        else:
            self.last_memory_usage = 0
            
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        self.last_processing_time = processing_time
        
        # Add common fields to result
        explanation_result.update({
            "target_class": target_class,
            "processing_time": processing_time,
            "memory_usage": self.last_memory_usage,
            "model_family": self.model_family
        })
        
        return explanation_result
        
    def visualize_explanation(self, img, explanation_result, title=None, save_path=None, show_heads=False):
        """Visualize the explanation heatmap overlaid on the original image
        
        Args:
            img: PIL Image or numpy array of original image
            explanation_result: Result from explain() method
            title: Optional title for the plot
            save_path: Optional path to save the visualization
            show_heads: Whether to visualize individual head contributions
            
        Returns:
            PIL Image of the visualization
        """
        # Import libraries
        import matplotlib.pyplot as plt
        
        # Handle text transformers
        if self.is_text_transformer and "token_analysis" in explanation_result:
            return self._visualize_text_explanation(explanation_result, save_path)
            
        # Vision transformer visualization
        # Convert PIL Image to numpy if needed
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img.copy()
            
        # Get heatmaps
        heatmap = explanation_result.get("heatmap", None)
        simplified_heatmap = explanation_result.get("simplified_heatmap", None)
        
        if heatmap is None or simplified_heatmap is None:
            print("Warning: No heatmaps found in explanation result")
            return None
            
        # Create figure and subplots
        if show_heads and "head_analysis" in explanation_result:
            # Show original + full attention + simplified + top heads
            top_heads = explanation_result["head_analysis"]["top_heads"]
            num_heads = len(top_heads)
            plt.figure(figsize=(5*(3+num_heads), 5))
            
            # Original image
            plt.subplot(1, 3+num_heads, 1)
            plt.imshow(img_np)
            plt.title("Original Image")
            plt.axis('off')
            
            # Full attention heatmap
            plt.subplot(1, 3+num_heads, 2)
            plt.imshow(img_np)
            plt.imshow(heatmap, alpha=0.5, cmap='jet')
            plt.title("Full Attention")
            plt.axis('off')
            
            # Simplified attention heatmap
            plt.subplot(1, 3+num_heads, 3)
            plt.imshow(img_np)
            plt.imshow(simplified_heatmap, alpha=0.5, cmap='jet')
            plt.title(f"Simplified Attention ({explanation_result.get('threshold_percent', 10)}%)")
            plt.axis('off')
            
            # Top heads
            for i, head_info in enumerate(top_heads):
                layer_idx = head_info["layer"]
                head_idx = head_info["head"]
                
                # Get this head's attention
                if layer_idx < len(self.attention_maps):
                    head_attn = self.attention_maps[layer_idx][0, head_idx].detach().cpu().numpy()
                    head_heatmap = attention_to_heatmap(np.expand_dims(head_attn, 0), 
                                                        self.patch_size, img_np.shape)
                    
                    plt.subplot(1, 3+num_heads, 4+i)
                    plt.imshow(img_np)
                    plt.imshow(head_heatmap, alpha=0.5, cmap='jet')
                    plt.title(f"Layer {layer_idx} Head {head_idx}")
                    plt.axis('off')
        else:
            # Standard visualization with 3 subplots
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title("Original Image")
            plt.axis('off')
            
            # Full attention heatmap
            plt.subplot(1, 3, 2)
            plt.imshow(img_np)
            plt.imshow(heatmap, alpha=0.5, cmap='jet')
            plt.title("Full Attention")
            plt.axis('off')
            
            # Simplified attention heatmap
            plt.subplot(1, 3, 3)
            plt.imshow(img_np)
            plt.imshow(simplified_heatmap, alpha=0.5, cmap='jet')
            plt.title(f"Simplified Attention ({explanation_result.get('threshold_percent', 10)}%)")
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            
        # Create a separate figure for return value
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img_np)
        ax.imshow(simplified_heatmap, alpha=0.5, cmap='jet')
        ax.set_title(title if title else f"Transformer Explanation ({explanation_result.get('model_family', 'vit')})")
        ax.axis('off')
        fig.tight_layout()
        
        # Convert matplotlib figure to PIL image
        fig.canvas.draw()
        plot_img = Image.frombytes('RGB', 
                                 fig.canvas.get_width_height(),
                                 fig.canvas.tostring_rgb())
        plt.close(fig)
        plt.close()
        
        return plot_img

    def _visualize_text_explanation(self, explanation_result, save_path=None):
        """Visualize text transformer explanation
        
        Args:
            explanation_result: Result from explain method containing token analysis
            save_path: Optional path to save visualization
            
        Returns:
            PIL Image of visualization
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        
        token_analysis = explanation_result.get("token_analysis", {})
        top_tokens = token_analysis.get("top_tokens", [])
        
        if not top_tokens:
            print("Warning: No token analysis available")
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract tokens and values
        tokens = [t[0] for t in top_tokens]
        values = [t[1] for t in top_tokens]
        
        # Create color map
        cmap = LinearSegmentedColormap.from_list("importance", ["#f7fbff", "#08306b"])
        normalized_values = np.array(values) / max(values)
        
        # Create visualization
        y_pos = np.arange(len(tokens))
        ax.barh(y_pos, values, color=[cmap(v) for v in normalized_values])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens)
        ax.invert_yaxis()  # Show most important token at top
        ax.set_xlabel("Attention Score")
        ax.set_title("Important Tokens by Attention Score")
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        # Convert to PIL image
        fig.canvas.draw()
        plot_img = Image.frombytes('RGB', 
                                 fig.canvas.get_width_height(),
                                 fig.canvas.tostring_rgb())
        plt.close(fig)
        
        return plot_img 