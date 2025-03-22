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
    cls_attention = attention[0, 0, 1:]  # Skip attention to CLS token itself
    
    # Determine grid dimensions
    h_patches = img_shape[2] // patch_size
    w_patches = img_shape[3] // patch_size
    
    # Reshape to patch grid
    attention_grid = cls_attention.reshape(h_patches, w_patches)
    
    # Upsample to original image size
    scale_h = img_shape[2] / attention_grid.shape[0]
    scale_w = img_shape[3] / attention_grid.shape[1]
    
    return zoom(attention_grid, (scale_h, scale_w), order=1)

class TransformerExplainer:
    """Class for explaining Vision Transformer models"""
    
    def __init__(self, model, patch_size=16):
        """Initialize the explainer
        
        Args:
            model: Vision Transformer model
            patch_size: Size of image patches used in the model
        """
        self.model = model
        self.model.eval()
        self.patch_size = patch_size
        self.attention_maps = []
        self.hook_handles = []
        self.last_memory_usage = 0
        self.last_processing_time = 0
        
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
            if any(x in name.lower() for x in ["attn", "attention", "mha", "multihead"]):
                handle = module.register_forward_hook(attention_hook)
                self.hook_handles.append(handle)
                
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
    def explain(self, img_tensor, target_class=None, threshold_percent=10, layer_idx=-1):
        """Generate explanation for Vision Transformer
        
        Args:
            img_tensor: Input image tensor
            target_class: Target class (None for predicted class)
            threshold_percent: Threshold for simplification
            layer_idx: Which transformer layer to use for explanation (-1 for last)
            
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
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
        # Get predicted class if not specified
        if target_class is None:
            if hasattr(outputs, "logits"):
                predicted = outputs.logits.argmax(-1).item()
            else:
                predicted = outputs.argmax(-1).item()
            target_class = predicted
            
        # Get selected layer attention maps
        try:
            # Different transformer libraries format attention differently
            if self.attention_maps:
                attention = self.attention_maps[layer_idx]
                if isinstance(attention, torch.Tensor):
                    attention = attention.detach().cpu().numpy()
                
                # Convert attention to heatmap
                heatmap = attention_to_heatmap(attention, self.patch_size, img_tensor.shape)
                
                # Apply simplification
                simplified_heatmap = simplify_attention_map(heatmap, threshold_percent)
            else:
                # Fallback if no attention maps were captured
                print("Warning: No attention maps captured")
                heatmap = np.zeros((img_tensor.shape[2], img_tensor.shape[3]))
                simplified_heatmap = heatmap
                
        except Exception as e:
            print(f"Error generating transformer explanation: {e}")
            heatmap = np.zeros((img_tensor.shape[2], img_tensor.shape[3]))
            simplified_heatmap = heatmap
            
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
        
        return {
            "heatmap": heatmap,
            "simplified_heatmap": simplified_heatmap,
            "target_class": target_class,
            "processing_time": processing_time,
            "memory_usage": self.last_memory_usage
        }
        
    def visualize_explanation(self, img, explanation_result, title=None, save_path=None):
        """Visualize the explanation heatmap overlaid on the original image
        
        Args:
            img: PIL Image or numpy array of original image
            explanation_result: Result from explain() method
            title: Optional title for the plot
            save_path: Optional path to save the visualization
            
        Returns:
            PIL Image of the visualization
        """
        # Import libraries
        import matplotlib.pyplot as plt
        
        # Convert PIL Image to numpy if needed
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img.copy()
            
        # Get heatmaps
        heatmap = explanation_result["heatmap"]
        simplified_heatmap = explanation_result["simplified_heatmap"]
        
        # Create figure and subplots
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
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img_np)
        ax.imshow(simplified_heatmap, alpha=0.5, cmap='jet')
        ax.set_title(title if title else "Transformer Explanation")
        ax.axis('off')
        fig.tight_layout(pad=0)
        
        # Convert matplotlib figure to PIL image
        fig.canvas.draw()
        overlay_img = Image.frombytes('RGB', 
                                    fig.canvas.get_width_height(),
                                    fig.canvas.tostring_rgb())
        plt.close(fig)
        plt.close()
        
        return overlay_img 