import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import gc

def simplify_temporal_importance(importance, threshold_percent=10):
    """Simplify temporal importance map by keeping only the top % most important timesteps
    
    Args:
        importance: Numpy array of importance values per timestep
        threshold_percent: Percentage threshold (1-100)
        
    Returns:
        simplified: Thresholded importance map
    """
    threshold = np.percentile(importance, 100 - threshold_percent)
    simplified = np.zeros_like(importance)
    simplified[importance > threshold] = importance[importance > threshold]
    return simplified

class RNNExplainer:
    """Class for explaining RNN/LSTM/GRU models"""
    
    def __init__(self, model):
        """Initialize the explainer
        
        Args:
            model: RNN-based model
        """
        self.model = model
        self.model.eval()
        self.hidden_states = []
        self.gradients = []
        self.hook_handles = []
        self.last_memory_usage = 0
        self.last_processing_time = 0
        
    def _register_hooks(self):
        """Register hooks to capture hidden states and gradients"""
        self.hidden_states = []
        self.gradients = []
        
        # Hook function to capture hidden states
        def capture_hidden_states(module, input, output):
            if isinstance(output, tuple):  # LSTM returns (hidden_state, cell_state)
                self.hidden_states.append(output[0].detach())
            else:
                self.hidden_states.append(output.detach())
        
        # Hook function to capture gradients
        def capture_gradients(module, grad_input, grad_output):
            if isinstance(grad_input, tuple) and len(grad_input) > 0:
                self.gradients.append(grad_input[0].detach())
            else:
                self.gradients.append(grad_input.detach())
        
        # Register hooks for RNN layers
        for name, module in self.model.named_modules():
            if any(x in name.lower() for x in ["lstm", "gru", "rnn"]):
                h1 = module.register_forward_hook(capture_hidden_states)
                h2 = module.register_full_backward_hook(capture_gradients)
                self.hook_handles.extend([h1, h2])
                
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
    def explain(self, sequence_tensor, target_class=None, threshold_percent=10):
        """Generate explanation for RNN model showing temporal importance
        
        Args:
            sequence_tensor: Input sequence tensor [batch_size, seq_len, features]
            target_class: Target class (None for predicted class)
            threshold_percent: Threshold for simplification
            
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
        self.model.zero_grad()
        outputs = self.model(sequence_tensor)
        
        # Get predicted class if not specified
        if target_class is None:
            if isinstance(outputs, dict) and "logits" in outputs:
                predicted = outputs["logits"].argmax(-1).item()
            else:
                predicted = outputs.argmax(-1).item()
            target_class = predicted
            
        # Create one-hot for target class
        if isinstance(outputs, dict) and "logits" in outputs:
            output_tensor = outputs["logits"]
        else:
            output_tensor = outputs
            
        one_hot = torch.zeros_like(output_tensor)
        one_hot[0, target_class] = 1
        
        # Backward pass
        output_tensor.backward(gradient=one_hot, retain_graph=True)
        
        try:
            # Get hidden states and gradients
            final_hidden = self.hidden_states[-1].cpu().numpy()
            final_grad = self.gradients[-1].cpu().numpy()
            
            # Calculate importance per time step
            # Element-wise product of hidden state and its gradient
            importance = np.abs(final_hidden * final_grad)
            
            # Sum across feature dimension
            importance = np.sum(importance, axis=-1).squeeze()
            
            # If 3D tensor (batch, seq_len, hidden_dim), take first batch
            if len(importance.shape) > 1:
                importance = importance[0]
                
            # Normalize
            if importance.max() > 0:
                importance = importance / importance.max()
                
            # Apply threshold
            simplified_importance = simplify_temporal_importance(importance, threshold_percent)
                
        except Exception as e:
            print(f"Error generating RNN explanation: {e}")
            # Create dummy importance array with length of sequence
            importance = np.zeros(sequence_tensor.shape[1])
            simplified_importance = importance.copy()
            
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
            "importance": importance,
            "simplified_importance": simplified_importance,
            "target_class": target_class,
            "processing_time": processing_time,
            "memory_usage": self.last_memory_usage
        }
        
    def visualize_explanation(self, sequence_data, explanation_result, feature_names=None, title=None, save_path=None):
        """Visualize the temporal importance
        
        Args:
            sequence_data: Original input sequence
            explanation_result: Result from explain() method
            feature_names: Optional list of feature names
            title: Optional title for the plot
            save_path: Optional path to save the visualization
            
        Returns:
            PIL Image of the visualization
        """
        # Import libraries
        import matplotlib.pyplot as plt
        
        # Get importance maps
        importance = explanation_result["importance"]
        simplified_importance = explanation_result["simplified_importance"]
        
        # Create temporal indices
        timesteps = np.arange(len(importance))
        
        # Create figure and subplots
        plt.figure(figsize=(12, 6))
        
        # Full importance plot
        plt.subplot(2, 1, 1)
        plt.bar(timesteps, importance, color='blue', alpha=0.6)
        plt.title("Full Temporal Importance")
        plt.xlabel("Time Step")
        plt.ylabel("Importance")
        
        # Simplified importance plot
        plt.subplot(2, 1, 2)
        plt.bar(timesteps, simplified_importance, color='red', alpha=0.6)
        plt.title(f"Simplified Importance ({explanation_result.get('threshold_percent', 10)}%)")
        plt.xlabel("Time Step")
        plt.ylabel("Importance")
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        # Create a separate figure for return value
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(timesteps, importance, color='blue', alpha=0.3, label='Full Importance')
        ax.bar(timesteps, simplified_importance, color='red', alpha=0.6, label='Simplified Importance')
        ax.set_title(title if title else "RNN Temporal Explanation")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Importance")
        ax.legend()
        fig.tight_layout()
        
        # Convert matplotlib figure to PIL image
        fig.canvas.draw()
        plot_img = Image.frombytes('RGB', 
                                 fig.canvas.get_width_height(),
                                 fig.canvas.tostring_rgb())
        plt.close(fig)
        plt.close()
        
        return plot_img 