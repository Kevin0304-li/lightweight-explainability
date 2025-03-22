import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import gc
import networkx as nx

def simplify_edge_importance(edge_importance, edge_index, threshold_percent=10):
    """Simplify edge importance map by keeping only the top % most important edges
    
    Args:
        edge_importance: Numpy array of importance values per edge
        edge_index: Edge connectivity tensor (2 x num_edges)
        threshold_percent: Percentage threshold (1-100)
        
    Returns:
        simplified_importance: Thresholded importance map
        important_edges: Edge indices that are retained
    """
    threshold = np.percentile(edge_importance, 100 - threshold_percent)
    simplified = np.zeros_like(edge_importance)
    mask = edge_importance > threshold
    simplified[mask] = edge_importance[mask]
    
    # Get important edge indices
    important_edges = edge_index[:, mask]
    
    return simplified, important_edges

class GNNExplainer:
    """Class for explaining Graph Neural Network models"""
    
    def __init__(self, model):
        """Initialize the explainer
        
        Args:
            model: GNN model
        """
        self.model = model
        self.model.eval()
        self.edge_weights = []
        self.node_features = []
        self.hook_handles = []
        self.last_memory_usage = 0
        self.last_processing_time = 0
        
    def _register_hooks(self):
        """Register hooks to capture edge weights and node features"""
        self.edge_weights = []
        self.node_features = []
        
        # Hook function to capture edge weights
        def capture_edge_weights(module, input, output):
            # Look for edge weights in the module attributes
            if hasattr(module, "edge_weight") and module.edge_weight is not None:
                self.edge_weights.append(module.edge_weight.detach())
                
            # Handle different GNN library implementations
            if hasattr(module, "_edge_weight") and module._edge_weight is not None:
                self.edge_weights.append(module._edge_weight.detach())
                
            # Some libraries keep weights in message passing modules
            if hasattr(module, "weight") and hasattr(module, "propagate"):
                weight = module.weight.detach()
                # Convert to edge weights if needed (depends on implementation)
                if len(weight.shape) == 2:  # If it's a weight matrix
                    # This is a simplification and may vary by GNN implementation
                    self.edge_weights.append(torch.ones(input[1].shape[1]).to(weight.device))
        
        # Hook function to capture node features
        def capture_node_features(module, input, output):
            if isinstance(output, tuple):
                self.node_features.append(output[0].detach())
            else:
                self.node_features.append(output.detach())
        
        # Register hooks for GNN layers
        for name, module in self.model.named_modules():
            if any(x in name.lower() for x in ["conv", "sage", "gat", "gcn", "gin"]):
                h1 = module.register_forward_hook(capture_edge_weights)
                h2 = module.register_forward_hook(capture_node_features)
                self.hook_handles.extend([h1, h2])
                
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
    def explain(self, graph_data, target_node=None, threshold_percent=10):
        """Generate explanation for GNN model showing edge importance
        
        Args:
            graph_data: Input graph data (in PyTorch Geometric format)
            target_node: Target node to explain (None for first node)
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
        
        # Set default target node if not specified
        if target_node is None:
            target_node = 0
            
        # Register hooks
        self._register_hooks()
        
        # Forward pass
        self.model.zero_grad()
        outputs = self.model(graph_data)
        
        # Get edge index
        if hasattr(graph_data, "edge_index"):
            edge_index = graph_data.edge_index
        else:
            raise ValueError("Graph data doesn't have edge_index attribute")
            
        # Create target for backpropagation
        if len(outputs.shape) > 1:
            target = outputs[target_node].max().reshape(-1)
        else:
            target = outputs[target_node].reshape(-1)
            
        # Backward pass
        target.backward()
        
        try:
            # Calculate edge importance
            if self.edge_weights:
                # Use the captured edge weights
                edge_weights = self.edge_weights[-1].cpu().abs().numpy()
                
                # If we have node features, we can compute gradient-based edge importance
                if self.node_features and hasattr(self.node_features[-1], "grad") and self.node_features[-1].grad is not None:
                    node_grads = self.node_features[-1].grad.cpu().numpy()
                    
                    # Compute edge importance as a function of connected node gradients
                    src_nodes = edge_index[0].cpu().numpy()
                    dst_nodes = edge_index[1].cpu().numpy()
                    
                    # This is a simplified edge importance - implementations may vary
                    gradient_importance = np.abs(node_grads[src_nodes]) + np.abs(node_grads[dst_nodes])
                    gradient_importance = gradient_importance.mean(axis=1)
                    
                    # Combine with edge weights
                    if len(edge_weights) == len(gradient_importance):
                        edge_importance = edge_weights * gradient_importance
                    else:
                        edge_importance = edge_weights
                else:
                    edge_importance = edge_weights
            else:
                # Fallback: use uniform edge importance
                num_edges = edge_index.shape[1]
                edge_importance = np.ones(num_edges)
                
            # Normalize
            if edge_importance.max() > 0:
                edge_importance = edge_importance / edge_importance.max()
                
            # Apply threshold
            simplified_importance, important_edges = simplify_edge_importance(
                edge_importance, edge_index, threshold_percent)
                
        except Exception as e:
            print(f"Error generating GNN explanation: {e}")
            num_edges = edge_index.shape[1]
            edge_importance = np.ones(num_edges)
            simplified_importance = edge_importance.copy()
            important_edges = edge_index
            
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
            "edge_importance": edge_importance,
            "simplified_importance": simplified_importance,
            "important_edges": important_edges,
            "target_node": target_node,
            "edge_index": edge_index,
            "processing_time": processing_time,
            "memory_usage": self.last_memory_usage
        }
        
    def visualize_explanation(self, graph_data, explanation_result, node_labels=None, title=None, save_path=None):
        """Visualize the graph with important edges highlighted
        
        Args:
            graph_data: Original input graph
            explanation_result: Result from explain() method
            node_labels: Optional dictionary of node labels
            title: Optional title for the plot
            save_path: Optional path to save the visualization
            
        Returns:
            PIL Image of the visualization
        """
        # Import libraries
        import matplotlib.pyplot as plt
        
        # Get edge data
        edge_index = explanation_result["edge_index"].cpu().numpy()
        edge_importance = explanation_result["edge_importance"]
        simplified_importance = explanation_result["simplified_importance"]
        target_node = explanation_result["target_node"]
        
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes
        num_nodes = len(graph_data.x) if hasattr(graph_data, 'x') else edge_index.max() + 1
        for i in range(num_nodes):
            G.add_node(i)
            
        # Add edges with weights
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            G.add_edge(src, dst, weight=edge_importance[i], color='gray')
            
        # Create position layout
        pos = nx.spring_layout(G, seed=42)
        
        # Set up node colors (highlight target node)
        node_colors = ['red' if i == target_node else 'lightblue' for i in G.nodes()]
        
        # Set up edge colors based on importance
        edge_colors = ['red' if simplified_importance[i] > 0 else 'lightgray' for i in range(len(edge_importance))]
        edge_widths = [3 if simplified_importance[i] > 0 else 1 for i in range(len(edge_importance))]
        
        # Create figure and draw graph
        plt.figure(figsize=(10, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
        
        # Draw edges
        edges = list(zip(edge_index[0], edge_index[1]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=edge_widths, 
                              edge_color=edge_colors, alpha=0.7)
        
        # Draw node labels if provided
        if node_labels:
            nx.draw_networkx_labels(G, pos, labels=node_labels)
        else:
            nx.draw_networkx_labels(G, pos)
        
        plt.title(title if title else f"GNN Explanation (Target Node: {target_node})")
        plt.axis('off')
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        # Convert matplotlib figure to PIL image
        fig = plt.gcf()
        fig.canvas.draw()
        plot_img = Image.frombytes('RGB', 
                                 fig.canvas.get_width_height(),
                                 fig.canvas.tostring_rgb())
        plt.close()
        
        return plot_img 