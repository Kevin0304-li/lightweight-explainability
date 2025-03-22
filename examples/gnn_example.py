import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
except ImportError:
    print("Warning: torch_geometric not found. Please install with: pip install torch-geometric")
    print("Continuing with simulated GNN data for demonstration purposes.")
    GCNConv = type('GCNConv', (), {'__init__': lambda self, *args, **kwargs: None})
    Data = type('Data', (), {'__init__': lambda self, *args, **kwargs: None})

from gnn_explainability import GNNExplainer
from universal_explainability import UniversalExplainer

# Define a simple GCN model for demonstration
class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGCN, self).__init__()
        try:
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        except Exception as e:
            print(f"Error creating GCNConv layers: {e}")
            # Create dummy layers for demonstration
            self.conv1 = nn.Linear(in_channels, hidden_channels)
            self.conv2 = nn.Linear(hidden_channels, out_channels)
            
    def forward(self, x, edge_index):
        try:
            # Use GCNConv for real GNN forward pass
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
        except Exception as e:
            print(f"Error in GNN forward pass: {e}")
            # Dummy forward pass for demonstration
            src, dst = edge_index
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
        return x

def generate_synthetic_graph(num_nodes=10, num_edges=20, num_features=3, num_classes=2):
    """Generate a synthetic graph for demonstration"""
    # Create node features
    x = torch.randn(num_nodes, num_features)
    
    # Create random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create important edges (connecting node 0 to various nodes)
    important_edges = torch.tensor([[0, 0, 0], [1, 3, 5]])
    
    # Combine regular and important edges
    if edge_index.shape[1] >= 3:
        edge_index = torch.cat([edge_index[:, :-3], important_edges], dim=1)
    
    # Create class labels for nodes
    y = torch.randint(0, num_classes, (num_nodes,))
    
    # Set target node class
    y[0] = 1
    
    try:
        # Create PyG data object
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
    except Exception as e:
        print(f"Error creating PyG Data object: {e}")
        # Return dictionary with tensors for demonstration
        return {
            "x": x,
            "edge_index": edge_index,
            "y": y
        }

def main():
    """Demonstrate GNN explainability"""
    # Create output directory
    os.makedirs('./results/gnn_examples', exist_ok=True)
    
    # Check if PyTorch Geometric is available
    try:
        import torch_geometric
        has_pyg = True
    except ImportError:
        has_pyg = False
        print("PyTorch Geometric not found. Running in demonstration mode only.")
    
    # Set up model and data
    num_nodes = 10
    num_features = 3
    hidden_channels = 16
    num_classes = 2
    
    # Initialize GNN model
    model = SimpleGCN(num_features, hidden_channels, num_classes)
    model.eval()
    
    # Generate synthetic graph
    data = generate_synthetic_graph(num_nodes, 20, num_features, num_classes)
    print(f"Generated graph with {num_nodes} nodes and {data['edge_index'].shape[1] if isinstance(data, dict) else data.edge_index.shape[1]} edges")
    
    # Skip actual explanation if PyG is not available
    if not has_pyg:
        print("Skipping GNN explanation since PyTorch Geometric is not installed.")
        print("To run the full example, install PyTorch Geometric: pip install torch-geometric")
        return
    
    # Method 1: Using specialized GNNExplainer
    print("Generating explanation with GNNExplainer...")
    explainer = GNNExplainer(model)
    target_node = 0  # Explain node 0
    explanation = explainer.explain(data, target_node=target_node, threshold_percent=20)
    
    # Visualize explanation
    vis_img = explainer.visualize_explanation(
        data, explanation,
        title=f"GNN Edge Importance (Node {target_node})",
        save_path="./results/gnn_examples/gnn_explanation_direct.png"
    )
    
    # Method 2: Using UniversalExplainer
    print("Generating explanation with UniversalExplainer...")
    universal_explainer = UniversalExplainer(model, model_type="gnn")
    universal_explanation = universal_explainer.explain(data, target_class=target_node, threshold_percent=20)
    
    # Visualize explanation
    universal_vis_img = universal_explainer.visualize_explanation(
        data, universal_explanation,
        title=f"Universal Explainer - GNN (Node {target_node})",
        save_path="./results/gnn_examples/gnn_explanation_universal.png"
    )
    
    # Run benchmark with different thresholds
    print("Running benchmark with different thresholds...")
    benchmark_results = universal_explainer.benchmark(
        data, target_class=target_node, thresholds=[10, 20, 30], num_runs=3
    )
    
    # Visualize benchmark results
    universal_explainer.visualize_benchmark(
        benchmark_results,
        save_path="./results/gnn_examples/gnn_benchmark.png"
    )
    
    # Print performance summary
    print("Performance Summary:")
    print(universal_explainer.get_performance_summary())
    
    # Define known important edges
    important_edges = [(0, 1), (0, 3), (0, 5)]
    
    # Compare with explanation
    edge_importance = explanation["edge_importance"]
    top_edges_idx = np.argsort(edge_importance)[-3:]  # Top 3 edges
    
    edge_index = data.edge_index.cpu().numpy()
    top_edges = [(edge_index[0, idx], edge_index[1, idx]) for idx in top_edges_idx]
    
    print(f"Top important edges (predicted): {top_edges}")
    print(f"True important edges: {important_edges}")
    
    # Calculate accuracy
    common = set(map(tuple, top_edges)).intersection(set(map(tuple, important_edges)))
    accuracy = len(common) / len(important_edges)
    print(f"Edge importance accuracy: {accuracy:.2f}")
    
    print("Done! Results saved in ./results/gnn_examples/")

if __name__ == "__main__":
    main() 