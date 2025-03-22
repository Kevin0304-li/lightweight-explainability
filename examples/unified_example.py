import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms

from universal_explainability import UniversalExplainer, get_explainer_for_model
from lightweight_explainability import ExplainableModel

def demonstrate_cnn_explainability():
    """Demonstrate CNN explainability using the original framework"""
    print("\n===== CNN Explainability Demonstration =====")
    
    # Create output directory
    os.makedirs('./results/unified_examples/cnn', exist_ok=True)
    
    # Load a pre-trained CNN model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    
    # Load sample image
    img_path = "../sample_images/dog.jpg"
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        return
    
    # Use the original ExplainableModel
    explainer = ExplainableModel(model_name='mobilenet_v2')
    
    # Preprocess image
    img_tensor, img = explainer.preprocess_image(img_path)
    
    # Generate explanation with the lightweight approach
    print("Generating CNN explanation...")
    heatmap, processing_time = explainer.simplified_gradcam(img_tensor, threshold_pct=10)
    
    # Visualize explanation
    overlay = explainer.visualize_explanation(
        img, heatmap,
        title="MobileNetV2 Lightweight Explanation",
        save_path="./results/unified_examples/cnn/mobilenet_explanation.png"
    )
    
    # Also use the UniversalExplainer
    universal_explainer = UniversalExplainer(model, model_type="cnn")
    universal_explanation = universal_explainer.explain(img_tensor, threshold_percent=10)
    
    # Print performance metrics
    print(f"Original Framework Processing Time: {processing_time:.5f} seconds")
    print(f"Universal Framework Processing Time: {universal_explanation['processing_time']:.5f} seconds")
    print(f"Memory Usage: {universal_explanation['memory_usage'] / (1024 * 1024):.2f} MB")
    
    return explainer, img_tensor, img

def demonstrate_transformer_explainability():
    """Demonstrate transformer explainability"""
    print("\n===== Transformer Explainability Demonstration =====")
    
    # Skip if transformers library not available
    try:
        from transformers import ViTForImageClassification, ViTFeatureExtractor
    except ImportError:
        print("Transformers library not found. Skipping ViT demonstration.")
        print("To run this example, install transformers: pip install transformers")
        return None, None, None
    
    # Create output directory
    os.makedirs('./results/unified_examples/transformer', exist_ok=True)
    
    # Load ViT model and feature extractor
    try:
        model_name = "google/vit-base-patch16-224"
        model = ViTForImageClassification.from_pretrained(model_name)
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading ViT model: {e}")
        return None, None, None
    
    model.eval()
    
    # Load sample image
    img_path = "../sample_images/dog.jpg"
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        return None, None, None
    
    # Process image
    image = Image.open(img_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    
    # Use the UniversalExplainer
    print("Generating Transformer explanation...")
    universal_explainer = UniversalExplainer(model, model_type="transformer", patch_size=16)
    explanation = universal_explainer.explain(pixel_values, threshold_percent=10)
    
    # Visualize explanation
    vis_img = universal_explainer.visualize_explanation(
        image, explanation,
        title="ViT Explanation",
        save_path="./results/unified_examples/transformer/vit_explanation.png"
    )
    
    # Print performance metrics
    print(f"Processing Time: {explanation['processing_time']:.5f} seconds")
    print(f"Memory Usage: {explanation['memory_usage'] / (1024 * 1024):.2f} MB")
    
    return universal_explainer, pixel_values, image

def demonstrate_rnn_explainability():
    """Demonstrate RNN explainability"""
    print("\n===== RNN Explainability Demonstration =====")
    
    # Create output directory
    os.makedirs('./results/unified_examples/rnn', exist_ok=True)
    
    # Create a simple LSTM model
    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(SimpleLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
            
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
    
    # Initialize model
    input_size = 5
    hidden_size = 20
    num_layers = 2
    num_classes = 2
    
    model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)
    model.eval()
    
    # Generate synthetic sequence data
    seq_length = 20
    batch_size = 1
    X = torch.randn(batch_size, seq_length, input_size)
    
    # Use the UniversalExplainer
    print("Generating RNN explanation...")
    universal_explainer = UniversalExplainer(model, model_type="rnn")
    explanation = universal_explainer.explain(X, threshold_percent=20)
    
    # Visualize explanation
    vis_img = universal_explainer.visualize_explanation(
        X.numpy(), explanation,
        title="LSTM Temporal Importance",
        save_path="./results/unified_examples/rnn/lstm_explanation.png"
    )
    
    # Print performance metrics
    print(f"Processing Time: {explanation['processing_time']:.5f} seconds")
    print(f"Memory Usage: {explanation['memory_usage'] / (1024 * 1024):.2f} MB")
    
    return universal_explainer, X, None

def demonstrate_gnn_explainability():
    """Demonstrate GNN explainability"""
    print("\n===== GNN Explainability Demonstration =====")
    
    # Skip if PyTorch Geometric not available
    try:
        from torch_geometric.nn import GCNConv
        from torch_geometric.data import Data
    except ImportError:
        print("PyTorch Geometric library not found. Skipping GNN demonstration.")
        print("To run this example, install PyTorch Geometric: pip install torch-geometric")
        return None, None, None
    
    # Create output directory
    os.makedirs('./results/unified_examples/gnn', exist_ok=True)
    
    # Create a simple GCN model
    class SimpleGCN(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(SimpleGCN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            return x
    
    # Initialize model
    num_features = 3
    hidden_channels = 16
    num_classes = 2
    
    model = SimpleGCN(num_features, hidden_channels, num_classes)
    model.eval()
    
    # Generate synthetic graph
    num_nodes = 10
    num_edges = 20
    
    # Node features
    x = torch.randn(num_nodes, num_features)
    
    # Create random edges
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Create important edges (connecting node 0 to various nodes)
    important_edges = torch.tensor([[0, 0, 0], [1, 3, 5]])
    
    # Combine regular and important edges
    edge_index = torch.cat([edge_index[:, :-3], important_edges], dim=1)
    
    # Create class labels for nodes
    y = torch.randint(0, num_classes, (num_nodes,))
    
    # Set target node class
    y[0] = 1
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Use the UniversalExplainer
    print("Generating GNN explanation...")
    universal_explainer = UniversalExplainer(model, model_type="gnn")
    explanation = universal_explainer.explain(data, target_class=0, threshold_percent=20)
    
    # Visualize explanation
    vis_img = universal_explainer.visualize_explanation(
        data, explanation,
        title="GNN Edge Importance",
        save_path="./results/unified_examples/gnn/gnn_explanation.png"
    )
    
    # Print performance metrics
    print(f"Processing Time: {explanation['processing_time']:.5f} seconds")
    print(f"Memory Usage: {explanation['memory_usage'] / (1024 * 1024):.2f} MB")
    
    return universal_explainer, data, None

def compare_all_explainers():
    """Compare all explainers in terms of performance"""
    print("\n===== Performance Comparison =====")
    
    # Create output directory
    os.makedirs('./results/unified_examples', exist_ok=True)
    
    # Run all demonstrations
    explainers = []
    
    print("Running CNN explainability...")
    cnn_explainer, cnn_input, _ = demonstrate_cnn_explainability()
    if cnn_explainer:
        explainers.append(("CNN", cnn_explainer, cnn_input))
    
    print("Running Transformer explainability...")
    transformer_explainer, transformer_input, _ = demonstrate_transformer_explainability()
    if transformer_explainer:
        explainers.append(("Transformer", transformer_explainer, transformer_input))
    
    print("Running RNN explainability...")
    rnn_explainer, rnn_input, _ = demonstrate_rnn_explainability()
    if rnn_explainer:
        explainers.append(("RNN", rnn_explainer, rnn_input))
    
    print("Running GNN explainability...")
    gnn_explainer, gnn_input, _ = demonstrate_gnn_explainability()
    if gnn_explainer:
        explainers.append(("GNN", gnn_explainer, gnn_input))
    
    # Collect benchmark results
    model_types = []
    speedups = []
    memory_reductions = []
    
    for model_type, explainer, input_data in explainers:
        print(f"\nBenchmarking {model_type} explainer...")
        
        # Run benchmark
        benchmark_results = explainer.benchmark(
            input_data, thresholds=[5, 10, 20], num_runs=3
        )
        
        # Store results
        model_types.append(model_type)
        
        # Average speedup across thresholds
        avg_speedup = np.mean(benchmark_results.get('speedups', [1.0]))
        speedups.append(avg_speedup)
        
        # Average memory reduction across thresholds
        avg_memory_reduction = np.mean(benchmark_results.get('memory_reductions', [0.0]))
        memory_reductions.append(avg_memory_reduction)
        
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"Average Memory Reduction: {avg_memory_reduction:.2f}%")
    
    # Create comparison chart
    plt.figure(figsize=(12, 6))
    
    # First subplot: Speedup
    plt.subplot(1, 2, 1)
    bars = plt.bar(model_types, speedups)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}x', ha='center', va='bottom')
    plt.title('Speedup by Model Type')
    plt.ylabel('Speedup Factor (x times)')
    plt.ylim(0, max(speedups) * 1.2)
    
    # Second subplot: Memory Reduction
    plt.subplot(1, 2, 2)
    bars = plt.bar(model_types, memory_reductions)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    plt.title('Memory Reduction by Model Type')
    plt.ylabel('Memory Reduction (%)')
    plt.ylim(0, max(memory_reductions) * 1.2)
    
    plt.tight_layout()
    plt.savefig('./results/unified_examples/performance_comparison.png')
    plt.close()
    
    print("\nPerformance comparison saved to './results/unified_examples/performance_comparison.png'")

def main():
    """Main function to demonstrate all explainability approaches"""
    print("Universal Explainability Framework Demonstration")
    print("================================================")
    print("This script demonstrates the extended explainability framework")
    print("for multiple neural network architectures.")
    
    # Create main output directory
    os.makedirs('./results/unified_examples', exist_ok=True)
    
    # Compare all explainers
    compare_all_explainers()
    
    print("\nDone! All examples have been executed and results saved.")

if __name__ == "__main__":
    main() 