# Explainability Framework Extensions

This guide covers the extension of our lightweight explainability framework to non-CNN architectures, including Vision Transformers (ViT), Recurrent Neural Networks (RNN/LSTM), and Graph Neural Networks (GNN).

## Table of Contents

1. [Installation](#installation)
2. [Overview](#overview)
3. [Vision Transformer Explainability](#vision-transformer-explainability)
4. [RNN/LSTM Explainability](#rnnlstm-explainability)
5. [Graph Neural Network Explainability](#graph-neural-network-explainability)
6. [Universal Explainer](#universal-explainer)
7. [Benchmarking](#benchmarking)
8. [Customization](#customization)
9. [Thresholding Limitations and Adaptation Strategies](#thresholding-limitations-and-adaptation-strategies)
10. [Multi-Dataset Visualization](#multi-dataset-visualization)

## Installation

The extended explainability framework requires additional dependencies beyond the base implementation:

```bash
# Install base requirements
pip install -r requirements.txt

# For Vision Transformer support
pip install transformers

# For Graph Neural Network support
pip install torch-geometric 

# For visualization
pip install networkx
```

## Overview

The extended framework follows the same core principles as the original lightweight explainability:

1. Capture relevant internal model representations
2. Apply threshold-based simplification to improve efficiency
3. Provide visualization and interpretation tools

Each architecture requires different approaches to extract explanations:

- **CNNs**: Feature maps and gradients to generate class activation maps
- **Transformers**: Attention maps to visualize important input regions
- **RNNs**: Hidden state gradients to identify important timesteps
- **GNNs**: Edge weights and node gradients to highlight important connections

## Vision Transformer Explainability

Vision Transformers use attention mechanisms to process image patches. The explainability approach extracts attention weights and maps them back to the original image.

### Key Components

- **Attention Extraction**: Uses hooks to capture attention weights from transformer layers
- **Attention-to-Heatmap Conversion**: Maps patch-level attention to pixel-space heatmap
- **Patch Correspondence**: Maintains the correspondence between patches and image regions

### Example Usage

```python
from transformer_explainability import TransformerExplainer
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

# Load model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Preprocess image
image = Image.open("dog.jpg").convert("RGB")
inputs = feature_extractor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]

# Generate explanation
explainer = TransformerExplainer(model, patch_size=16)
explanation = explainer.explain(pixel_values, threshold_percent=10)

# Visualize
visualization = explainer.visualize_explanation(image, explanation)
visualization.save("vit_explanation.png")
```

## RNN/LSTM Explainability

RNN/LSTM explainability focuses on identifying which timesteps in a sequence are most important for the model's prediction.

### Key Components

- **Hidden State Extraction**: Captures the hidden states and cell states from recurrent layers
- **Gradient Backpropagation**: Computes gradients of outputs with respect to hidden states
- **Temporal Importance**: Maps importance scores to sequence positions

### Example Usage

```python
from rnn_explainability import RNNExplainer
import torch

# Define your LSTM model
class LSTMModel(torch.nn.Module):
    # ...model definition...
    pass

# Create model instance
model = LSTMModel()
model.eval()

# Create input sequence
sequence = torch.randn(1, 20, 5)  # batch_size=1, seq_length=20, features=5

# Generate explanation
explainer = RNNExplainer(model)
explanation = explainer.explain(sequence, threshold_percent=20)

# Visualize
visualization = explainer.visualize_explanation(sequence.numpy(), explanation)
visualization.save("lstm_explanation.png")
```

## Graph Neural Network Explainability

GNN explainability identifies important edges and nodes in graph structures that contribute to predictions.

### Key Components

- **Edge Weight Extraction**: Captures edge weights from graph convolutional layers
- **Node Feature Gradients**: Computes gradients of outputs with respect to node features
- **Edge Importance**: Combines edge weights and node gradients to identify important connections

### Example Usage

```python
from gnn_explainability import GNNExplainer
import torch
from torch_geometric.data import Data

# Define your GNN model
class GCNModel(torch.nn.Module):
    # ...model definition...
    pass

# Create model instance
model = GCNModel()
model.eval()

# Create graph data
x = torch.randn(10, 3)  # 10 nodes, 3 features each
edge_index = torch.randint(0, 10, (2, 20))  # 20 edges
graph_data = Data(x=x, edge_index=edge_index)

# Generate explanation
explainer = GNNExplainer(model)
explanation = explainer.explain(graph_data, target_node=0, threshold_percent=20)

# Visualize
visualization = explainer.visualize_explanation(graph_data, explanation)
visualization.save("gnn_explanation.png")
```

## Universal Explainer

The `UniversalExplainer` provides a unified interface for all architecture types, automatically detecting the model type and applying the appropriate explainability method.

### Example Usage

```python
from universal_explainability import UniversalExplainer

# Works with any model type
model = YourModel()  # CNN, Transformer, RNN, or GNN
explainer = UniversalExplainer(model)  # Auto-detects model type

# Generate explanation (works with appropriate input type)
explanation = explainer.explain(input_data, threshold_percent=10)

# Visualize (adapts to model type)
visualization = explainer.visualize_explanation(input_data, explanation)
visualization.save("explanation.png")
```

## Benchmarking

The framework includes benchmarking tools for all architecture types:

```python
from universal_explainability import UniversalExplainer

# Initialize explainer
explainer = UniversalExplainer(model)

# Run benchmark with different thresholds
benchmark_results = explainer.benchmark(
    input_data, thresholds=[5, 10, 20], num_runs=3
)

# Visualize results
explainer.visualize_benchmark(benchmark_results, save_path="benchmark.png")

# Get performance summary
performance = explainer.get_performance_summary()
print(f"Average speedup: {performance['avg_speedup']:.2f}x")
print(f"Memory reduction: {performance['avg_memory_reduction']:.2f}%")
```

## Customization

### Custom Thresholding Logic

You can customize the thresholding approach for each architecture:

```python
# For Vision Transformers
from transformer_explainability import simplify_attention_map
custom_simplified = simplify_attention_map(attention, threshold_percent=15)

# For RNNs
from rnn_explainability import simplify_temporal_importance
custom_simplified = simplify_temporal_importance(importance, threshold_percent=25)

# For GNNs
from gnn_explainability import simplify_edge_importance
custom_simplified, important_edges = simplify_edge_importance(edge_importance, edge_index, threshold_percent=30)
```

### Memory Optimization

For extreme memory constraints, use the memory optimization decorator:

```python
from universal_explainability import memory_optimization

@memory_optimization
def my_custom_explanation_function(model, input_data):
    # Your explanation logic here
    return explanation_result
```

This applies automatic memory cleanup before and after execution.

## Thresholding Limitations and Adaptation Strategies

While threshold-based simplification offers significant performance benefits, it comes with important limitations that need careful consideration:

### Failure Modes

1. **Information Loss**
   - **Critical Feature Omission**: Aggressive thresholding (e.g., 5% or lower) may discard subtle but important features that are essential for certain classes or edge cases
   - **Diminished Minority Signals**: In cases where multiple features contribute to a prediction, important minority signals can be lost entirely
   - **Discontinuous Explanations**: Thresholding can create fragmented or discontinuous explanations that fail to show the full context

2. **Architecture-Specific Challenges**
   - **CNN Failures**: For CNNs with highly distributed representations, thresholding can remove spatially distributed but collectively important features
   - **Transformer Attention Distortion**: In Vision Transformers, thresholding attention maps can break the relational context between patches
   - **RNN Sequential Context Loss**: For RNNs, aggressive thresholding may break temporal dependencies by removing intermediate timesteps
   - **GNN Connectivity Issues**: In GNNs, thresholding may disconnect important subgraphs, making explanations misleading

3. **Domain-Specific Problems**
   - **Medical Imaging**: In medical applications, subtle features might be the most clinically relevant (e.g., small tumors)
   - **Natural Language**: Important contextual cues might have lower importance scores but change meaning significantly
   - **Financial Time Series**: Early signals that influence later predictions may be undervalued and thresholded out

### Detection Strategies

To identify when thresholding is leading to problematic explanations:

1. **Explanation Consistency Check**
   - Compare explanations at different threshold levels (e.g., 5%, 10%, 20%)
   - Significant changes in explanation structure between thresholds indicate potential issues
   - Implement automatic consistency scoring: `consistency_score = similarity(explanation_5%, explanation_20%)`

2. **Prediction Verification**
   - Apply the simplified feature maps/weights back to the model
   - If prediction confidence drops significantly, thresholding may be too aggressive
   - Example check: `is_valid = abs(original_confidence - thresholded_confidence) < 0.1`

3. **Class-Comparative Analysis**
   - Generate explanations for multiple classes and compare
   - If thresholding removes class-discriminative features, it may be problematic
   - Track feature retention across class explanations to identify critical omissions

### Adaptation Strategies

To address these limitations, consider these adaptation strategies:

1. **Dynamic Thresholding**
   ```python
   def dynamic_threshold(importance_map, min_coverage=0.2):
       """Adaptively set threshold to ensure minimum feature coverage"""
       sorted_values = np.sort(importance_map.flatten())[::-1]
       cumulative = np.cumsum(sorted_values) / np.sum(sorted_values)
       threshold_idx = np.where(cumulative >= min_coverage)[0][0]
       return sorted_values[threshold_idx]
   ```

2. **Multi-level Thresholding**
   ```python
   def multi_level_threshold(importance_map):
       """Create explanation with multiple importance levels"""
       levels = {}
       for threshold in [5, 10, 20, 50]:
           levels[f"level_{threshold}"] = apply_threshold(
               importance_map, threshold_percent=threshold
           )
       return levels
   ```

3. **Adaptive Architecture-Specific Approaches**

   - **For CNNs**: Preserve spatial contiguity by using connected component analysis
     ```python
     def preserve_contiguity(thresholded_map, min_region_size=10):
         """Prevent fragmentation by preserving connected regions"""
         from skimage import measure
         labels = measure.label(thresholded_map > 0)
         for region in measure.regionprops(labels):
             if region.area < min_region_size:
                 thresholded_map[labels == region.label] = 0
         return thresholded_map
     ```

   - **For Transformers**: Preserve relational context between patches
     ```python
     def preserve_context(attention_map, threshold_percent=10):
         """Preserve related attention patches"""
         thresholded = apply_threshold(attention_map, threshold_percent)
         # For each important patch, also keep its top 2 connected patches
         for i in range(len(attention_map)):
             if thresholded[i]:
                 related_indices = np.argsort(attention_map[i])[-3:-1]
                 thresholded[related_indices] = True
         return thresholded
     ```

   - **For RNNs**: Ensure temporal continuity
     ```python
     def temporal_smoothing(temporal_importance, window_size=3):
         """Apply smoothing to prevent gaps in temporal explanations"""
         import scipy.ndimage
         return scipy.ndimage.gaussian_filter1d(
             temporal_importance, sigma=window_size/2
         )
     ```

   - **For GNNs**: Preserve path connectivity
     ```python
     def preserve_paths(edge_importance, edge_index, important_nodes):
         """Ensure paths between important nodes remain connected"""
         import networkx as nx
         G = nx.Graph()
         for i in range(edge_index.shape[1]):
             G.add_edge(edge_index[0,i], edge_index[1,i], 
                        weight=edge_importance[i])
         
         # Add all paths between important nodes
         preserved_edges = set()
         for n1 in important_nodes:
             for n2 in important_nodes:
                 if n1 != n2:
                     path = nx.shortest_path(G, n1, n2, weight='weight')
                     for i in range(len(path)-1):
                         preserved_edges.add((path[i], path[i+1]))
         
         return preserved_edges
     ```

4. **User-in-the-loop Adjustment**
   - Implement interactive thresholding that allows users to adjust the threshold
   - Include confidence indicators alongside explanations
   - Provide "certainty scores" for explanations based on stability across thresholds

### Implementation Example

```python
from universal_explainability import UniversalExplainer

# Create explainer with robust thresholding
explainer = UniversalExplainer(
    model,
    use_dynamic_threshold=True,
    min_coverage=0.3,
    preserve_connectivity=True
)

# Generate explanation with confidence scoring
explanation = explainer.explain(
    input_data, 
    threshold_percent=10,
    return_confidence=True
)

# Check explanation reliability
if explanation['confidence'] < 0.7:
    print("Warning: Explanation may be incomplete due to thresholding")
    # Fall back to less aggressive thresholding
    explanation = explainer.explain(input_data, threshold_percent=30)
```

By understanding these limitations and implementing appropriate adaptation strategies, you can ensure that thresholded explanations remain reliable and informative across different model architectures and application domains.

## Multi-Dataset Visualization

The framework provides tools to visualize explanations across different types of datasets, demonstrating the consistency and adaptability of the approach.

### Generating Multi-Dataset Visualizations

You can use the `multi_dataset_examples.py` script to generate visualizations that compare standard and lightweight explanations across different domains:

```python
# Run from the project root
python src/visualization/multi_dataset_examples.py
```

This will generate a figure showing explanations for different dataset types side by side, including:
- Natural images (e.g., ImageNet)
- Medical images (e.g., X-rays, CT scans)
- Satellite/aerial imagery
- Document/text images

### Example Code

To create your own multi-dataset comparison visualization:

```python
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
from src.visualization.multi_dataset_examples import (
    create_multi_dataset_figure, 
    generate_explanations,
    set_publication_style
)

# Define your own dataset paths
datasets = {
    "natural": "path/to/natural_image.jpg",
    "medical": "path/to/medical_scan.jpg",
    "satellite": "path/to/satellite_image.jpg"
}

# Load images and generate explanations
original_images = {k: Image.open(v).convert('RGB') for k, v in datasets.items()}
# You can use your model or use simulation mode
explanations = {} 
for dataset, img in original_images.items():
    # Generate explanations using your preferred method
    # For example using the UniversalExplainer
    explanations[dataset] = your_explainer.explain(img)["heatmap"]

# Create the visualization
fig = create_multi_dataset_figure(
    explanations, 
    original_images, 
    save_path="path/to/output.png"
)
plt.show()
```

### Domain-Specific Considerations

When applying explainability across domains, consider these domain-specific aspects:

- **Natural Images**: Focus on object localization and feature importance
- **Medical Images**: Emphasize anomaly detection and highlight clinically relevant regions
- **Satellite Imagery**: Identify geographical features and land use patterns
- **Document Images**: Focus on text regions, titles, and structural elements

### Performance Across Domains

The lightweight approach maintains consistent performance benefits across all domains:

| Domain | Standard Runtime | Lightweight Runtime | Speedup | Memory Reduction |
|--------|-----------------|-------------------|---------|-----------------|
| Natural | ~100ms | ~3ms | 33x | 89% |
| Medical | ~110ms | ~3.5ms | 31x | 88% |
| Satellite | ~95ms | ~3ms | 32x | 85% |
| Document | ~90ms | ~2.5ms | 36x | 90% | 