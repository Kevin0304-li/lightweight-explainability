import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from PIL import Image
from scipy.ndimage import gaussian_filter

# Try importing model-specific explainers
try:
    from src.lightweight_explainer import LightweightExplainer
    cnn_available = True
except ImportError:
    print("CNNExplainer not available. CNN explanation will not be supported.")
    cnn_available = False

try:
    from src.transformer_explainability import TransformerExplainer
    transformer_available = True
except ImportError:
    print("TransformerExplainer not available. Transformer explanation will not be supported.")
    transformer_available = False

try:
    from src.rnn_explainability import RNNExplainer
    rnn_available = True
except ImportError:
    print("RNNExplainer not available. RNN explanation will not be supported.")
    rnn_available = False

try:
    from src.gnn_explainability import GNNExplainer
    gnn_available = True
except ImportError:
    print("GNNExplainer not available. GNN explanation will not be supported.")
    gnn_available = False

def memory_optimization(func):
    """Decorator to optimize memory usage during explanation generation."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Force garbage collection before and after
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return wrapper

class UniversalExplainer:
    """Universal explainability framework that works with any model architecture."""
    
    def __init__(self, model, **kwargs):
        """Initialize the universal explainer.
        
        Args:
            model: The model to explain (CNN, Transformer, RNN, or GNN)
            **kwargs: Additional architecture-specific parameters
        """
        self.model = model
        self.model_type = self._detect_model_type(model)
        self.explainer = self._initialize_explainer(model, **kwargs)
        
        # Advanced thresholding options
        self.use_dynamic_threshold = kwargs.get('use_dynamic_threshold', False)
        self.min_coverage = kwargs.get('min_coverage', 0.2)
        self.preserve_connectivity = kwargs.get('preserve_connectivity', False)
        
        # Performance metrics
        self.metrics = {
            'time': [],
            'memory': [],
            'confidence': []
        }
    
    def _detect_model_type(self, model):
        """Detect the type of model (CNN, Transformer, RNN, or GNN)."""
        # Check if model is a Transformer
        if transformer_available and hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            return 'transformer'
        
        # Check if model is an RNN/LSTM
        if rnn_available and any(isinstance(module, (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU)) 
                              for module in model.modules()):
            return 'rnn'
        
        # Check if model is a GNN
        if gnn_available and hasattr(model, 'conv') and 'geometric' in str(type(model)):
            return 'gnn'
        
        # Default to CNN if no specific type is detected
        if cnn_available:
            return 'cnn'
        
        raise ValueError("Could not detect model type or appropriate explainer not available")
    
    def _initialize_explainer(self, model, **kwargs):
        """Initialize the appropriate explainer based on model type."""
        if self.model_type == 'transformer' and transformer_available:
            return TransformerExplainer(model, **kwargs)
        elif self.model_type == 'rnn' and rnn_available:
            return RNNExplainer(model, **kwargs)
        elif self.model_type == 'gnn' and gnn_available:
            return GNNExplainer(model, **kwargs)
        elif self.model_type == 'cnn' and cnn_available:
            return LightweightExplainer(model, **kwargs)
        else:
            raise ValueError(f"No explainer available for model type: {self.model_type}")
    
    def _calculate_dynamic_threshold(self, importance_map, min_coverage=0.2):
        """Adaptively set threshold to ensure minimum feature coverage."""
        flat_values = importance_map.flatten()
        sorted_values = np.sort(flat_values)[::-1]
        cumulative = np.cumsum(sorted_values) / np.sum(sorted_values)
        
        # Find threshold that ensures the min_coverage
        threshold_idx = np.where(cumulative >= min_coverage)[0]
        if len(threshold_idx) == 0:
            return 0  # No threshold needed
        
        threshold_idx = threshold_idx[0]
        threshold_value = sorted_values[threshold_idx]
        
        # Calculate equivalent percent threshold
        total_values = len(flat_values)
        percent_threshold = (threshold_idx / total_values) * 100
        
        return threshold_value, percent_threshold
    
    def _apply_connectivity_preservation(self, explanation):
        """Apply architecture-specific connectivity preservation."""
        if self.model_type == 'cnn':
            return self._preserve_spatial_contiguity(explanation)
        elif self.model_type == 'transformer':
            return self._preserve_attention_context(explanation)
        elif self.model_type == 'rnn':
            return self._preserve_temporal_continuity(explanation)
        elif self.model_type == 'gnn':
            return self._preserve_graph_connectivity(explanation)
        return explanation
    
    def _preserve_spatial_contiguity(self, explanation):
        """Preserve spatial contiguity in CNN explanations."""
        if 'heatmap' not in explanation:
            return explanation
        
        try:
            from skimage import measure
            
            heatmap = explanation['heatmap']
            binary_map = heatmap > 0
            labels = measure.label(binary_map)
            
            # Keep only regions above minimum size
            min_region_size = max(1, int(0.01 * heatmap.size))
            for region in measure.regionprops(labels):
                if region.area < min_region_size:
                    heatmap[labels == region.label] = 0
            
            # Optional: apply slight smoothing to reduce fragmentation
            heatmap = gaussian_filter(heatmap, sigma=0.5)
            
            explanation['heatmap'] = heatmap
            explanation['contiguity_preserved'] = True
            
        except ImportError:
            # If skimage not available, skip this step
            explanation['contiguity_preserved'] = False
        
        return explanation
    
    def _preserve_attention_context(self, explanation):
        """Preserve relational context in transformer attention maps."""
        if 'attention_map' not in explanation:
            return explanation
        
        attention_map = explanation['attention_map']
        thresholded_map = explanation.get('thresholded_map', attention_map > 0)
        
        # For each important patch, also include its top connected patches
        for i in range(len(attention_map)):
            if thresholded_map[i].any():
                # Find top related patches for this important patch
                for j in range(len(attention_map[i])):
                    if thresholded_map[i][j]:
                        related_indices = np.argsort(attention_map[i])[-3:]
                        for idx in related_indices:
                            thresholded_map[i][idx] = True
        
        explanation['thresholded_map'] = thresholded_map
        explanation['context_preserved'] = True
        
        return explanation
    
    def _preserve_temporal_continuity(self, explanation):
        """Ensure temporal continuity in RNN explanations."""
        if 'temporal_importance' not in explanation:
            return explanation
        
        importance = explanation['temporal_importance']
        
        # Apply smoothing to ensure temporal continuity
        smoothed = gaussian_filter1d(importance, sigma=1.0)
        
        # Make sure we preserve the overall magnitude
        scale_factor = np.sum(importance) / np.sum(smoothed) if np.sum(smoothed) > 0 else 1.0
        smoothed = smoothed * scale_factor
        
        explanation['temporal_importance'] = smoothed
        explanation['temporal_continuity_preserved'] = True
        
        return explanation
    
    def _preserve_graph_connectivity(self, explanation):
        """Preserve path connectivity between important nodes in GNN explanations."""
        if 'edge_importance' not in explanation or 'edge_index' not in explanation:
            return explanation
        
        edge_importance = explanation['edge_importance']
        edge_index = explanation['edge_index']
        
        if 'important_nodes' in explanation:
            important_nodes = explanation['important_nodes']
            
            try:
                import networkx as nx
                
                # Create a graph from the edges
                G = nx.Graph()
                for i in range(edge_index.shape[1]):
                    src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                    G.add_edge(src, dst, weight=edge_importance[i].item())
                
                # Find paths between important nodes
                preserved_edges = set()
                for n1 in important_nodes:
                    for n2 in important_nodes:
                        if n1 != n2 and n1 in G and n2 in G:
                            try:
                                path = nx.shortest_path(G, n1, n2, weight='weight')
                                for i in range(len(path)-1):
                                    preserved_edges.add((path[i], path[i+1]))
                                    preserved_edges.add((path[i+1], path[i]))  # Undirected
                            except nx.NetworkXNoPath:
                                continue
                
                # Mark these edges as important
                new_edge_importance = edge_importance.clone()
                for i in range(edge_index.shape[1]):
                    e = (edge_index[0, i].item(), edge_index[1, i].item())
                    if e in preserved_edges:
                        new_edge_importance[i] = max(new_edge_importance[i], 0.5)  # Ensure visibility
                
                explanation['edge_importance'] = new_edge_importance
                explanation['connectivity_preserved'] = True
                
            except ImportError:
                explanation['connectivity_preserved'] = False
        
        return explanation
    
    def _calculate_explanation_confidence(self, explanation, original_prediction=None):
        """Calculate confidence score for the explanation."""
        # Architecture-specific confidence calculation
        if self.model_type == 'cnn' and 'heatmap' in explanation:
            # For CNNs: check concentrated or dispersed attention
            heatmap = explanation['heatmap']
            total = np.sum(heatmap)
            if total == 0:
                return 0.0
                
            # Calculate concentration metrics
            activated_pixels = np.sum(heatmap > 0)
            total_pixels = heatmap.size
            concentration_ratio = activated_pixels / total_pixels
            
            # Highly concentrated (very small ratio) or highly dispersed (large ratio) 
            # explanations are potentially less reliable
            if concentration_ratio < 0.01:  # Too concentrated
                return 0.5
            elif concentration_ratio > 0.5:  # Too dispersed
                return 0.6
            else:
                return 0.9
                
        elif self.model_type == 'transformer' and 'attention_map' in explanation:
            # For transformers: check attention distribution
            attention = explanation['attention_map']
            entropy = -np.sum(attention * np.log(attention + 1e-10))
            max_entropy = -np.log(1/len(attention))
            normalized_entropy = entropy / max_entropy
            
            # Lower entropy means more confident attention
            return 1.0 - normalized_entropy
            
        elif self.model_type == 'rnn' and 'temporal_importance' in explanation:
            # For RNNs: check temporal distribution
            importance = explanation['temporal_importance']
            # If the importance is very concentrated on specific timesteps, more confident
            top_importance_ratio = np.sum(np.sort(importance)[-3:]) / np.sum(importance)
            return min(0.9, top_importance_ratio)
            
        elif self.model_type == 'gnn' and 'edge_importance' in explanation:
            # For GNNs: check connectivity preservation
            if explanation.get('connectivity_preserved', False):
                return 0.85
            
            edge_importance = explanation['edge_importance']
            # If some edges are clearly more important
            sorted_imp = np.sort(edge_importance)
            if len(sorted_imp) > 0:
                top_vs_avg = sorted_imp[-1] / max(np.mean(sorted_imp), 1e-10)
                return min(0.9, 0.5 + (0.4 * min(1.0, np.log10(top_vs_avg) / 2)))
        
        # Default confidence
        return 0.7
    
    @memory_optimization
    def explain(self, input_data, target_class=None, threshold_percent=10, return_confidence=False):
        """Generate explanation for the model's prediction.
        
        Args:
            input_data: Model input (image, sequence, or graph)
            target_class: Target class index to explain (default: highest predicted class)
            threshold_percent: Percentage of features to keep (0-100)
            return_confidence: Whether to return confidence score for the explanation
            
        Returns:
            Dictionary containing explanation data
        """
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Get original prediction for verification
        original_prediction = None
        if return_confidence and hasattr(self.model, 'forward'):
            with torch.no_grad():
                if isinstance(input_data, torch.Tensor):
                    original_prediction = self.model(input_data)
        
        # Apply dynamic thresholding if enabled
        effective_threshold = threshold_percent
        if self.use_dynamic_threshold:
            # Temporarily set threshold to a low value to get raw importance
            raw_explanation = self.explainer.explain(input_data, target_class, threshold_percent=1)
            
            # Extract importance map based on model type
            importance_map = None
            if self.model_type == 'cnn' and 'raw_heatmap' in raw_explanation:
                importance_map = raw_explanation['raw_heatmap']
            elif self.model_type == 'transformer' and 'attention_map' in raw_explanation:
                importance_map = raw_explanation['attention_map']
            elif self.model_type == 'rnn' and 'temporal_importance' in raw_explanation:
                importance_map = raw_explanation['temporal_importance']
            elif self.model_type == 'gnn' and 'edge_importance' in raw_explanation:
                importance_map = raw_explanation['edge_importance']
                
            if importance_map is not None:
                # Calculate dynamic threshold
                _, effective_threshold = self._calculate_dynamic_threshold(
                    importance_map.detach().cpu().numpy() if isinstance(importance_map, torch.Tensor) else importance_map,
                    min_coverage=self.min_coverage
                )
        
        # Generate explanation with the effective threshold
        explanation = self.explainer.explain(input_data, target_class, threshold_percent=effective_threshold)
        
        # Apply connectivity preservation if enabled
        if self.preserve_connectivity:
            explanation = self._apply_connectivity_preservation(explanation)
        
        # Add metadata
        explanation['model_type'] = self.model_type
        explanation['threshold_percent'] = effective_threshold
        explanation['applied_threshold'] = effective_threshold if not self.use_dynamic_threshold else 'dynamic'
        
        # Calculate and add confidence if requested
        if return_confidence:
            confidence = self._calculate_explanation_confidence(explanation, original_prediction)
            explanation['confidence'] = confidence
            self.metrics['confidence'].append(confidence)
        
        # Record performance metrics
        elapsed_time = time.time() - start_time
        memory_used = (torch.cuda.memory_allocated() - start_memory) if torch.cuda.is_available() else 0
        
        self.metrics['time'].append(elapsed_time)
        self.metrics['memory'].append(memory_used)
        
        explanation['performance'] = {
            'time': elapsed_time,
            'memory': memory_used
        }
        
        return explanation
    
    def visualize_explanation(self, input_data, explanation, save_path=None):
        """Visualize explanation according to model type."""
        # Delegate to the appropriate explainer
        visualization = self.explainer.visualize_explanation(input_data, explanation)
        
        # Add confidence indicator if available
        if 'confidence' in explanation:
            confidence = explanation['confidence']
            plt.figure(figsize=(10, 1))
            plt.barh(0, confidence, color='green' if confidence >= 0.7 else 'orange' if confidence >= 0.5 else 'red')
            plt.xlim(0, 1)
            plt.title(f"Explanation Confidence: {confidence:.2f}")
            plt.tight_layout()
            
            if save_path:
                confidence_path = save_path.replace('.png', '_confidence.png')
                plt.savefig(confidence_path)
                plt.close()
        
        if save_path and visualization is not None:
            if isinstance(visualization, plt.Figure):
                visualization.savefig(save_path)
                plt.close(visualization)
            elif isinstance(visualization, Image.Image):
                visualization.save(save_path)
        
        return visualization
    
    def benchmark(self, input_data, thresholds=[5, 10, 20], num_runs=3):
        """Benchmark explainability with different thresholds.
        
        Args:
            input_data: Input to explain
            thresholds: List of threshold percentages to test
            num_runs: Number of runs per threshold for averaging
            
        Returns:
            Dictionary of benchmark results
        """
        results = {
            'thresholds': thresholds,
            'times': [],
            'memory_usage': [],
            'confidence': []
        }
        
        for threshold in thresholds:
            threshold_times = []
            threshold_memory = []
            threshold_confidence = []
            
            for _ in range(num_runs):
                explanation = self.explain(input_data, threshold_percent=threshold, return_confidence=True)
                
                threshold_times.append(explanation['performance']['time'])
                threshold_memory.append(explanation['performance']['memory'])
                threshold_confidence.append(explanation.get('confidence', 0.0))
            
            results['times'].append(np.mean(threshold_times))
            results['memory_usage'].append(np.mean(threshold_memory))
            results['confidence'].append(np.mean(threshold_confidence))
        
        return results
    
    def visualize_benchmark(self, benchmark_results, save_path=None):
        """Visualize benchmark results."""
        thresholds = benchmark_results['thresholds']
        times = benchmark_results['times']
        memory = benchmark_results['memory_usage']
        confidence = benchmark_results['confidence']
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Processing time
        ax1.plot(thresholds, times, 'o-', color='blue')
        ax1.set_title('Processing Time vs Threshold')
        ax1.set_xlabel('Threshold (%)')
        ax1.set_ylabel('Time (seconds)')
        
        # Memory usage
        ax2.plot(thresholds, [m/1024/1024 for m in memory], 'o-', color='orange')
        ax2.set_title('Memory Usage vs Threshold')
        ax2.set_xlabel('Threshold (%)')
        ax2.set_ylabel('Memory (MB)')
        
        # Confidence
        ax3.plot(thresholds, confidence, 'o-', color='green')
        ax3.set_title('Explanation Confidence vs Threshold')
        ax3.set_xlabel('Threshold (%)')
        ax3.set_ylabel('Confidence Score')
        ax3.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        
        return fig
    
    def get_performance_summary(self):
        """Get performance summary statistics."""
        if not self.metrics['time']:
            return {"error": "No performance data available. Run explain() first."}
        
        time_avg = np.mean(self.metrics['time'])
        memory_avg = np.mean(self.metrics['memory']) / (1024 * 1024)  # Convert to MB
        
        # If we have a baseline (traditional explainer) performance to compare against
        baseline_time = getattr(self.explainer, 'baseline_time', None)
        baseline_memory = getattr(self.explainer, 'baseline_memory', None)
        
        avg_speedup = baseline_time / time_avg if baseline_time else "N/A"
        memory_reduction = 100 * (1 - memory_avg / baseline_memory) if baseline_memory else "N/A"
        
        if self.metrics['confidence']:
            avg_confidence = np.mean(self.metrics['confidence'])
        else:
            avg_confidence = "N/A"
        
        return {
            "avg_processing_time": time_avg,
            "avg_memory_usage_mb": memory_avg,
            "avg_speedup": avg_speedup,
            "avg_memory_reduction": memory_reduction,
            "avg_confidence": avg_confidence,
            "model_type": self.model_type
        }

def multi_level_threshold(importance_map, thresholds=[5, 10, 20, 50]):
    """Create explanation with multiple importance levels.
    
    Args:
        importance_map: Raw importance map to threshold
        thresholds: List of threshold percentages
        
    Returns:
        Dictionary of thresholded maps at different levels
    """
    levels = {}
    
    for threshold in thresholds:
        # Sort values in descending order
        sorted_values = np.sort(importance_map.flatten())[::-1]
        
        # Find threshold value
        threshold_idx = int(len(sorted_values) * threshold / 100)
        if threshold_idx < len(sorted_values):
            threshold_value = sorted_values[threshold_idx]
        else:
            threshold_value = sorted_values[-1]
        
        # Apply threshold
        thresholded = importance_map.copy()
        thresholded[thresholded < threshold_value] = 0
        
        levels[f"level_{threshold}"] = thresholded
    
    return levels

def simplify_importance_map(importance_map, threshold_percent=10, ensure_coverage=False, min_coverage=0.2):
    """Generic importance map simplification with adaptive options.
    
    Args:
        importance_map: Raw importance map (numpy array)
        threshold_percent: Percentage of values to keep
        ensure_coverage: Whether to ensure minimum coverage
        min_coverage: Minimum coverage to ensure
        
    Returns:
        Simplified importance map
    """
    # Flatten and sort values
    flat_values = importance_map.flatten()
    sorted_values = np.sort(flat_values)[::-1]
    
    # Calculate threshold value
    threshold_idx = int(len(sorted_values) * threshold_percent / 100)
    if threshold_idx < len(sorted_values):
        threshold_value = sorted_values[threshold_idx]
    else:
        threshold_value = sorted_values[-1] if len(sorted_values) > 0 else 0
    
    # Apply threshold
    simplified = importance_map.copy()
    simplified[simplified < threshold_value] = 0
    
    # Ensure minimum coverage if needed
    if ensure_coverage:
        coverage = np.sum(simplified) / np.sum(importance_map) if np.sum(importance_map) > 0 else 0
        
        # If coverage is too low, adjust threshold
        if coverage < min_coverage:
            # Find new threshold that ensures min_coverage
            cumulative = np.cumsum(sorted_values) / np.sum(sorted_values)
            new_threshold_idx = np.where(cumulative >= min_coverage)[0]
            
            if len(new_threshold_idx) > 0:
                new_threshold_value = sorted_values[new_threshold_idx[0]]
                simplified = importance_map.copy()
                simplified[simplified < new_threshold_value] = 0
    
    return simplified 