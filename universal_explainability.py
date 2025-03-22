import os
import time
import torch
import numpy as np
import gc
import functools
from PIL import Image
import matplotlib.pyplot as plt

# Import specific explainers
try:
    from transformer_explainability import TransformerExplainer
    from rnn_explainability import RNNExplainer
    from gnn_explainability import GNNExplainer
    from lightweight_explainability import ExplainableModel
except ImportError:
    print("Warning: Some explainer modules couldn't be imported")

def memory_optimization(func):
    """Decorator to optimize memory usage for explanation functions"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Clear memory before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Track peak memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
            
        # Call original function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Calculate memory usage
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated()
            memory_usage = peak_mem - start_mem
        else:
            memory_usage = 0
            
        # Add memory and time metrics to result
        if isinstance(result, dict):
            result["memory_usage"] = memory_usage
            result["processing_time"] = end_time - start_time
            return result
        else:
            return {
                "explanation": result, 
                "memory_usage": memory_usage,
                "processing_time": end_time - start_time
            }
        
    return wrapper

class UniversalExplainer:
    """Unified explainability framework supporting multiple neural architectures"""
    
    def __init__(self, model, model_type=None, **kwargs):
        """Initialize explainer with a model
        
        Args:
            model: Neural network model
            model_type: One of "cnn", "transformer", "rnn", "gnn", or None (auto-detect)
            **kwargs: Additional parameters for specific explainers
        """
        self.model = model
        
        # Auto-detect model type if not specified
        if model_type is None:
            self.model_type = self._detect_model_type()
        else:
            self.model_type = model_type
            
        # Initialize specific explainer based on model type
        if self.model_type == "cnn":
            self.explainer = ExplainableModel(model=model, **kwargs)
        elif self.model_type == "transformer":
            self.explainer = TransformerExplainer(model, **kwargs)
        elif self.model_type == "rnn":
            self.explainer = RNNExplainer(model)
        elif self.model_type == "gnn":
            self.explainer = GNNExplainer(model)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        # Performance tracking
        self.performance_stats = {
            "model_type": self.model_type,
            "processing_time": [],
            "memory_usage": [],
            "threshold_percents": []
        }
        
    def _detect_model_type(self):
        """Determine model architecture type by inspecting layers"""
        # Check for transformer components
        has_attention = any("attn" in name.lower() for name, _ in self.model.named_modules())
        has_mha = any("multihead" in name.lower() for name, _ in self.model.named_modules())
        
        # Check for CNN components
        has_conv = any(isinstance(module, torch.nn.Conv2d) for _, module in self.model.named_modules())
        
        # Check for RNN components
        has_rnn = any(isinstance(module, (torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU)) 
                      for _, module in self.model.named_modules())
        
        # Check for GNN components (look for common GNN module names)
        has_graph_conv = any(any(x in name.lower() for x in ["graphconv", "gatconv", "sageconv", "gcnconv"]) 
                            for name, _ in self.model.named_modules())
        
        # Determine model type based on components
        if has_attention or has_mha:
            return "transformer"
        elif has_conv:
            return "cnn"
        elif has_rnn:
            return "rnn"
        elif has_graph_conv:
            return "gnn"
        else:
            # Default to CNN since that's the original framework's focus
            print("Warning: Could not definitively detect model type. Defaulting to CNN.")
            return "cnn"
    
    @memory_optimization
    def explain(self, input_data, target_class=None, threshold_percent=10, **kwargs):
        """Generate explanations for the given input and target
        
        Args:
            input_data: Model input (format depends on model type)
            target_class: Target class to explain (None for predicted class)
            threshold_percent: Percentage threshold for simplification
            **kwargs: Additional parameters for specific explainers
            
        Returns:
            Dictionary with explanation data
        """
        # Call the appropriate explainer based on model type
        if self.model_type == "cnn":
            # For CNN, use the original ExplainableModel
            if hasattr(self.explainer, 'simplified_gradcam'):
                # Use the lightweight method with thresholding
                heatmap, processing_time = self.explainer.simplified_gradcam(
                    input_data, target_class=target_class, threshold_pct=threshold_percent)
                explanation_result = {
                    "heatmap": heatmap,
                    "processing_time": processing_time,
                    "threshold_percent": threshold_percent,
                    "target_class": target_class or self.explainer.last_prediction,
                    "memory_usage": self.explainer.last_memory_usage
                }
            else:
                # Fallback to standard Grad-CAM
                heatmap = self.explainer.generate_gradcam(input_data, target_class=target_class)
                # Apply thresholding
                from lightweight_explainability import simplify_cam
                simplified_heatmap = simplify_cam(heatmap, threshold_percent)
                explanation_result = {
                    "heatmap": heatmap,
                    "simplified_heatmap": simplified_heatmap,
                    "threshold_percent": threshold_percent,
                    "target_class": target_class
                }
        elif self.model_type == "transformer":
            # For transformer, use the TransformerExplainer
            explanation_result = self.explainer.explain(
                input_data, target_class=target_class, threshold_percent=threshold_percent, **kwargs)
        elif self.model_type == "rnn":
            # For RNN, use the RNNExplainer
            explanation_result = self.explainer.explain(
                input_data, target_class=target_class, threshold_percent=threshold_percent)
        elif self.model_type == "gnn":
            # For GNN, use the GNNExplainer
            explanation_result = self.explainer.explain(
                input_data, target_node=target_class, threshold_percent=threshold_percent)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        # Update performance stats
        self._update_performance_stats(explanation_result, threshold_percent)
            
        return explanation_result
    
    def visualize_explanation(self, input_data, explanation_result, **kwargs):
        """Visualize the explanation based on model type
        
        Args:
            input_data: Original input data
            explanation_result: Result from explain() method
            **kwargs: Additional visualization parameters
            
        Returns:
            PIL Image of the visualization
        """
        if self.model_type == "cnn":
            # For CNN, use the original visualization method
            if hasattr(self.explainer, 'visualize_explanation'):
                return self.explainer.visualize_explanation(input_data, explanation_result.get('heatmap'), **kwargs)
            else:
                # Fallback to show_heatmap
                from lightweight_explainability import show_heatmap
                return show_heatmap(input_data, explanation_result.get('simplified_heatmap', explanation_result.get('heatmap')))
        elif self.model_type == "transformer":
            # For transformer, use the TransformerExplainer visualization
            return self.explainer.visualize_explanation(input_data, explanation_result, **kwargs)
        elif self.model_type == "rnn":
            # For RNN, use the RNNExplainer visualization
            return self.explainer.visualize_explanation(input_data, explanation_result, **kwargs)
        elif self.model_type == "gnn":
            # For GNN, use the GNNExplainer visualization
            return self.explainer.visualize_explanation(input_data, explanation_result, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _update_performance_stats(self, explanation_result, threshold_percent):
        """Update performance statistics
        
        Args:
            explanation_result: Result from explain() method
            threshold_percent: Threshold percentage used
        """
        processing_time = explanation_result.get('processing_time', 0)
        memory_usage = explanation_result.get('memory_usage', 0)
        
        self.performance_stats["processing_time"].append(processing_time)
        self.performance_stats["memory_usage"].append(memory_usage)
        self.performance_stats["threshold_percents"].append(threshold_percent)
    
    def get_performance_summary(self):
        """Get summary of performance statistics
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.performance_stats["processing_time"]:
            return {"error": "No performance data available. Run explain() first."}
            
        return {
            "model_type": self.model_type,
            "avg_processing_time": np.mean(self.performance_stats["processing_time"]),
            "min_processing_time": np.min(self.performance_stats["processing_time"]),
            "max_processing_time": np.max(self.performance_stats["processing_time"]),
            "avg_memory_usage_mb": np.mean(self.performance_stats["memory_usage"]) / (1024 * 1024),
            "num_explanations": len(self.performance_stats["processing_time"]),
            "thresholds_used": list(set(self.performance_stats["threshold_percents"]))
        }
    
    def benchmark(self, input_data, target_class=None, thresholds=[1, 5, 10, 20], num_runs=5):
        """Run benchmark with different thresholds
        
        Args:
            input_data: Model input
            target_class: Target class to explain
            thresholds: List of threshold percentages to test
            num_runs: Number of runs for each threshold
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            "model_type": self.model_type,
            "thresholds": thresholds,
            "times": [],
            "memory_usage": [],
            "speedups": [],
            "memory_reductions": []
        }
        
        # First, run baseline (full explanation without simplification)
        baseline_times = []
        baseline_memory = []
        
        print(f"Running baseline benchmark ({num_runs} runs)...")
        for _ in range(num_runs):
            # For CNN, we use threshold 100 to get full explanation
            # For other models, we use a very high threshold that effectively keeps everything
            if self.model_type == "cnn":
                result = self.explain(input_data, target_class, 100)
            else:
                result = self.explain(input_data, target_class, 100)
            
            baseline_times.append(result["processing_time"])
            baseline_memory.append(result["memory_usage"])
            
        # Calculate baseline averages
        avg_baseline_time = np.mean(baseline_times)
        avg_baseline_memory = np.mean(baseline_memory)
        
        # Now test each threshold
        for threshold in thresholds:
            threshold_times = []
            threshold_memory = []
            
            print(f"Running benchmark with {threshold}% threshold ({num_runs} runs)...")
            for _ in range(num_runs):
                result = self.explain(input_data, target_class, threshold)
                threshold_times.append(result["processing_time"])
                threshold_memory.append(result["memory_usage"])
                
            # Calculate averages
            avg_time = np.mean(threshold_times)
            avg_memory = np.mean(threshold_memory)
            
            # Calculate speedup and memory reduction
            speedup = avg_baseline_time / avg_time if avg_time > 0 else 0
            memory_reduction = (avg_baseline_memory - avg_memory) / avg_baseline_memory * 100
            
            # Store results
            results["times"].append(avg_time)
            results["memory_usage"].append(avg_memory)
            results["speedups"].append(speedup)
            results["memory_reductions"].append(memory_reduction)
            
        return results
        
    def visualize_benchmark(self, benchmark_results, save_path=None):
        """Visualize benchmark results
        
        Args:
            benchmark_results: Results from benchmark() method
            save_path: Optional path to save the visualization
            
        Returns:
            None
        """
        thresholds = benchmark_results["thresholds"]
        speedups = benchmark_results["speedups"]
        memory_reductions = benchmark_results["memory_reductions"]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot speedup
        ax1.bar(range(len(thresholds)), speedups, tick_label=[f"{t}%" for t in thresholds])
        ax1.set_title(f"Speedup Factor ({benchmark_results['model_type']})")
        ax1.set_ylabel("Speedup (x times)")
        
        # Plot memory reduction
        ax2.bar(range(len(thresholds)), memory_reductions, tick_label=[f"{t}%" for t in thresholds])
        ax2.set_title(f"Memory Reduction ({benchmark_results['model_type']})")
        ax2.set_ylabel("Memory Reduction (%)")
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            
        plt.show()
        
def get_explainer_for_model(model, **kwargs):
    """Factory function to get appropriate explainer for a model
    
    Args:
        model: The model to explain
        **kwargs: Additional parameters for the explainer
        
    Returns:
        UniversalExplainer instance
    """
    return UniversalExplainer(model, **kwargs) 