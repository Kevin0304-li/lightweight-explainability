#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation metrics for explainability methods.

This script provides functions to evaluate explainability methods using
precision-recall curves and other quantitative metrics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.ndimage import gaussian_filter


def create_ground_truth_mask(image_size, important_regions=None):
    """Create a synthetic ground truth mask for evaluation.
    
    Args:
        image_size: Tuple of (height, width) for the mask
        important_regions: List of tuples (x, y, radius) for important regions
        
    Returns:
        Binary ground truth mask
    """
    if important_regions is None:
        # Default: create two important regions
        h, w = image_size
        important_regions = [
            (h // 3, w // 3, h // 10),  # Region 1: (y, x, radius)
            (2 * h // 3, 2 * w // 3, h // 8)  # Region 2: (y, x, radius)
        ]
    
    # Create empty mask
    mask = np.zeros(image_size, dtype=np.bool_)
    
    # Fill important regions
    for y, x, r in important_regions:
        y_grid, x_grid = np.ogrid[:image_size[0], :image_size[1]]
        dist = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)
        mask[dist <= r] = True
    
    return mask


def calculate_metrics(explanation, ground_truth):
    """Calculate precision, recall, and F1 score for an explanation.
    
    Args:
        explanation: Heatmap or importance map from explainability method
        ground_truth: Binary mask of truly important regions
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    # Binarize explanation map (non-zero values considered important)
    binary_explanation = explanation > 0
    
    # Calculate precision, recall, and F1
    true_positives = np.sum(binary_explanation & ground_truth)
    false_positives = np.sum(binary_explanation & ~ground_truth)
    false_negatives = np.sum(~binary_explanation & ground_truth)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }


def generate_precision_recall_curve(explanation, ground_truth):
    """Generate precision-recall curve for an explanation.
    
    Args:
        explanation: Heatmap or importance map from explainability method
        ground_truth: Binary mask of truly important regions
        
    Returns:
        Tuple of (precision, recall, thresholds) arrays
    """
    # Flatten arrays
    flat_explanation = explanation.flatten()
    flat_ground_truth = ground_truth.flatten()
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(flat_ground_truth, flat_explanation)
    
    return precision, recall, thresholds


def compare_thresholds_pr_curve(explanation, ground_truth, thresholds=[1, 5, 10, 20, 30, 50]):
    """Compare different thresholds using precision-recall metrics.
    
    Args:
        explanation: Raw heatmap or importance map from explainability method
        ground_truth: Binary mask of truly important regions
        thresholds: List of threshold percentages to evaluate
        
    Returns:
        Dictionary with metrics for each threshold
    """
    results = {}
    
    # Get full precision-recall curve
    precision, recall, _ = generate_precision_recall_curve(explanation, ground_truth)
    
    # Calculate average precision
    ap = average_precision_score(ground_truth.flatten(), explanation.flatten())
    
    results["full_curve"] = {
        "precision": precision,
        "recall": recall,
        "average_precision": ap
    }
    
    # Evaluate specific thresholds
    for threshold in thresholds:
        # Apply threshold
        sorted_values = np.sort(explanation.flatten())[::-1]
        threshold_idx = int(len(sorted_values) * threshold / 100)
        threshold_value = sorted_values[threshold_idx]
        
        thresholded = explanation.copy()
        thresholded[thresholded < threshold_value] = 0
        
        # Calculate metrics
        metrics = calculate_metrics(thresholded, ground_truth)
        
        # Store results
        results[f"threshold_{threshold}"] = {
            "metrics": metrics,
            "threshold_value": threshold_value
        }
    
    return results


def plot_precision_recall_curves(results, model_names=None, save_path=None):
    """Plot precision-recall curves for multiple models or thresholding approaches.
    
    Args:
        results: Dictionary of results from compare_thresholds_pr_curve
                or list of such dictionaries for multiple models
        model_names: List of model names (if results is a list)
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 8))
    
    # Handle both single and multiple models
    if not isinstance(results, list):
        results = [results]
    
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(results))]
    
    # Colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Plot each model's precision-recall curve
    for i, (result, model_name) in enumerate(zip(results, model_names)):
        precision = result["full_curve"]["precision"]
        recall = result["full_curve"]["recall"]
        ap = result["full_curve"]["average_precision"]
        
        plt.plot(recall, precision, color=colors[i], 
                 label=f"{model_name} (AP={ap:.3f})", linewidth=2)
        
        # Mark threshold points if available
        threshold_keys = [key for key in result.keys() if key.startswith("threshold_")]
        for key in threshold_keys:
            threshold = int(key.split("_")[1])
            metrics = result[key]["metrics"]
            plt.scatter(metrics["recall"], metrics["precision"], 
                        color=colors[i], s=50, zorder=5)
            plt.annotate(f"{threshold}%", 
                        (metrics["recall"], metrics["precision"]),
                        xytext=(5, 5), textcoords='offset points')
    
    # Set plot properties
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves for Explanation Quality at Different Thresholds', fontsize=16)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', fontsize=12)
    
    # Add iso-f1 curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        valid_idx = y <= 1
        plt.plot(x[valid_idx], y[valid_idx], color='gray', alpha=0.3)
        plt.annotate(f'f1={f_score:0.1f}', xy=(x[valid_idx][-1], y[valid_idx][-1]))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def compare_architectures_pr_curves(explanations, ground_truths, architecture_names, thresholds=[10], save_path=None):
    """Compare PR curves across different model architectures.
    
    Args:
        explanations: Dictionary mapping architecture names to explanation maps
        ground_truths: Dictionary mapping architecture names to ground truth masks
        architecture_names: List of architecture names to include
        thresholds: List of threshold percentages to evaluate
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(12, 10))
    
    # Colors for different architectures
    colors = plt.cm.tab10(np.linspace(0, 1, len(architecture_names)))
    
    # Track average precision scores
    ap_scores = {}
    
    # Plot each architecture's precision-recall curve
    for i, arch_name in enumerate(architecture_names):
        if arch_name not in explanations or arch_name not in ground_truths:
            continue
            
        explanation = explanations[arch_name]
        ground_truth = ground_truths[arch_name]
        
        # Generate precision-recall curve
        precision, recall, _ = generate_precision_recall_curve(explanation, ground_truth)
        
        # Calculate average precision
        ap = average_precision_score(ground_truth.flatten(), explanation.flatten())
        ap_scores[arch_name] = ap
        
        # Plot PR curve
        plt.plot(recall, precision, color=colors[i], 
                 label=f"{arch_name} (AP={ap:.3f})", linewidth=2)
        
        # Mark threshold points
        for threshold in thresholds:
            # Apply threshold
            sorted_values = np.sort(explanation.flatten())[::-1]
            threshold_idx = int(len(sorted_values) * threshold / 100)
            threshold_value = sorted_values[threshold_idx]
            
            thresholded = explanation.copy()
            thresholded[thresholded < threshold_value] = 0
            
            # Calculate metrics
            metrics = calculate_metrics(thresholded, ground_truth)
            
            # Plot point
            plt.scatter(metrics["recall"], metrics["precision"], 
                        color=colors[i], s=80, zorder=5, marker='o')
            plt.annotate(f"{threshold}%", 
                        (metrics["recall"], metrics["precision"]),
                        xytext=(5, 5), textcoords='offset points')
    
    # Set plot properties
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves Across Different Architectures', fontsize=16)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', fontsize=12)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf(), ap_scores


def evaluate_explanation(explanation, ground_truth, threshold_percent=10, plot=False, save_path=None):
    """Evaluate an explanation against ground truth.
    
    Args:
        explanation: Heatmap or importance map from explainability method
        ground_truth: Binary mask of truly important regions
        threshold_percent: Percentage threshold to apply
        plot: Whether to generate visualization
        save_path: Path to save visualization (if plot=True)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Apply threshold
    sorted_values = np.sort(explanation.flatten())[::-1]
    threshold_idx = int(len(sorted_values) * threshold_percent / 100)
    threshold_value = sorted_values[threshold_idx]
    
    thresholded = explanation.copy()
    thresholded[thresholded < threshold_value] = 0
    
    # Calculate metrics
    metrics = calculate_metrics(thresholded, ground_truth)
    
    # Create visualization if requested
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot ground truth
        axs[0].imshow(ground_truth, cmap='gray')
        axs[0].set_title('Ground Truth')
        axs[0].axis('off')
        
        # Plot raw explanation
        axs[1].imshow(explanation, cmap='jet')
        axs[1].set_title('Raw Explanation')
        axs[1].axis('off')
        
        # Plot thresholded explanation
        axs[2].imshow(thresholded, cmap='jet')
        axs[2].set_title(f'Thresholded ({threshold_percent}%)')
        axs[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return {
        "metrics": metrics,
        "threshold_value": threshold_value,
        "threshold_percent": threshold_percent
    }


def generate_tradeoff_plot(results, metric_x='recall', metric_y='precision', save_path=None):
    """Generate trade-off plot for different thresholds.
    
    Args:
        results: Dictionary with threshold results from compare_thresholds_pr_curve
        metric_x: X-axis metric ('recall', 'precision', or 'f1')
        metric_y: Y-axis metric ('recall', 'precision', or 'f1')
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=(10, 8))
    
    thresholds = []
    x_values = []
    y_values = []
    
    for key, value in results.items():
        if key.startswith("threshold_"):
            threshold = int(key.split("_")[1])
            thresholds.append(threshold)
            x_values.append(value["metrics"][metric_x])
            y_values.append(value["metrics"][metric_y])
    
    # Sort by threshold
    sorted_idx = np.argsort(thresholds)
    thresholds = [thresholds[i] for i in sorted_idx]
    x_values = [x_values[i] for i in sorted_idx]
    y_values = [y_values[i] for i in sorted_idx]
    
    # Plot trade-off curve
    plt.plot(x_values, y_values, 'o-', linewidth=2)
    
    # Annotate points with threshold values
    for i, threshold in enumerate(thresholds):
        plt.annotate(f"{threshold}%", 
                    (x_values[i], y_values[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    # Set plot properties
    plt.xlabel(metric_x.capitalize(), fontsize=14)
    plt.ylabel(metric_y.capitalize(), fontsize=14)
    plt.title(f'{metric_y.capitalize()}-{metric_x.capitalize()} Trade-off for Different Thresholds', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def main():
    """Run demonstration of evaluation metrics."""
    # Output directory
    output_dir = "./results/evaluation_metrics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic data for demonstration
    image_size = (100, 100)
    
    # Ground truth: two important regions
    ground_truth = create_ground_truth_mask(image_size)
    
    # Create synthetic explanations
    # 1. Ideal explanation (matches ground truth perfectly with some intensity variation)
    ideal_explanation = np.zeros(image_size)
    ideal_explanation[ground_truth] = np.random.uniform(0.5, 1.0, size=np.sum(ground_truth))
    
    # 2. Realistic explanation (matches ground truth with noise and some false positives)
    realistic_explanation = np.zeros(image_size)
    realistic_explanation[ground_truth] = np.random.uniform(0.3, 1.0, size=np.sum(ground_truth))
    # Add some noise
    realistic_explanation += np.random.uniform(0, 0.2, size=image_size)
    # Add some false positives
    fp_mask = create_ground_truth_mask(image_size, [(20, 80, 10)])
    realistic_explanation[fp_mask] = np.random.uniform(0.2, 0.7, size=np.sum(fp_mask))
    
    # 3. Poor explanation (mostly noise with weak signal)
    poor_explanation = np.random.uniform(0, 0.3, size=image_size)
    poor_explanation[ground_truth] = np.random.uniform(0.2, 0.5, size=np.sum(ground_truth))
    
    # Evaluate explanations at different thresholds
    thresholds = [1, 5, 10, 20, 30, 50]
    
    ideal_results = compare_thresholds_pr_curve(ideal_explanation, ground_truth, thresholds)
    realistic_results = compare_thresholds_pr_curve(realistic_explanation, ground_truth, thresholds)
    poor_results = compare_thresholds_pr_curve(poor_explanation, ground_truth, thresholds)
    
    # Plot PR curves
    all_results = [ideal_results, realistic_results, poor_results]
    model_names = ["Ideal Explanation", "Realistic Explanation", "Poor Explanation"]
    
    plot_precision_recall_curves(all_results, model_names, 
                                 save_path=f"{output_dir}/pr_curves.png")
    
    # Generate trade-off plots
    generate_tradeoff_plot(realistic_results, 'recall', 'precision', 
                          save_path=f"{output_dir}/precision_recall_tradeoff.png")
    generate_tradeoff_plot(realistic_results, 'recall', 'f1', 
                          save_path=f"{output_dir}/f1_recall_tradeoff.png")
    
    # Compare different architecture types (simulated)
    architecture_explanations = {
        "CNN": realistic_explanation,
        "Transformer": gaussian_filter(realistic_explanation, sigma=1.0),
        "RNN": gaussian_filter(realistic_explanation, sigma=2.0),
        "GNN": gaussian_filter(realistic_explanation, sigma=3.0)
    }
    
    # Use same ground truth for all architectures in this demo
    architecture_ground_truths = {
        "CNN": ground_truth,
        "Transformer": ground_truth,
        "RNN": ground_truth,
        "GNN": ground_truth
    }
    
    compare_architectures_pr_curves(
        architecture_explanations, 
        architecture_ground_truths,
        ["CNN", "Transformer", "RNN", "GNN"],
        thresholds=[10, 30],
        save_path=f"{output_dir}/architecture_comparison.png"
    )
    
    print(f"Evaluation results saved to {output_dir}")


if __name__ == "__main__":
    main() 