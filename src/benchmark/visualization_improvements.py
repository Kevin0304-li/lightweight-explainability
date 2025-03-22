#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate improved visualizations for explainability evaluation metrics.

This script enhances the precision-recall curve visualizations for publication quality,
with improved readability and proper academic styling.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
from sklearn.metrics import precision_recall_curve, average_precision_score

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the evaluation metrics module
try:
    # First try direct import (if in the same package)
    from evaluation_metrics import (
        create_ground_truth_mask, 
        compare_thresholds_pr_curve,
        compare_architectures_pr_curves
    )
except ImportError:
    try:
        # Then try from benchmark package
        from benchmark.evaluation_metrics import (
            create_ground_truth_mask, 
            compare_thresholds_pr_curve,
            compare_architectures_pr_curves
        )
    except ImportError:
        try:
            # Try absolute import
            from src.benchmark.evaluation_metrics import (
                create_ground_truth_mask, 
                compare_thresholds_pr_curve,
                compare_architectures_pr_curves
            )
        except ImportError:
            print("Evaluation metrics module not found. Make sure evaluation_metrics.py is in the benchmark directory.")
            sys.exit(1)

# Set up publication-quality plot parameters
def set_publication_style():
    """Configure matplotlib for publication-quality plots."""
    # Increase font sizes
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 14
    rcParams['axes.labelsize'] = 16
    rcParams['axes.titlesize'] = 18
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14
    rcParams['legend.fontsize'] = 14
    rcParams['figure.titlesize'] = 20
    
    # Make lines thicker
    rcParams['lines.linewidth'] = 2.5
    rcParams['axes.linewidth'] = 1.5
    rcParams['xtick.major.width'] = 1.5
    rcParams['ytick.major.width'] = 1.5
    
    # Set DPI for print quality
    rcParams['figure.dpi'] = 300
    rcParams['savefig.dpi'] = 300
    
    # Make grid lighter
    rcParams['grid.alpha'] = 0.3
    rcParams['grid.linestyle'] = '--'

def plot_improved_pr_curves(results, model_names=None, save_path=None, add_caption=True):
    """Generate publication-quality precision-recall curve plot.
    
    Args:
        results: List of dictionaries from compare_thresholds_pr_curve
        model_names: List of model names
        save_path: Path to save the figure
        add_caption: Whether to add a figure caption
        
    Returns:
        Matplotlib figure
    """
    # Set publication style
    set_publication_style()
    
    # Create figure
    fig = plt.figure(figsize=(10, 8 if add_caption else 7))
    
    # Add an extra subplot for the caption if needed
    if add_caption:
        gs = fig.add_gridspec(2, 1, height_ratios=[6, 1])
        ax = fig.add_subplot(gs[0, 0])
    else:
        ax = fig.add_subplot(111)
    
    # Handle case with single results
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
        
        ax.plot(recall, precision, color=colors[i], 
                label=f"{model_name} (AP={ap:.3f})", linewidth=2.5)
        
        # Mark threshold points if available
        threshold_keys = [key for key in result.keys() if key.startswith("threshold_")]
        for key in threshold_keys:
            threshold = int(key.split("_")[1])
            metrics = result[key]["metrics"]
            ax.scatter(metrics["recall"], metrics["precision"], 
                      color=colors[i], s=70, zorder=5)
            
            # Only annotate important thresholds to avoid clutter
            ax.annotate(f"{threshold}%", 
                      (metrics["recall"], metrics["precision"]),
                      xytext=(5, 5), textcoords='offset points',
                      fontsize=12, fontweight='bold')
    
    # Set plot properties
    ax.set_xlabel('Recall', fontsize=16, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=16, fontweight='bold')
    ax.set_title('Precision-Recall Curves for Explanation Quality at Different Thresholds', 
                fontsize=18, fontweight='bold')
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', fontsize=14)
    
    # Make all spines visible and thicker
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    
    # Add iso-f1 curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        valid_idx = y <= 1
        ax.plot(x[valid_idx], y[valid_idx], color='gray', alpha=0.4, linestyle=':')
        ax.annotate(f'f1={f_score:0.1f}', xy=(x[valid_idx][-1], y[valid_idx][-1]),
                   fontsize=12, alpha=0.7)
    
    # Add caption if requested
    if add_caption:
        caption_ax = fig.add_subplot(gs[1, 0])
        caption_ax.axis('off')
        caption = ("Figure 1. Precision-Recall curves comparing explanation quality across different thresholds. "
                  "These curves illustrate the trade-off between explanation faithfulness (precision) and "
                  "completeness (recall). Higher precision indicates fewer false positives (higher faithfulness), "
                  "while higher recall indicates fewer false negatives (higher completeness). "
                  "Threshold percentages indicate the proportion of most important features retained. "
                  "The ideal curve (AP=1.0) represents perfect alignment with ground truth importance.")
        caption_ax.text(0, 0.5, caption, fontsize=12, wrap=True, va='center')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_improved_architecture_comparison(explanations, ground_truths, architecture_names, 
                                         thresholds=[10, 30], save_path=None, add_caption=True):
    """Generate publication-quality architecture comparison plot.
    
    Args:
        explanations: Dictionary mapping architecture names to explanation maps
        ground_truths: Dictionary mapping architecture names to ground truth masks
        architecture_names: List of architecture names to include
        thresholds: List of threshold percentages to evaluate
        save_path: Path to save the figure
        add_caption: Whether to add a figure caption
        
    Returns:
        Matplotlib figure
    """
    # Set publication style
    set_publication_style()
    
    # Create figure
    fig = plt.figure(figsize=(10, 8 if add_caption else 7))
    
    # Add an extra subplot for the caption if needed
    if add_caption:
        gs = fig.add_gridspec(2, 1, height_ratios=[6, 1])
        ax = fig.add_subplot(gs[0, 0])
    else:
        ax = fig.add_subplot(111)
    
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
        precision, recall, _ = precision_recall_curve(ground_truth.flatten(), explanation.flatten())
        
        # Calculate average precision
        ap = average_precision_score(ground_truth.flatten(), explanation.flatten())
        ap_scores[arch_name] = ap
        
        # Plot PR curve
        ax.plot(recall, precision, color=colors[i], 
               label=f"{arch_name} (AP={ap:.3f})", linewidth=2.5)
        
        # Mark threshold points
        for threshold in thresholds:
            # Apply threshold
            sorted_values = np.sort(explanation.flatten())[::-1]
            threshold_idx = int(len(sorted_values) * threshold / 100)
            threshold_value = sorted_values[threshold_idx]
            
            thresholded = explanation.copy()
            thresholded[thresholded < threshold_value] = 0
            
            # Calculate metrics
            true_positives = np.sum((thresholded > 0) & ground_truth)
            false_positives = np.sum((thresholded > 0) & ~ground_truth)
            false_negatives = np.sum(~(thresholded > 0) & ground_truth)
            
            precision_val = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall_val = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # Plot point
            ax.scatter(recall_val, precision_val, 
                      color=colors[i], s=80, zorder=5, marker='o')
            ax.annotate(f"{threshold}%", 
                      (recall_val, precision_val),
                      xytext=(5, 5), textcoords='offset points',
                      fontsize=12, fontweight='bold')
    
    # Set plot properties
    ax.set_xlabel('Recall', fontsize=16, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=16, fontweight='bold')
    ax.set_title('Precision-Recall Curves Across Different Architectures', 
                fontsize=18, fontweight='bold')
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', fontsize=14)
    
    # Make all spines visible and thicker
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    
    # Add caption if requested
    if add_caption:
        caption_ax = fig.add_subplot(gs[1, 0])
        caption_ax.axis('off')
        caption = ("Figure 2. Precision-Recall curves comparing explanation quality across different neural "
                  "network architectures. Each architecture demonstrates a different faithfulness-completeness "
                  "trade-off profile. Average Precision (AP) scores quantify overall explanation quality, with "
                  "higher values indicating better alignment with ground truth. Marked points show specific "
                  "threshold levels, revealing how aggressively filtering explanation features affects performance "
                  "for each architecture type.")
        caption_ax.text(0, 0.5, caption, fontsize=12, wrap=True, va='center')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ap_scores

def main():
    """Generate improved visualizations."""
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
    
    # Generate improved PR curve plot
    all_results = [ideal_results, realistic_results, poor_results]
    model_names = ["Ideal Explanation", "Realistic Explanation", "Poor Explanation"]
    
    # Generate plots with captions
    plot_improved_pr_curves(
        all_results, 
        model_names, 
        save_path=f"{output_dir}/pr_curves_publication.png",
        add_caption=True
    )
    
    # Also generate a version without caption for flexibility
    plot_improved_pr_curves(
        all_results, 
        model_names, 
        save_path=f"{output_dir}/pr_curves_publication_no_caption.png",
        add_caption=False
    )
    
    # Compare different architecture types (simulated)
    from scipy.ndimage import gaussian_filter
    
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
    
    # Generate improved architecture comparison plot
    plot_improved_architecture_comparison(
        architecture_explanations, 
        architecture_ground_truths,
        ["CNN", "Transformer", "RNN", "GNN"],
        thresholds=[10, 30],
        save_path=f"{output_dir}/architecture_comparison_publication.png",
        add_caption=True
    )
    
    print(f"Improved visualization results saved to {output_dir}")
    print("Generated files:")
    print(f" - {output_dir}/pr_curves_publication.png (with caption)")
    print(f" - {output_dir}/pr_curves_publication_no_caption.png (without caption)")
    print(f" - {output_dir}/architecture_comparison_publication.png")


if __name__ == "__main__":
    main() 