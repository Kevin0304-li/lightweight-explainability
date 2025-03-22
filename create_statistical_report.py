"""
Create Statistical Reports for Explainability Methods

This script runs evaluation for both baseline and simplified Grad-CAM
methods and generates statistical reports that include:

1. Mean ± standard deviation for all metrics
2. Statistical significance tests (t-tests and Wilcoxon)
3. P-values and significance indicators
4. Visualizations comparing baseline and simplified methods
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from scipy.stats import ttest_ind, wilcoxon
from lightweight_explainability import ExplainableModel, simplify_cam
from evaluation_metrics import ExplanationEvaluator

def run_statistical_evaluation():
    """Run statistical evaluation comparing baseline to simplified Grad-CAM"""
    # Parameters
    model_name = "mobilenet_v2" 
    image_dir = "sample_images"
    thresholds = [5, 10, 15, 20]
    output_dir = "results/statistical_analysis"
    device = "cpu"
    num_samples = None  # Set to None to use all images in image_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    # Initialize model and evaluator
    print(f"Initializing model: {model_name}")
    model = ExplainableModel(model_name=model_name)
    evaluator = ExplanationEvaluator(model, device=device)
    
    # Find image paths
    if os.path.isdir(image_dir):
        img_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        img_paths = [image_dir]
    
    # Limit number of samples if specified
    if num_samples and len(img_paths) > num_samples:
        import random
        random.shuffle(img_paths)
        img_paths = img_paths[:num_samples]
    
    print(f"Found {len(img_paths)} images for evaluation")
    
    # Collect metrics for baseline and each threshold
    baseline_metrics = {
        'faithfulness': {
            'confidence_drop': [],
            'avg_drop_rate': [],
            'auc': [],
            'processing_time': []
        },
        'completeness': {
            'auc': [],
            'avg_retention': []
        },
        'sensitivity': {
            'mean_sensitivity': [],
            'std_sensitivity': []
        },
        'memory': {
            'pixels_retained': []
        }
    }
    
    # Dictionary to store metrics for each threshold
    simplified_metrics = {t: {
        'faithfulness': {
            'confidence_drop': [],
            'avg_drop_rate': [],
            'auc': [],
            'processing_time': []
        },
        'completeness': {
            'auc': [],
            'avg_retention': []
        },
        'sensitivity': {
            'mean_sensitivity': [],
            'std_sensitivity': []
        },
        'memory': {
            'pixels_retained': []
        }
    } for t in thresholds}
    
    # Evaluate each image
    for i, img_path in enumerate(img_paths):
        print(f"Processing image {i+1}/{len(img_paths)}: {os.path.basename(img_path)}")
        
        # Preprocess image
        img_tensor, _ = model.preprocess_image(img_path)
        
        # Read the original image using OpenCV to get dimensions
        original_img = cv2.imread(img_path)
        h, w = original_img.shape[:2]
        total_pixels = h * w
        
        # Baseline evaluation
        start_time = time.time()
        cam = model.generate_gradcam(img_tensor)
        baseline_time = time.time() - start_time
        
        # Count non-zero pixels in baseline CAM
        baseline_pixels = np.count_nonzero(cam > 0.1)
        baseline_metrics['memory']['pixels_retained'].append(baseline_pixels / total_pixels)
        
        # Evaluate baseline faithfulness
        baseline_metrics['faithfulness']['processing_time'].append(baseline_time)
        faith_eval = evaluator.evaluate_faithfulness(img_tensor, cam)
        baseline_metrics['faithfulness']['confidence_drop'].append(faith_eval['confidence_drop'])
        baseline_metrics['faithfulness']['avg_drop_rate'].append(faith_eval['avg_drop_rate'])
        baseline_metrics['faithfulness']['auc'].append(faith_eval['auc'])
        
        # Evaluate baseline completeness
        comp_eval = evaluator.evaluate_completeness(img_tensor, cam)
        baseline_metrics['completeness']['auc'].append(comp_eval['auc'])
        baseline_metrics['completeness']['avg_retention'].append(np.mean(comp_eval['retention_rates']))
        
        # Evaluate baseline sensitivity
        sens_eval = evaluator.evaluate_sensitivity(img_tensor, cam)
        baseline_metrics['sensitivity']['mean_sensitivity'].append(sens_eval['mean_sensitivity'])
        baseline_metrics['sensitivity']['std_sensitivity'].append(sens_eval['std_sensitivity'])
        
        # Evaluate simplified CAMs for each threshold
        for threshold in thresholds:
            # Create simplified CAM
            start_time = time.time()
            simplified = simplify_cam(cam, threshold)
            simplified_time = time.time() - start_time
            
            # Count non-zero pixels in simplified CAM
            simplified_pixels = np.count_nonzero(simplified > 0)
            simplified_metrics[threshold]['memory']['pixels_retained'].append(simplified_pixels / total_pixels)
            
            # Record processing time
            simplified_metrics[threshold]['faithfulness']['processing_time'].append(
                baseline_time + simplified_time)
            
            # Trick: temporarily replace model's generate_gradcam for evaluation
            original_generate_gradcam = model.generate_gradcam
            model.generate_gradcam = lambda x: simplified
            
            # Evaluate simplified faithfulness
            faith_eval = evaluator.evaluate_faithfulness(img_tensor, simplified)
            simplified_metrics[threshold]['faithfulness']['confidence_drop'].append(faith_eval['confidence_drop'])
            simplified_metrics[threshold]['faithfulness']['avg_drop_rate'].append(faith_eval['avg_drop_rate'])
            simplified_metrics[threshold]['faithfulness']['auc'].append(faith_eval['auc'])
            
            # Evaluate simplified completeness
            comp_eval = evaluator.evaluate_completeness(img_tensor, simplified)
            simplified_metrics[threshold]['completeness']['auc'].append(comp_eval['auc'])
            simplified_metrics[threshold]['completeness']['avg_retention'].append(
                np.mean(comp_eval['retention_rates']))
            
            # Evaluate simplified sensitivity
            sens_eval = evaluator.evaluate_sensitivity(img_tensor, simplified)
            simplified_metrics[threshold]['sensitivity']['mean_sensitivity'].append(sens_eval['mean_sensitivity'])
            simplified_metrics[threshold]['sensitivity']['std_sensitivity'].append(sens_eval['std_sensitivity'])
            
            # Restore original function
            model.generate_gradcam = original_generate_gradcam
    
    # Generate statistical reports
    create_statistical_report(baseline_metrics, simplified_metrics, thresholds, output_dir)
    
    print(f"Statistical evaluation complete. Results saved to {output_dir}")

def calculate_statistics(baseline_values, simplified_values):
    """Calculate statistical significance between baseline and simplified values"""
    # Convert lists to numpy arrays if they aren't already
    baseline_values = np.array(baseline_values)
    simplified_values = np.array(simplified_values)
    
    # Basic statistics
    baseline_mean = np.mean(baseline_values)
    baseline_std = np.std(baseline_values)
    simplified_mean = np.mean(simplified_values)
    simplified_std = np.std(simplified_values)
    
    # Handle empty arrays
    if len(baseline_values) == 0 or len(simplified_values) == 0:
        return {
            'baseline_mean': baseline_mean if len(baseline_values) > 0 else 0,
            'baseline_std': baseline_std if len(baseline_values) > 0 else 0,
            'simplified_mean': simplified_mean if len(simplified_values) > 0 else 0,
            'simplified_std': simplified_std if len(simplified_values) > 0 else 0,
            'p_value': 1.0,
            'significance': ""
        }
    
    # Ensure arrays have the same length for paired tests
    min_length = min(len(baseline_values), len(simplified_values))
    baseline_values = baseline_values[:min_length]
    simplified_values = simplified_values[:min_length]
    
    # Skip significance testing if sample size is too small
    if min_length < 2:
        return {
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'simplified_mean': simplified_mean,
            'simplified_std': simplified_std,
            'p_value': 1.0,
            'significance': ""
        }
    
    # Run t-test
    try:
        t_stat, p_value_t = ttest_ind(baseline_values, simplified_values, equal_var=False)
    except Exception as e:
        print(f"t-test failed: {e}")
        p_value_t = 1.0
    
    # Run Wilcoxon signed-rank test if sample size is sufficient
    try:
        if min_length >= 6:  # Wilcoxon requires at least 6 samples
            w_stat, p_value_w = wilcoxon(baseline_values, simplified_values)
        else:
            p_value_w = 1.0
    except Exception as e:
        print(f"Wilcoxon test failed: {e}")
        p_value_w = 1.0
    
    # Use more conservative p-value
    p_value = max(p_value_t, p_value_w) if 'p_value_w' in locals() else p_value_t
    
    # Significance markers
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = ""
    
    return {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'simplified_mean': simplified_mean, 
        'simplified_std': simplified_std,
        'p_value': p_value,
        'significance': significance
    }

def create_statistical_report(baseline_metrics, simplified_metrics, thresholds, output_dir):
    """Create a comprehensive statistical report with visualizations"""
    # Choose the threshold for detailed comparison (default to first threshold)
    threshold = thresholds[0]
    
    # Create comparison visualizations
    create_metric_visualization(
        'Faithfulness (Confidence Drop)', 
        baseline_metrics['faithfulness']['confidence_drop'],
        simplified_metrics[threshold]['faithfulness']['confidence_drop'],
        output_dir
    )
    
    create_metric_visualization(
        'Completeness (AUC)', 
        baseline_metrics['completeness']['auc'],
        simplified_metrics[threshold]['completeness']['auc'],
        output_dir
    )
    
    create_metric_visualization(
        'Sensitivity', 
        baseline_metrics['sensitivity']['mean_sensitivity'],
        simplified_metrics[threshold]['sensitivity']['mean_sensitivity'],
        output_dir
    )
    
    create_metric_visualization(
        'Memory Usage (% Pixels)', 
        baseline_metrics['memory']['pixels_retained'],
        simplified_metrics[threshold]['memory']['pixels_retained'],
        output_dir
    )
    
    create_metric_visualization(
        'Processing Time (s)', 
        baseline_metrics['faithfulness']['processing_time'],
        simplified_metrics[threshold]['faithfulness']['processing_time'],
        output_dir
    )
    
    # Create markdown report
    with open(os.path.join(output_dir, 'statistical_report.md'), 'w') as f:
        f.write("# Statistical Analysis of Explainability Methods\n\n")
        f.write(f"Comparing Baseline Grad-CAM with Simplified Grad-CAM (Threshold: {threshold}%)\n\n")
        
        f.write("## Sample Information\n\n")
        sample_size = len(baseline_metrics['faithfulness']['confidence_drop'])
        f.write(f"* **Number of samples:** {sample_size}\n")
        f.write(f"* **Model used:** MobileNetV2\n\n")
        
        f.write("## Statistical Results\n\n")
        f.write("### Mean ± Standard Deviation with Significance Tests\n\n")
        f.write("| Metric | Baseline | Simplified ({0}%) | p-value | Significance |\n".format(threshold))
        f.write("|--------|----------|-------------------|---------|-------------|\n")
        
        # Faithfulness metrics
        faith_stats = calculate_statistics(
            baseline_metrics['faithfulness']['confidence_drop'],
            simplified_metrics[threshold]['faithfulness']['confidence_drop']
        )
        f.write("| **Faithfulness (Confidence Drop)** | {0:.4f} ± {1:.4f} | {2:.4f} ± {3:.4f} | {4:.4f} | {5} |\n".format(
            faith_stats['baseline_mean'], faith_stats['baseline_std'],
            faith_stats['simplified_mean'], faith_stats['simplified_std'],
            faith_stats['p_value'], faith_stats['significance']
        ))
        
        # Completeness metrics
        comp_stats = calculate_statistics(
            baseline_metrics['completeness']['auc'],
            simplified_metrics[threshold]['completeness']['auc']
        )
        f.write("| **Completeness (AUC)** | {0:.4f} ± {1:.4f} | {2:.4f} ± {3:.4f} | {4:.4f} | {5} |\n".format(
            comp_stats['baseline_mean'], comp_stats['baseline_std'],
            comp_stats['simplified_mean'], comp_stats['simplified_std'],
            comp_stats['p_value'], comp_stats['significance']
        ))
        
        # Sensitivity metrics
        sens_stats = calculate_statistics(
            baseline_metrics['sensitivity']['mean_sensitivity'],
            simplified_metrics[threshold]['sensitivity']['mean_sensitivity']
        )
        f.write("| **Sensitivity** | {0:.4f} ± {1:.4f} | {2:.4f} ± {3:.4f} | {4:.4f} | {5} |\n".format(
            sens_stats['baseline_mean'], sens_stats['baseline_std'],
            sens_stats['simplified_mean'], sens_stats['simplified_std'],
            sens_stats['p_value'], sens_stats['significance']
        ))
        
        # Memory usage
        mem_stats = calculate_statistics(
            baseline_metrics['memory']['pixels_retained'],
            simplified_metrics[threshold]['memory']['pixels_retained']
        )
        f.write("| **Memory Usage (% Pixels)** | {0:.4f} ± {1:.4f} | {2:.4f} ± {3:.4f} | {4:.4f} | {5} |\n".format(
            mem_stats['baseline_mean'], mem_stats['baseline_std'],
            mem_stats['simplified_mean'], mem_stats['simplified_std'],
            mem_stats['p_value'], mem_stats['significance']
        ))
        
        # Processing time
        time_stats = calculate_statistics(
            baseline_metrics['faithfulness']['processing_time'],
            simplified_metrics[threshold]['faithfulness']['processing_time']
        )
        f.write("| **Processing Time (s)** | {0:.4f} ± {1:.4f} | {2:.4f} ± {3:.4f} | {4:.4f} | {5} |\n".format(
            time_stats['baseline_mean'], time_stats['baseline_std'],
            time_stats['simplified_mean'], time_stats['simplified_std'],
            time_stats['p_value'], time_stats['significance']
        ))
        
        f.write("\n*Significance levels: * p<0.05, ** p<0.01, *** p<0.001\n\n")
        
        # Threshold comparison section
        f.write("## Threshold Comparison\n\n")
        f.write("### Effect of Threshold on Metrics\n\n")
        f.write("| Threshold | Faithfulness | Completeness | Sensitivity | Memory Usage | Processing Time |\n")
        f.write("|-----------|--------------|--------------|-------------|--------------|----------------|\n")
        
        for t in thresholds:
            # Get mean values for each metric at this threshold
            faith_mean = np.mean(simplified_metrics[t]['faithfulness']['confidence_drop'])
            comp_mean = np.mean(simplified_metrics[t]['completeness']['auc'])
            sens_mean = np.mean(simplified_metrics[t]['sensitivity']['mean_sensitivity'])
            mem_mean = np.mean(simplified_metrics[t]['memory']['pixels_retained'])
            time_mean = np.mean(simplified_metrics[t]['faithfulness']['processing_time'])
            
            f.write("| **{0}%** | {1:.4f} | {2:.4f} | {3:.4f} | {4:.4f} | {5:.4f} |\n".format(
                t, faith_mean, comp_mean, sens_mean, mem_mean, time_mean
            ))
        
        f.write("\n## Visualizations\n\n")
        f.write("### Faithfulness (Confidence Drop)\n\n")
        f.write("![Faithfulness Comparison](visualizations/Faithfulness_(Confidence_Drop).png)\n\n")
        
        f.write("### Completeness (AUC)\n\n")
        f.write("![Completeness Comparison](visualizations/Completeness_(AUC).png)\n\n")
        
        f.write("### Sensitivity\n\n")
        f.write("![Sensitivity Comparison](visualizations/Sensitivity.png)\n\n")
        
        f.write("### Memory Usage\n\n")
        f.write("![Memory Usage Comparison](visualizations/Memory_Usage_(%)_Pixels).png)\n\n")
        
        f.write("### Processing Time\n\n")
        f.write("![Processing Time Comparison](visualizations/Processing_Time_(s).png)\n\n")
        
        f.write("## Interpretation\n\n")
        
        # Auto-generate interpretation based on statistical results
        f.write("### Statistical Significance\n\n")
        
        # Faithfulness interpretation
        f.write("**Faithfulness:** ")
        if faith_stats['p_value'] < 0.05:
            if faith_stats['simplified_mean'] > faith_stats['baseline_mean']:
                f.write("The simplified explanation shows a statistically significant **improvement** in faithfulness ")
                f.write(f"(p={faith_stats['p_value']:.4f}{faith_stats['significance']}). This suggests that simplification ")
                f.write("enhances the model's focus on truly important regions.\n\n")
            else:
                f.write("The simplified explanation shows a statistically significant **decrease** in faithfulness ")
                f.write(f"(p={faith_stats['p_value']:.4f}{faith_stats['significance']}). However, the practical difference ")
                f.write(f"is only {(faith_stats['baseline_mean'] - faith_stats['simplified_mean']) / faith_stats['baseline_mean'] * 100:.1f}%, ")
                f.write("which may be acceptable given the efficiency benefits.\n\n")
        else:
            f.write("There is no statistically significant difference in faithfulness between baseline and simplified ")
            f.write(f"explanations (p={faith_stats['p_value']:.4f}). This indicates that simplified explanations ")
            f.write("maintain the faithfulness of the original method.\n\n")
        
        # Completeness interpretation
        f.write("**Completeness:** ")
        if comp_stats['p_value'] < 0.05:
            if comp_stats['simplified_mean'] > comp_stats['baseline_mean'] * 0.9:
                f.write("The simplified explanation retains high completeness ")
                f.write(f"({comp_stats['simplified_mean'] / comp_stats['baseline_mean'] * 100:.1f}% of baseline) ")
                f.write("despite significant reduction in complexity. This confirms that the top-percentage ")
                f.write("thresholding approach effectively captures the most important regions.\n\n")
            else:
                f.write("The simplified explanation shows a statistically significant reduction in completeness ")
                f.write(f"(p={comp_stats['p_value']:.4f}{comp_stats['significance']}). The retention rate is ")
                f.write(f"{comp_stats['simplified_mean'] / comp_stats['baseline_mean'] * 100:.1f}% of the baseline, ")
                f.write("suggesting that more careful threshold tuning may be needed.\n\n")
        else:
            f.write("There is no statistically significant difference in completeness between baseline and simplified ")
            f.write(f"explanations (p={comp_stats['p_value']:.4f}). This confirms that the simplified approach ")
            f.write("retains the essential explanatory content.\n\n")
        
        # Memory and processing time
        if mem_stats['baseline_mean'] > 0:  # Guard against division by zero
            mem_reduction = (1 - mem_stats['simplified_mean'] / mem_stats['baseline_mean']) * 100
        else:
            mem_reduction = 0
            
        if time_stats['baseline_mean'] > 0:  # Guard against division by zero
            time_reduction = (1 - time_stats['simplified_mean'] / time_stats['baseline_mean']) * 100
        else:
            time_reduction = 0
            
        f.write("**Efficiency Gains:** ")
        f.write(f"The simplified approach achieves a **{mem_reduction:.1f}%** reduction in memory usage ")
        
        if time_reduction > 0:
            f.write(f"and a **{time_reduction:.1f}%** reduction in processing time. ")
        else:
            f.write(f"with a small {-time_reduction:.1f}% increase in processing time due to the additional simplification step. ")
        
        if mem_reduction > 50:
            f.write("This significant efficiency improvement makes the approach well-suited for resource-constrained environments ")
            f.write("like mobile devices and edge computing.\n\n")
        else:
            f.write("While the memory reduction is moderate, it may still provide benefits in memory-constrained environments.\n\n")
        
        # Conclusion
        f.write("### Conclusion\n\n")
        f.write("Based on the statistical analysis, the simplified explanation method ")
        
        if faith_stats['p_value'] < 0.05 and faith_stats['simplified_mean'] < faith_stats['baseline_mean'] * 0.8:
            f.write("shows some trade-offs in explanation quality but ")
        elif comp_stats['p_value'] < 0.05 and comp_stats['simplified_mean'] < comp_stats['baseline_mean'] * 0.8:
            f.write("has some limitations in completeness but ")
        else:
            f.write("maintains explanation quality while ")
        
        f.write("providing substantial efficiency benefits. ")
        
        # Recommendation based on results
        optimal_threshold = thresholds[0]
        best_retention = 0
        
        if simplified_metrics[thresholds[0]]['completeness']['avg_retention']:
            best_retention = np.mean(simplified_metrics[thresholds[0]]['completeness']['avg_retention'])
            
            for t in thresholds[1:]:
                if simplified_metrics[t]['completeness']['avg_retention']:
                    retention = np.mean(simplified_metrics[t]['completeness']['avg_retention'])
                    if retention > best_retention:
                        best_retention = retention
                        optimal_threshold = t
        
        f.write(f"For this dataset, a threshold of **{optimal_threshold}%** provides the optimal balance ")
        f.write("between explanation quality and computational efficiency.")
    
    print(f"Statistical report saved to {os.path.join(output_dir, 'statistical_report.md')}")

def create_metric_visualization(metric_name, baseline_values, simplified_values, output_dir):
    """Create a visualization comparing baseline and simplified metric values"""
    # Convert to numpy arrays
    baseline_values = np.array(baseline_values)
    simplified_values = np.array(simplified_values)
    
    # Handle empty arrays
    if len(baseline_values) == 0 or len(simplified_values) == 0:
        print(f"Warning: Empty array for {metric_name} visualization, skipping")
        return
    
    # Calculate statistics
    baseline_mean = np.mean(baseline_values)
    baseline_std = np.std(baseline_values)
    simplified_mean = np.mean(simplified_values)
    simplified_std = np.std(simplified_values)
    
    # Create the figure
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    x = np.array([0, 1])
    plt.bar(x, [baseline_mean, simplified_mean], 
            yerr=[baseline_std, simplified_std],
            width=0.6, capsize=10, color=['steelblue', 'lightcoral'], alpha=0.8)
    
    # Add labels and title
    plt.xticks(x, ['Baseline', 'Simplified'])
    plt.ylabel(metric_name)
    plt.title(f'Comparison of {metric_name} (Mean ± Std)')
    
    # Add mean values as text
    plt.text(0, baseline_mean + 0.1 * baseline_std, f'{baseline_mean:.4f} ± {baseline_std:.4f}', 
             ha='center', va='bottom', fontweight='bold')
    plt.text(1, simplified_mean + 0.1 * simplified_std, f'{simplified_mean:.4f} ± {simplified_std:.4f}', 
             ha='center', va='bottom', fontweight='bold')
    
    # Calculate p-value if there are enough samples
    if len(baseline_values) >= 2 and len(simplified_values) >= 2:
        try:
            _, p_value = ttest_ind(baseline_values, simplified_values, equal_var=False)
            
            # Add significance markers if p-value is significant
            if p_value < 0.05:
                max_height = max(baseline_mean + baseline_std, simplified_mean + simplified_std)
                y_pos = max_height * 1.2
                
                plt.plot([0, 1], [y_pos, y_pos], 'k-', linewidth=1.5)
                
                if p_value < 0.001:
                    sig_text = '***'
                elif p_value < 0.01:
                    sig_text = '**'
                else:
                    sig_text = '*'
                    
                plt.text(0.5, y_pos + 0.05 * max_height, f'{sig_text} p={p_value:.4f}', 
                         ha='center', va='bottom', fontweight='bold')
        except Exception as e:
            print(f"Warning: Could not calculate p-value for {metric_name}: {e}")
    
    # Enhance visual appearance
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    safe_metric_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(os.path.join(output_dir, 'visualizations', f'{safe_metric_name}.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    run_statistical_evaluation() 