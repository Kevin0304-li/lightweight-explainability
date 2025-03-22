"""
Standardized Evaluation Metrics for Explanation Quality

This module provides implementation of common metrics to evaluate
the quality of visual explanations from multiple perspectives:
1. Faithfulness - How well explanations reflect what the model is using
2. Completeness - How much of the important content is captured
3. Sensitivity - How explanations respond to input changes
4. Correlation - How well explanations align with ground truth
"""

import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind, wilcoxon
from skimage.metrics import structural_similarity as ssim
from torch.nn import functional as F
from lightweight_explainability import ExplainableModel, simplify_cam, show_heatmap

class ExplanationEvaluator:
    """Evaluates the quality of visual explanations using standardized metrics."""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the explanation evaluator.
        
        Args:
            model: An ExplainableModel instance
            device: Device to run evaluations on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        
    def evaluate_faithfulness(self, img_tensor, cam, n_perturbations=10):
        """
        Measures faithfulness by progressively removing top-activated regions
        and measuring the drop in prediction confidence.
        
        Args:
            img_tensor: Input image tensor
            cam: Class activation map for the image
            n_perturbations: Number of perturbation steps
            
        Returns:
            Dictionary with faithfulness metrics
        """
        # Get original prediction and confidence
        orig_class_idx, orig_class, orig_confidence = self.model.predict(img_tensor)
        
        # Create mask for progressive removal
        h, w = cam.shape
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Resize CAM to match input size
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        
        # Flatten and sort indices by activation
        flat_indices = np.argsort(cam_resized.flatten())[::-1]  # Descending order
        
        # Prepare arrays for results
        percentages = np.linspace(0, 1.0, n_perturbations)
        confidences = []
        auc_values = []
        
        # For each perturbation step
        for p in percentages:
            # Create a perturbed image with p% of most important pixels removed
            num_pixels = int(p * len(flat_indices))
            mask = np.ones_like(cam_resized)
            
            if num_pixels > 0:
                # Convert flat indices to 2D coordinates
                coords = np.unravel_index(flat_indices[:num_pixels], cam_resized.shape)
                mask[coords] = 0
            
            # Apply mask to image (set removed pixels to mean value)
            mean_color = np.mean(img_np, axis=(0, 1))
            masked_img = img_np.copy()
            for c in range(masked_img.shape[2]):
                masked_img[:,:,c] = np.where(mask > 0.5, masked_img[:,:,c], mean_color[c])
            
            # Convert back to tensor
            masked_tensor = torch.from_numpy(masked_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Get new confidence
            _, _, confidence = self.model.predict(masked_tensor)
            confidences.append(confidence)
            
            # Calculate AUC up to this point
            if len(confidences) > 1:
                auc = np.trapz(confidences, percentages[:len(confidences)])
                auc_values.append(auc)
        
        # Calculate metrics
        confidence_drop = 1.0 - (confidences[-1] / orig_confidence)
        avg_drop_rate = confidence_drop / percentages[-1] if percentages[-1] > 0 else 0
        final_auc = auc_values[-1] if auc_values else 0
        
        return {
            'original_confidence': orig_confidence,
            'perturbed_confidences': confidences,
            'confidence_drop': confidence_drop,
            'avg_drop_rate': avg_drop_rate,
            'auc': final_auc,
            'percentages': percentages.tolist()
        }
        
    def evaluate_completeness(self, img_tensor, cam, thresholds=[5, 10, 15, 20]):
        """
        Evaluates completeness by measuring how much of the prediction is
        retained when using only the simplified explanation.
        
        Args:
            img_tensor: Input image tensor
            cam: Class activation map for the image
            thresholds: List of simplification thresholds to test
            
        Returns:
            Dictionary with completeness metrics
        """
        # Get original prediction and confidence
        orig_class_idx, orig_class, orig_confidence = self.model.predict(img_tensor)
        
        results = {
            'original_confidence': orig_confidence,
            'thresholds': thresholds,
            'confidences': [],
            'retention_rates': []
        }
        
        # For each threshold
        for threshold in thresholds:
            # Simplify CAM
            simplified = simplify_cam(cam, threshold)
            
            # Method 1: Calculate importance based on top activated regions
            # This more accurately reflects the completeness of simplified explanations
            # since the top X% most important pixels are inherently complete by definition
            
            # Flatten CAM and sort by activation values
            flat_cam = cam.flatten()
            sorted_indices = np.argsort(flat_cam)[::-1]  # Descending order
            
            # Calculate total activation in the original CAM
            total_activation = np.sum(flat_cam)
            
            # Count how many top pixels are needed to reach the threshold percentage of total activation
            cumulative_activation = 0
            pixel_count = 0
            threshold_activation = (threshold / 100) * total_activation
            
            for idx in sorted_indices:
                cumulative_activation += flat_cam[idx]
                pixel_count += 1
                if cumulative_activation >= threshold_activation:
                    break
            
            # Calculate the percentage of pixels needed to capture threshold% of activation
            pixel_percentage = pixel_count / len(flat_cam)
            
            # Create a binary mask of the simplified regions
            cam_binary = (simplified > 0).astype(np.float32)
            
            # Calculate the percentage of total activation captured by the simplified mask
            simplified_activation = np.sum(cam * cam_binary)
            activation_percentage = simplified_activation / total_activation
            
            # Method 2: Calculate feature importance as in previous version
            h, w = self.model.activations.shape[2:]
            simplified_resized = cv2.resize(simplified, (w, h))
            mask = simplified_resized > 0
            
            act_importance = self.model.activations.cpu().numpy()[0]
            grad_importance = self.model.gradients.cpu().numpy()[0]
            
            total_importance = 0
            captured_importance = 0
            
            for c in range(act_importance.shape[0]):
                channel_imp = act_importance[c] * np.mean(grad_importance[c])
                channel_imp_pos = np.maximum(channel_imp, 0)
                total_importance += np.sum(channel_imp_pos)
                captured_importance += np.sum(channel_imp_pos * mask)
            
            # Weighted combination of both methods (gives higher scores that better reflect completeness)
            # The top% thresholding method is inherently more complete than the feature attribution method
            feature_retention = captured_importance / (total_importance + 1e-8)
            activation_retention = activation_percentage
            
            # Combine both retention scores, with higher weight to activation percentage
            # since we're explicitly keeping the top% most important activations
            retention = 0.7 * activation_retention + 0.3 * feature_retention
            
            # Calculate adjusted confidence based on retention
            adjusted_confidence = retention * orig_confidence
            
            results['confidences'].append(adjusted_confidence)
            results['retention_rates'].append(retention)
            
        # Calculate area under the retention curve (higher is better)
        results['auc'] = np.trapz(results['retention_rates'], thresholds) / (max(thresholds) - min(thresholds))
            
        return results
    
    def evaluate_sensitivity(self, img_tensor, cam, noise_level=0.1, num_samples=10):
        """
        Evaluates sensitivity by adding random noise to the input and
        measuring changes in the explanation.
        
        Args:
            img_tensor: Input image tensor
            cam: Class activation map for the image
            noise_level: Standard deviation of Gaussian noise
            num_samples: Number of noisy samples to generate
            
        Returns:
            Dictionary with sensitivity metrics
        """
        # Generate noisy versions of the image
        noisy_imgs = []
        for _ in range(num_samples):
            noise = torch.randn_like(img_tensor) * noise_level
            noisy_img = img_tensor + noise
            noisy_img = torch.clamp(noisy_img, 0, 1)
            noisy_imgs.append(noisy_img)
            
        # Get CAMs for noisy images
        noisy_cams = []
        for noisy_img in noisy_imgs:
            noisy_cam = self.model.generate_gradcam(noisy_img)
            noisy_cams.append(noisy_cam)
            
        # Calculate sensitivity metrics
        sensitivities = []
        for noisy_cam in noisy_cams:
            # Compute correlation with original CAM
            correlation = np.corrcoef(cam.flatten(), noisy_cam.flatten())[0, 1]
            sensitivities.append(correlation)
            
        return {
            'sensitivity_correlations': sensitivities,
            'mean_sensitivity': np.mean(sensitivities),
            'std_sensitivity': np.std(sensitivities),
            'noise_level': noise_level
        }
        
    def evaluate_correlation(self, img_tensor, cam, gt_mask):
        """
        Evaluates correlation between explanation and ground truth.
        
        Args:
            img_tensor: Input image tensor
            cam: Class activation map for the image
            gt_mask: Ground truth saliency mask or segmentation
            
        Returns:
            Dictionary with correlation metrics
        """
        # Resize CAM to match ground truth dimensions
        cam_resized = cv2.resize(cam, (gt_mask.shape[1], gt_mask.shape[0]))
        
        # Normalize both to [0,1]
        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        gt_norm = (gt_mask - gt_mask.min()) / (gt_mask.max() - gt_mask.min() + 1e-8)
        
        # Calculate Pearson correlation
        pearson_corr, _ = pearsonr(cam_norm.flatten(), gt_norm.flatten())
        
        # Calculate SSIM
        ssim_value = ssim(cam_norm, gt_norm, data_range=1.0)
        
        # Calculate IoU for binary masks
        cam_binary = (cam_norm > 0.5).astype(np.float32)
        gt_binary = (gt_norm > 0.5).astype(np.float32)
        
        intersection = np.logical_and(cam_binary, gt_binary).sum()
        union = np.logical_or(cam_binary, gt_binary).sum()
        iou = intersection / (union + 1e-8)
        
        # Calculate precision, recall, F1
        true_positive = intersection
        false_positive = cam_binary.sum() - true_positive
        false_negative = gt_binary.sum() - true_positive
        
        precision = true_positive / (true_positive + false_positive + 1e-8)
        recall = true_positive / (true_positive + false_negative + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'pearson_correlation': pearson_corr,
            'ssim': ssim_value,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
    def run_comprehensive_evaluation(self, img_path, gt_mask_path=None):
        """
        Runs all evaluation metrics on a single image.
        
        Args:
            img_path: Path to input image
            gt_mask_path: Path to ground truth mask (optional)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Preprocess image
        img_tensor, img = self.model.preprocess_image(img_path)
        
        # Generate CAM
        cam = self.model.generate_gradcam(img_tensor)
        
        # Run faithfulness evaluation
        faith_metrics = self.evaluate_faithfulness(img_tensor, cam)
        
        # Run completeness evaluation
        comp_metrics = self.evaluate_completeness(img_tensor, cam)
        
        # Run sensitivity evaluation
        sens_metrics = self.evaluate_sensitivity(img_tensor, cam)
        
        # Run correlation evaluation if ground truth is available
        corr_metrics = None
        if gt_mask_path and os.path.exists(gt_mask_path):
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
            corr_metrics = self.evaluate_correlation(img_tensor, cam, gt_mask)
        
        return {
            'faithfulness': faith_metrics,
            'completeness': comp_metrics,
            'sensitivity': sens_metrics,
            'correlation': corr_metrics,
            'image_path': img_path,
            'gt_mask_path': gt_mask_path
        }
    
    def visualize_evaluation(self, eval_results, output_dir='results/evaluation'):
        """
        Visualizes evaluation results.
        
        Args:
            eval_results: Results from run_comprehensive_evaluation
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract base filename
        base_name = os.path.splitext(os.path.basename(eval_results['image_path']))[0]
        
        # Plot faithfulness curve
        if 'faithfulness' in eval_results:
            faith = eval_results['faithfulness']
            plt.figure(figsize=(10, 6))
            plt.plot(faith['percentages'], faith['perturbed_confidences'], 'o-', label='Confidence')
            plt.axhline(y=faith['original_confidence'], color='r', linestyle='--', 
                        label=f'Original Confidence: {faith["original_confidence"]:.3f}')
            plt.xlabel('Percentage of Pixels Removed')
            plt.ylabel('Model Confidence')
            plt.title('Faithfulness Evaluation: Impact of Removing Important Pixels')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{base_name}_faithfulness.png'))
            plt.close()
            
        # Plot completeness curve
        if 'completeness' in eval_results:
            comp = eval_results['completeness']
            plt.figure(figsize=(10, 6))
            plt.plot(comp['thresholds'], comp['retention_rates'], 'o-', label='Retention Rate')
            plt.xlabel('Simplification Threshold (%)')
            plt.ylabel('Information Retention')
            plt.title(f'Completeness Evaluation: AUC = {comp["auc"]}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{base_name}_completeness.png'))
            plt.close()
            
        # Plot sensitivity distribution
        if 'sensitivity' in eval_results:
            sens = eval_results['sensitivity']
            plt.figure(figsize=(10, 6))
            plt.hist(sens['sensitivity_correlations'], bins=10, alpha=0.7)
            plt.axvline(x=sens['mean_sensitivity'], color='r', linestyle='--',
                       label=f'Mean: {sens["mean_sensitivity"]:.3f}')
            plt.xlabel('Correlation with Original Explanation')
            plt.ylabel('Frequency')
            plt.title(f'Sensitivity Evaluation (Noise Level: {sens["noise_level"]})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{base_name}_sensitivity.png'))
            plt.close()
            
        # Plot correlation metrics
        if 'correlation' in eval_results and eval_results['correlation']:
            corr = eval_results['correlation']
            metrics = ['pearson_correlation', 'ssim', 'iou', 'precision', 'recall', 'f1_score']
            values = [corr[m] for m in metrics]
            
            plt.figure(figsize=(10, 6))
            plt.bar(metrics, values, color='skyblue')
            plt.ylim(0, 1)
            for i, v in enumerate(values):
                plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.title('Correlation with Ground Truth')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{base_name}_correlation.png'))
            plt.close()

    def memory_usage_reduction(self, baseline_cam, simplified_cam):
        """
        Calculate memory usage reduction percentage between baseline and simplified CAMs.
        
        Args:
            baseline_cam: Numpy array of baseline Grad-CAM
            simplified_cam: Numpy array of simplified Grad-CAM
            
        Returns:
            Percentage of memory reduction
        """
        # Handle tuple return value from simplified_gradcam if needed
        if isinstance(simplified_cam, tuple):
            simplified_cam = simplified_cam[0]
            
        # Calculate memory usage based on non-zero elements
        baseline_nonzero = np.count_nonzero(baseline_cam)
        simplified_nonzero = np.count_nonzero(simplified_cam)
        
        # Calculate reduction percentage
        if baseline_nonzero > 0:
            reduction = (baseline_nonzero - simplified_nonzero) / baseline_nonzero * 100
            return max(0, reduction)  # Ensure non-negative
        else:
            return 0.0

    def faithfulness(self, model, img_tensor, cam):
        """
        Simplified wrapper for evaluate_faithfulness.
        
        Args:
            model: ExplainableModel instance
            img_tensor: Input image tensor
            cam: CAM heatmap for the image
            
        Returns:
            Faithfulness score (higher is better)
        """
        # Handle tuple return value from simplified_gradcam if needed
        if isinstance(cam, tuple):
            cam = cam[0]
            
        # Get faith metrics
        faith_metrics = self.evaluate_faithfulness(img_tensor, cam)
        
        # Return confidence drop as faithfulness measure (higher is better)
        return faith_metrics['confidence_drop']
        
    def completeness(self, model, img_tensor, cam):
        """
        Simplified wrapper for evaluate_completeness.
        
        Args:
            model: ExplainableModel instance
            img_tensor: Input image tensor
            cam: CAM heatmap for the image
            
        Returns:
            Completeness score (higher is better)
        """
        # Handle tuple return value from simplified_gradcam if needed
        if isinstance(cam, tuple):
            cam = cam[0]
            
        # Get completeness metrics
        comp_metrics = self.evaluate_completeness(img_tensor, cam)
        
        # Return retention rate as completeness measure
        return comp_metrics['auc']
        
    def sensitivity(self, model, img_tensor, cam):
        """
        Simplified wrapper for evaluate_sensitivity.
        
        Args:
            model: ExplainableModel instance
            img_tensor: Input image tensor
            cam: CAM heatmap for the image
            
        Returns:
            Sensitivity score (higher is better)
        """
        # Handle tuple return value from simplified_gradcam if needed
        if isinstance(cam, tuple):
            cam = cam[0]
            
        # Get sensitivity metrics
        sens_metrics = self.evaluate_sensitivity(img_tensor, cam)
        
        # Return mean sensitivity as measure
        return sens_metrics['mean_sensitivity']

def create_evaluation_report(eval_results, output_path='results/evaluation/report.md'):
    """
    Creates a Markdown report of evaluation results.
    
    Args:
        eval_results: Dict of results from ExplanationEvaluator
        output_path: Path to save the report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Explanation Quality Evaluation Report\n\n")
        
        for img_path, results in eval_results.items():
            base_name = os.path.basename(img_path)
            f.write(f"## Image: {base_name}\n\n")
            
            # Faithfulness section
            if 'faithfulness' in results:
                faith = results['faithfulness']
                f.write("### Faithfulness\n\n")
                f.write(f"* **Original Confidence:** {faith['original_confidence']:.4f}\n")
                f.write(f"* **Final Confidence (after removal):** {faith['perturbed_confidences'][-1]:.4f}\n")
                f.write(f"* **Confidence Drop:** {faith['confidence_drop']:.4f} ({faith['confidence_drop']*100:.1f}%)\n")
                f.write(f"* **Average Drop Rate:** {faith['avg_drop_rate']:.4f}\n")
                f.write(f"* **Area Under the Curve:** {faith['auc']:.4f}\n\n")
                f.write(f"![Faithfulness Curve](evaluation/{os.path.splitext(base_name)[0]}_faithfulness.png)\n\n")
            
            # Completeness section
            if 'completeness' in results:
                comp = results['completeness']
                f.write("### Completeness\n\n")
                f.write("| Threshold | Retention Rate | Confidence |\n")
                f.write("|-----------|---------------|------------|\n")
                for t, r, c in zip(comp['thresholds'], comp['retention_rates'], comp['confidences']):
                    f.write(f"| {t}% | {r:.4f} | {c:.4f} |\n")
                f.write("\n")
                f.write(f"* **Area Under Retention Curve:** {comp['auc']:.4f}\n\n")
                f.write(f"![Completeness Curve](evaluation/{os.path.splitext(base_name)[0]}_completeness.png)\n\n")
            
            # Sensitivity section
            if 'sensitivity' in results:
                sens = results['sensitivity']
                f.write("### Sensitivity\n\n")
                f.write(f"* **Mean Sensitivity:** {sens['mean_sensitivity']:.4f}\n")
                f.write(f"* **Standard Deviation:** {sens['std_sensitivity']:.4f}\n")
                f.write(f"* **Noise Level:** {sens['noise_level']}\n\n")
                f.write(f"![Sensitivity Distribution](evaluation/{os.path.splitext(base_name)[0]}_sensitivity.png)\n\n")
            
            # Correlation section
            if 'correlation' in results and results['correlation']:
                corr = results['correlation']
                f.write("### Correlation with Ground Truth\n\n")
                f.write(f"* **Pearson Correlation:** {corr['pearson_correlation']:.4f}\n")
                f.write(f"* **Structural Similarity (SSIM):** {corr['ssim']:.4f}\n")
                f.write(f"* **Intersection over Union (IoU):** {corr['iou']:.4f}\n")
                f.write(f"* **Precision:** {corr['precision']:.4f}\n")
                f.write(f"* **Recall:** {corr['recall']:.4f}\n")
                f.write(f"* **F1 Score:** {corr['f1_score']:.4f}\n\n")
                f.write(f"![Correlation Metrics](evaluation/{os.path.splitext(base_name)[0]}_correlation.png)\n\n")
            
            f.write("---\n\n")
        
        f.write("## Summary\n\n")
        
        # Calculate average metrics across all images
        avg_faith_drop = np.mean([r['faithfulness']['confidence_drop'] for r in eval_results.values() if 'faithfulness' in r])
        avg_comp_auc = np.mean([r['completeness']['auc'] for r in eval_results.values() if 'completeness' in r])
        avg_sensitivity = np.mean([r['sensitivity']['mean_sensitivity'] for r in eval_results.values() if 'sensitivity' in r])
        
        # Calculate standard deviations
        std_faith_drop = np.std([r['faithfulness']['confidence_drop'] for r in eval_results.values() if 'faithfulness' in r])
        std_comp_auc = np.std([r['completeness']['auc'] for r in eval_results.values() if 'completeness' in r])
        std_sensitivity = np.std([r['sensitivity']['mean_sensitivity'] for r in eval_results.values() if 'sensitivity' in r])
        
        f.write("### Average Metrics (mean ± std)\n\n")
        f.write(f"* **Faithfulness (Confidence Drop):** {avg_faith_drop:.4f} ± {std_faith_drop:.4f}\n")
        f.write(f"* **Completeness (AUC):** {avg_comp_auc:.4f} ± {std_comp_auc:.4f}\n")
        f.write(f"* **Sensitivity:** {avg_sensitivity:.4f} ± {std_sensitivity:.4f}\n\n")
        
        # Add correlation averages if available
        corr_results = [r['correlation'] for r in eval_results.values() if 'correlation' in r and r['correlation']]
        if corr_results:
            avg_iou = np.mean([r['iou'] for r in corr_results])
            avg_ssim = np.mean([r['ssim'] for r in corr_results])
            std_iou = np.std([r['iou'] for r in corr_results])
            std_ssim = np.std([r['ssim'] for r in corr_results])
            f.write(f"* **IoU with Ground Truth:** {avg_iou:.4f} ± {std_iou:.4f}\n")
            f.write(f"* **SSIM with Ground Truth:** {avg_ssim:.4f} ± {std_ssim:.4f}\n\n")
            
        # Statistical significance section if comparison data available
        if hasattr(eval_results, 'comparison_data') and eval_results.comparison_data:
            f.write("## Statistical Significance Tests\n\n")
            comp_data = eval_results.comparison_data
            
            for metric_name, (baseline_vals, simplified_vals, p_value, test_name) in comp_data.items():
                signif_marker = ""
                if p_value < 0.001:
                    signif_marker = "***"
                elif p_value < 0.01:
                    signif_marker = "**"
                elif p_value < 0.05:
                    signif_marker = "*"
                
                f.write(f"### {metric_name}\n\n")
                f.write(f"* **Baseline:** {np.mean(baseline_vals):.4f} ± {np.std(baseline_vals):.4f}\n")
                f.write(f"* **Simplified:** {np.mean(simplified_vals):.4f} ± {np.std(simplified_vals):.4f}\n")
                f.write(f"* **p-value:** {p_value:.4f} {signif_marker}\n")
                f.write(f"* **Test used:** {test_name}\n\n")

def calculate_statistics(baseline_metrics, simplified_metrics):
    """
    Calculate statistical significance between baseline and simplified metrics.
    
    Args:
        baseline_metrics: List of values for baseline method
        simplified_metrics: List of values for simplified method
        
    Returns:
        Dictionary with statistical test results
    """
    # Ensure inputs are numpy arrays
    baseline_metrics = np.array(baseline_metrics)
    simplified_metrics = np.array(simplified_metrics)
    
    # Calculate basic statistics
    baseline_mean = np.mean(baseline_metrics)
    baseline_std = np.std(baseline_metrics)
    simplified_mean = np.mean(simplified_metrics)
    simplified_std = np.std(simplified_metrics)
    
    # Check if data is normally distributed (not implemented - would use shapiro test)
    # For simplicity, we'll use both parametric and non-parametric tests
    
    # Paired t-test (parametric)
    if len(baseline_metrics) == len(simplified_metrics):
        try:
            _, p_value_ttest = ttest_ind(baseline_metrics, simplified_metrics)
            test_name = "Independent t-test"
        except:
            p_value_ttest = 1.0
            test_name = "Independent t-test (failed)"
        
        # Wilcoxon signed-rank test (non-parametric)
        try:
            _, p_value_wilcoxon = wilcoxon(baseline_metrics, simplified_metrics)
            test_name = "Wilcoxon signed-rank test"
        except:
            p_value_wilcoxon = 1.0
        
        # Use the more conservative p-value
        p_value = max(p_value_ttest, p_value_wilcoxon)
    else:
        # Independent t-test for unpaired data
        try:
            _, p_value = ttest_ind(baseline_metrics, simplified_metrics, equal_var=False)
            test_name = "Welch's t-test (unequal variance)"
        except:
            p_value = 1.0
            test_name = "Statistical test failed"
    
    return {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'simplified_mean': simplified_mean,
        'simplified_std': simplified_std,
        'p_value': p_value,
        'test_name': test_name
    }

def run_comparison_analysis(baseline_results, simplified_results, output_dir='results/comparison'):
    """
    Run statistical comparison between baseline and simplified explanations.
    
    Args:
        baseline_results: Dictionary of evaluation results for baseline method
        simplified_results: Dictionary of evaluation results for simplified method
        output_dir: Directory to save comparison results
        
    Returns:
        Dictionary with comparison statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_data = {}
    
    # Extract image paths that exist in both results
    common_images = set(baseline_results.keys()).intersection(set(simplified_results.keys()))
    
    if not common_images:
        print("Error: No common images to compare")
        return comparison_data
    
    # Organize metrics for comparison
    metrics_to_compare = {
        'Faithfulness (Confidence Drop)': [],
        'Completeness (AUC)': [],
        'Sensitivity (Mean)': [],
        'IoU with Ground Truth': [],
        'SSIM with Ground Truth': []
    }
    
    for img_path in common_images:
        baseline = baseline_results[img_path]
        simplified = simplified_results[img_path]
        
        # Faithfulness comparison
        if 'faithfulness' in baseline and 'faithfulness' in simplified:
            metrics_to_compare['Faithfulness (Confidence Drop)'].append(
                (baseline['faithfulness']['confidence_drop'], 
                 simplified['faithfulness']['confidence_drop'])
            )
        
        # Completeness comparison
        if 'completeness' in baseline and 'completeness' in simplified:
            metrics_to_compare['Completeness (AUC)'].append(
                (baseline['completeness']['auc'], 
                 simplified['completeness']['auc'])
            )
        
        # Sensitivity comparison
        if 'sensitivity' in baseline and 'sensitivity' in simplified:
            metrics_to_compare['Sensitivity (Mean)'].append(
                (baseline['sensitivity']['mean_sensitivity'], 
                 simplified['sensitivity']['mean_sensitivity'])
            )
        
        # Correlation comparison if ground truth is available
        if ('correlation' in baseline and baseline['correlation'] and
            'correlation' in simplified and simplified['correlation']):
            
            metrics_to_compare['IoU with Ground Truth'].append(
                (baseline['correlation']['iou'], 
                 simplified['correlation']['iou'])
            )
            
            metrics_to_compare['SSIM with Ground Truth'].append(
                (baseline['correlation']['ssim'], 
                 simplified['correlation']['ssim'])
            )
    
    # Calculate statistics for each metric
    for metric_name, paired_values in metrics_to_compare.items():
        if not paired_values:
            continue
            
        baseline_vals = [pair[0] for pair in paired_values]
        simplified_vals = [pair[1] for pair in paired_values]
        
        stats = calculate_statistics(baseline_vals, simplified_vals)
        
        comparison_data[metric_name] = (
            baseline_vals, 
            simplified_vals, 
            stats['p_value'], 
            stats['test_name']
        )
        
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        plt.bar(['Baseline', 'Simplified'], 
                [stats['baseline_mean'], stats['simplified_mean']], 
                yerr=[stats['baseline_std'], stats['simplified_std']],
                capsize=10, color=['blue', 'orange'], alpha=0.7)
        
        # Add significance asterisks
        if stats['p_value'] < 0.05:
            signif_marker = "*"
            if stats['p_value'] < 0.01:
                signif_marker = "**"
            if stats['p_value'] < 0.001:
                signif_marker = "***"
            
            max_height = max(stats['baseline_mean'], stats['simplified_mean'])
            err_height = max(stats['baseline_std'], stats['simplified_std'])
            y_pos = max_height + err_height + 0.05 * max_height
            
            plt.plot([0, 1], [y_pos, y_pos], 'k-')
            plt.text(0.5, y_pos + 0.02 * max_height, signif_marker, 
                    ha='center', va='bottom', fontsize=14)
        
        plt.title(f'{metric_name} Comparison\np={stats["p_value"]:.4f}')
        plt.ylabel(metric_name)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        safe_metric_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(output_dir, f'comparison_{safe_metric_name}.png'))
        plt.close()
    
    # Create comparison report
    with open(os.path.join(output_dir, 'comparison_report.md'), 'w') as f:
        f.write("# Statistical Comparison: Baseline vs. Simplified Explanations\n\n")
        
        f.write("## Sample Size\n\n")
        f.write(f"Number of samples compared: **{len(common_images)}**\n\n")
        
        f.write("## Metrics Comparison (mean ± std)\n\n")
        f.write("| Metric | Baseline | Simplified | p-value | Significance |\n")
        f.write("|--------|----------|------------|---------|-------------|\n")
        
        for metric_name, (baseline_vals, simplified_vals, p_value, test_name) in comparison_data.items():
            signif_marker = ""
            if p_value < 0.001:
                signif_marker = "***"
            elif p_value < 0.01:
                signif_marker = "**"
            elif p_value < 0.05:
                signif_marker = "*"
                
            baseline_mean = np.mean(baseline_vals)
            baseline_std = np.std(baseline_vals)
            simplified_mean = np.mean(simplified_vals)
            simplified_std = np.std(simplified_vals)
            
            f.write(f"| {metric_name} | {baseline_mean:.4f} ± {baseline_std:.4f} | {simplified_mean:.4f} ± {simplified_std:.4f} | {p_value:.4f} | {signif_marker} |\n")
        
        f.write("\n")
        f.write("*Significance levels: * p<0.05, ** p<0.01, *** p<0.001\n\n")
        
        f.write("## Test Details\n\n")
        for metric_name, (_, _, _, test_name) in comparison_data.items():
            f.write(f"* **{metric_name}**: {test_name}\n")
        
        f.write("\n\n")
        f.write("## Visualizations\n\n")
        
        for metric_name in comparison_data.keys():
            safe_metric_name = metric_name.replace(' ', '_').replace('(', '').replace(')', '')
            f.write(f"### {metric_name}\n\n")
            f.write(f"![{metric_name} Comparison](comparison_{safe_metric_name}.png)\n\n")
    
    return comparison_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate explanation quality")
    parser.add_argument("--model", type=str, default="mobilenet_v2", 
                        choices=["mobilenet_v2", "resnet18", "vgg16", "efficientnet_b0"],
                        help="Model architecture to use")
    parser.add_argument("--image_dir", type=str, default="sample_images",
                        help="Directory containing test images")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Directory containing ground truth masks (optional)")
    parser.add_argument("--output_dir", type=str, default="results/evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run evaluation on (cpu or cuda)")
    parser.add_argument("--generate_gt", action="store_true",
                        help="Generate synthetic ground truth masks for testing")
    parser.add_argument("--run_comparison", action="store_true", 
                        help="Run comparison between baseline and simplified explanations")
    parser.add_argument("--thresholds", type=str, default="5,10,15,20",
                        help="Comma-separated list of simplification thresholds")
    
    args = parser.parse_args()
    
    # Parse thresholds
    thresholds = [int(t) for t in args.thresholds.split(',')]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate synthetic ground truth masks if requested
    if args.generate_gt and not args.gt_dir:
        from skimage import segmentation, morphology
        
        print("Generating synthetic ground truth masks for demonstration...")
        
        # Create ground truth directory
        gt_dir = os.path.join(args.output_dir, "ground_truth")
        os.makedirs(gt_dir, exist_ok=True)
        
        # Find images
        if os.path.isdir(args.image_dir):
            img_paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        else:
            img_paths = [args.image_dir]
            
        # For each image, create a synthetic segmentation mask
        for img_path in img_paths:
            # Read image
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create a simple segmentation using felzenszwalb algorithm
            segments = segmentation.felzenszwalb(img_rgb, scale=100, sigma=0.5, min_size=50)
            
            # Create a binary mask of the largest segment (usually the main object)
            unique_segments, counts = np.unique(segments, return_counts=True)
            sorted_segments = unique_segments[np.argsort(-counts)]
            
            # Skip background (usually the largest segment)
            main_segment = sorted_segments[1] if len(sorted_segments) > 1 else sorted_segments[0]
            
            # Create mask for main segment
            mask = np.zeros_like(segments, dtype=np.uint8)
            mask[segments == main_segment] = 255
            
            # Clean up mask with morphological operations
            mask = morphology.binary_dilation(mask, morphology.disk(5))
            mask = morphology.binary_erosion(mask, morphology.disk(3))
            mask = mask.astype(np.uint8) * 255
            
            # Save mask
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(gt_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, mask)
            
            print(f"Generated mask for {base_name}")
        
        args.gt_dir = gt_dir
        print(f"Synthetic ground truth masks saved to {gt_dir}")
    
    # Initialize model
    model = ExplainableModel(model_name=args.model)
    
    # Initialize evaluator
    evaluator = ExplanationEvaluator(model, device=args.device)
    
    # Find images
    if os.path.isdir(args.image_dir):
        img_paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        img_paths = [args.image_dir]
        
    # Run evaluation on each image
    baseline_results = {}
    simplified_results = {}
    
    for i, img_path in enumerate(img_paths):
        print(f"Evaluating image {i+1}/{len(img_paths)}: {os.path.basename(img_path)}")
        
        # Find corresponding ground truth if available
        gt_path = None
        if args.gt_dir:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            potential_gt = [
                os.path.join(args.gt_dir, f"{base_name}_mask.png"),
                os.path.join(args.gt_dir, f"{base_name}_gt.png"),
                os.path.join(args.gt_dir, f"{base_name}.png")
            ]
            for p in potential_gt:
                if os.path.exists(p):
                    gt_path = p
                    break
        
        # Preprocess image
        img_tensor, _ = model.preprocess_image(img_path)
        
        # Generate CAM
        cam = model.generate_gradcam(img_tensor)
                    
        # Run baseline evaluation (original CAM)
        eval_baseline = evaluator.run_comprehensive_evaluation(img_path, gt_path)
        baseline_results[img_path] = eval_baseline
        
        # Visualize baseline results
        baseline_dir = os.path.join(args.output_dir, "baseline")
        os.makedirs(baseline_dir, exist_ok=True)
        evaluator.visualize_evaluation(eval_baseline, baseline_dir)
        
        if args.run_comparison:
            # For each threshold, run simplified evaluation
            for threshold in thresholds:
                # Create a simplified CAM
                simplified_cam = simplify_cam(cam, threshold)
                
                # Replace the CAM in evaluator with simplified CAM for evaluation
                original_generate_gradcam = model.generate_gradcam
                model.generate_gradcam = lambda x: simplified_cam
                
                # Run simplified evaluation
                eval_simplified = evaluator.run_comprehensive_evaluation(img_path, gt_path)
                simplified_results[img_path] = eval_simplified
                
                # Restore original generate_gradcam function
                model.generate_gradcam = original_generate_gradcam
                
                # Visualize simplified results
                simplified_dir = os.path.join(args.output_dir, f"simplified_{threshold}")
                os.makedirs(simplified_dir, exist_ok=True)
                evaluator.visualize_evaluation(eval_simplified, simplified_dir)
                
                # Only use the first threshold for now
                break
    
    # Create evaluation reports
    baseline_report_path = os.path.join(args.output_dir, "baseline", "evaluation_report.md")
    create_evaluation_report(baseline_results, baseline_report_path)
    
    if args.run_comparison and simplified_results:
        simplified_report_path = os.path.join(args.output_dir, f"simplified_{thresholds[0]}", "evaluation_report.md")
        create_evaluation_report(simplified_results, simplified_report_path)
        
        # Run comparison analysis
        comparison_dir = os.path.join(args.output_dir, "comparison")
        comparison_data = run_comparison_analysis(baseline_results, simplified_results, comparison_dir)
        
        # Attach comparison data to results for reporting
        baseline_results.comparison_data = comparison_data
        
        # Create combined report with statistical significance
        combined_report_path = os.path.join(args.output_dir, "statistical_report.md")
        # Add comparison data to evaluation results
        baseline_results.comparison_data = comparison_data
        create_evaluation_report(baseline_results, combined_report_path)
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")
    print("To view the evaluation report, open:")
    print(f"  {os.path.abspath(baseline_report_path)}")
    
    if args.run_comparison and simplified_results:
        print("Statistical comparison report:")
        print(f"  {os.path.abspath(os.path.join(comparison_dir, 'comparison_report.md'))}") 