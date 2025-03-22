"""
Sample Evaluation Script

This script demonstrates the improved completeness metrics for
lightweight explainability, showing how the new implementation
more accurately reflects the activation retention.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from evaluation_metrics import ExplanationEvaluator
from lightweight_explainability import ExplainableModel, simplify_cam, show_heatmap

def run_sample_evaluation():
    """Run a sample evaluation on test images"""
    print("Running sample evaluation with improved completeness metrics...")
    
    # Create directory for results
    output_dir = os.path.join("results", "sample_evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = ExplainableModel(model_name="mobilenet_v2")
    
    # Initialize evaluator
    evaluator = ExplanationEvaluator(model)
    
    # Find sample images 
    search_dirs = ["sample_images", "static/uploads"]
    img_paths = []
    for dir_path in search_dirs:
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_paths.append(os.path.join(dir_path, file))
    
    # If no images found, exit
    if not img_paths:
        print("Error: No sample images found. Please add some images to sample_images/ directory")
        return
    
    # Select a few images for demonstration
    if len(img_paths) > 5:
        import random
        img_paths = random.sample(img_paths, 5)
    
    # Prepare results
    all_results = {}
    threshold_comparison = {
        "thresholds": [5, 10, 15, 20],
        "retention_rates": {img_path: [] for img_path in img_paths}
    }
    
    # Evaluate each image
    for img_path in img_paths:
        print(f"Evaluating {os.path.basename(img_path)}...")
        
        # Preprocess image
        img_tensor, img = model.preprocess_image(img_path)
        
        # Generate CAM
        cam = model.generate_gradcam(img_tensor)
        
        # Run completeness evaluation
        completeness_results = evaluator.evaluate_completeness(img_tensor, cam)
        
        # Store results
        all_results[img_path] = completeness_results
        
        # Add to threshold comparison
        threshold_comparison["retention_rates"][img_path] = completeness_results["retention_rates"]
        
        # Create visualizations of original and simplified CAMs
        plt.figure(figsize=(12, 8))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')
        
        # Original CAM
        plt.subplot(2, 3, 2)
        plt.imshow(show_heatmap(cam, img))
        plt.title("Full Grad-CAM")
        plt.axis('off')
        
        # Add simplified CAMs at different thresholds
        for i, threshold in enumerate([5, 10, 20]):
            simplified = simplify_cam(cam, threshold)
            plt.subplot(2, 3, i+3)
            plt.imshow(show_heatmap(simplified, img))
            plt.title(f"Top {threshold}% (Retention: {completeness_results['retention_rates'][i]:.2f})")
            plt.axis('off')
        
        # Save visualization
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_visualizations.png"))
        plt.close()
    
    # Create retention rate comparison chart
    plt.figure(figsize=(10, 6))
    for img_path, rates in threshold_comparison["retention_rates"].items():
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        plt.plot(threshold_comparison["thresholds"], rates, 'o-', label=base_name)
    
    plt.xlabel("Simplification Threshold (%)")
    plt.ylabel("Information Retention")
    plt.title("Completeness: Information Retention vs. Simplification")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "retention_comparison.png"))
    plt.close()
    
    # Calculate average AUC
    avg_auc = np.mean([results["auc"] for results in all_results.values()])
    
    # Create summary
    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write("# Improved Completeness Evaluation Results\n\n")
        f.write("## Summary\n\n")
        f.write(f"Average Completeness (AUC): **{avg_auc:.3f}**\n\n")
        f.write("### Explanation\n\n")
        f.write("The improved completeness metric now better reflects the inherent completeness of simplified explanations\n")
        f.write("by directly measuring what percentage of 'activation energy' is preserved during simplification.\n\n")
        
        f.write("### Per-Image Results\n\n")
        f.write("| Image | AUC | Average Retention |\n")
        f.write("|-------|-----|------------------|\n")
        
        for img_path, results in all_results.items():
            base_name = os.path.basename(img_path)
            avg_retention = np.mean(results["retention_rates"])
            f.write(f"| {base_name} | {results['auc']:.3f} | {avg_retention:.3f} |\n")
        
        f.write("\n\n")
        f.write("![Retention Comparison](retention_comparison.png)\n\n")
        
        f.write("## Visualizations\n\n")
        for img_path in all_results.keys():
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            f.write(f"### {base_name}\n\n")
            f.write(f"![{base_name} Visualizations]({base_name}_visualizations.png)\n\n")
    
    print(f"Evaluation complete! Results saved to {output_dir}")
    print(f"Open {os.path.join(output_dir, 'summary.md')} to view the results")

if __name__ == "__main__":
    run_sample_evaluation() 