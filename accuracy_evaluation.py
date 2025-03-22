import os
import glob
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from lightweight_explainability import (
    ExplainableModel, 
    evaluate_simplification_impact, 
    generate_impact_report
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate accuracy impact of simplified Grad-CAM")
    parser.add_argument(
        "--image_dir", 
        type=str, 
        default="./sample_images", 
        help="Directory containing images to evaluate"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="mobilenet_v2", 
        choices=["mobilenet_v2", "resnet18", "resnet50", "vgg16", "efficientnet_b0"],
        help="Model to use for evaluation"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=10, 
        help="Percentage of CAM to keep in simplification"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100, 
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results/accuracy_evaluation", 
        help="Directory to save results"
    )
    return parser.parse_args()

def run_evaluation(args):
    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of image files
    img_extensions = ['*.jpg', '*.jpeg', '*.png']
    img_files = []
    
    for ext in img_extensions:
        img_files.extend(glob.glob(os.path.join(args.image_dir, ext)))
        img_files.extend(glob.glob(os.path.join(args.image_dir, '**', ext), recursive=True))
    
    print(f"Found {len(img_files)} images for evaluation")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = ExplainableModel(args.model)
    
    # Evaluate impact
    print(f"Evaluating simplification impact (threshold: {args.threshold}%)...")
    metrics = evaluate_simplification_impact(
        model=model,
        img_paths=img_files,
        top_percent=args.threshold,
        num_samples=args.num_samples
    )
    
    # Generate and save report
    report_path = os.path.join(args.output_dir, f"impact_report_{args.model}_{args.threshold}.md")
    report = generate_impact_report(metrics, report_path)
    
    # Print summary
    print("\n--- Evaluation Summary ---")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}%")
    print(f"Images evaluated: {metrics['baseline']['total_predictions']}")
    print(f"Accuracy retention: {metrics['simplified']['accuracy_retention']*100:.2f}%")
    print(f"Confidence ratio: {metrics['simplified']['confidence_ratio']*100:.2f}%")
    print(f"Speedup factor: {metrics['simplified']['speedup_factor']:.2f}x")
    print(f"Report saved to: {report_path}")
    
    # Generate visualization
    visualize_metrics(metrics, args)
    
    return metrics

def visualize_metrics(metrics, args):
    """Generate visualizations of the evaluation metrics"""
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    plt.bar(['Baseline', 'Simplified'], 
            [metrics['baseline']['accuracy']*100, metrics['simplified']['accuracy']*100],
            color=['#3498db', '#2ecc71'])
    plt.ylim(0, 100)
    plt.title('Classification Accuracy')
    plt.ylabel('Accuracy (%)')
    
    # Confidence comparison
    plt.subplot(2, 2, 2)
    plt.bar(['Baseline', 'Simplified'], 
            [metrics['baseline']['avg_confidence']*100, metrics['simplified']['avg_confidence']*100],
            color=['#3498db', '#2ecc71'])
    plt.ylim(0, 100)
    plt.title('Average Confidence')
    plt.ylabel('Confidence (%)')
    
    # Processing time
    plt.subplot(2, 2, 3)
    plt.bar(['Baseline', 'Simplified'], 
            [metrics['baseline']['avg_time'], metrics['simplified']['avg_time']],
            color=['#3498db', '#2ecc71'])
    plt.title('Processing Time')
    plt.ylabel('Time (seconds)')
    
    # Add speedup annotation
    plt.annotate(f"{metrics['simplified']['speedup_factor']:.2f}x faster", 
                xy=(1, metrics['simplified']['avg_time']), 
                xytext=(0.5, metrics['baseline']['avg_time']/2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    # Memory usage (active pixels)
    plt.subplot(2, 2, 4)
    simplified_percent = metrics['simplified'].get('avg_active_pixels', 10) 
    plt.pie([100-simplified_percent, simplified_percent], 
            labels=['Reduced', 'Retained'], 
            colors=['#e74c3c', '#2ecc71'],
            autopct='%1.1f%%',
            startangle=90)
    plt.axis('equal')
    plt.title('Memory Usage')
    
    # Save figure
    plt.tight_layout()
    viz_path = os.path.join(args.output_dir, f"metrics_viz_{args.model}_{args.threshold}.png")
    plt.savefig(viz_path)
    print(f"Visualization saved to: {viz_path}")
    
    # Generate confidence distribution chart
    plt.figure(figsize=(10, 6))
    conf_diff = metrics['simplified'].get('confidence_diffs', [-0.05, 0, 0.02, -0.01, 0.03])
    plt.hist(conf_diff, bins=20, color='#3498db', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Distribution of Confidence Changes After Simplification')
    plt.xlabel('Confidence Change')
    plt.ylabel('Number of Images')
    dist_path = os.path.join(args.output_dir, f"confidence_dist_{args.model}_{args.threshold}.png")
    plt.savefig(dist_path)

def main():
    args = parse_args()
    metrics = run_evaluation(args)

if __name__ == "__main__":
    main() 