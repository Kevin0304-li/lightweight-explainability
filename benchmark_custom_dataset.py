import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from lightweight_explainability import ExplainableModel, preprocess_image
from evaluation_metrics import ExplanationEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark custom dataset')
    parser.add_argument('--data_dir', type=str, default='./examples', help='Directory with custom dataset')
    parser.add_argument('--output_dir', type=str, default='./results/custom_benchmark', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples per class')
    parser.add_argument('--thresholds', type=str, default='1,5,10,20', help='Comma-separated list of thresholds')
    return parser.parse_args()

def load_custom_dataset(data_dir, num_samples=5):
    """Load images from custom dataset"""
    dataset = []
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"Found {len(classes)} classes: {classes}")
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Take up to num_samples from this class
        sample_files = files[:num_samples]
        
        for file in sample_files:
            file_path = os.path.join(class_dir, file)
            dataset.append({
                'path': file_path,
                'class': class_name,
                'id': f"{class_name}_{os.path.basename(file)}"
            })
    
    print(f"Loaded {len(dataset)} images from custom dataset")
    return dataset, classes

def calculate_metrics(model, dataset, thresholds, output_dir):
    """Calculate metrics for each threshold"""
    os.makedirs(output_dir, exist_ok=True)
    evaluator = ExplanationEvaluator(model)
    
    # Convert thresholds to integers
    thresholds = [int(t) for t in thresholds.split(',')]
    
    results = {
        'thresholds': thresholds,
        'metrics': {
            'faithfulness': [],
            'completeness': [],
            'sensitivity': [],
            'memory_usage': [],
            'processing_time': [],
            'speedup': []
        },
        'per_class': {}
    }
    
    for sample in tqdm(dataset, desc="Evaluating samples"):
        img_path = sample['path']
        img_id = sample['id']
        class_name = sample['class']
        
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess_image(img)
        
        # Get baseline metrics
        start_time = time.time()
        prediction, confidence = model.predict(img_tensor)
        baseline_cam = model.generate_gradcam(img_tensor)
        baseline_time = time.time() - start_time
        
        # Create directory for this image if needed
        img_output_dir = os.path.join(output_dir, img_id.split('_')[0])  # Use class as directory
        os.makedirs(img_output_dir, exist_ok=True)
        
        # Save baseline visualization
        baseline_path = os.path.join(img_output_dir, f"baseline_{img_id}.png")
        model.visualize_explanation(img, baseline_cam, title=f"Baseline: {prediction}", 
                                   save_path=baseline_path)
        
        # Test each threshold
        for threshold in thresholds:
            # Generate simplified explanation
            start_time = time.time()
            simplified_cam = model.simplified_gradcam(img_tensor, threshold_pct=threshold)
            simplified_time = time.time() - start_time
            
            # Calculate metrics
            memory_reduction = evaluator.memory_usage_reduction(baseline_cam, simplified_cam)
            faithfulness = evaluator.faithfulness(model, img_tensor, simplified_cam)
            completeness = evaluator.completeness(model, img_tensor, simplified_cam)
            sensitivity = evaluator.sensitivity(model, img_tensor, simplified_cam)
            speedup = baseline_time / simplified_time if simplified_time > 0 else float('inf')
            
            # Save simplified visualization
            simplified_path = os.path.join(img_output_dir, f"simplified_{threshold}_{img_id}.png")
            model.visualize_explanation(img, simplified_cam, 
                                      title=f"Threshold {threshold}%: {prediction}",
                                      save_path=simplified_path)
            
            # Store metrics for this threshold and class
            if class_name not in results['per_class']:
                results['per_class'][class_name] = {
                    'faithfulness': [[] for _ in thresholds],
                    'completeness': [[] for _ in thresholds],
                    'sensitivity': [[] for _ in thresholds],
                    'memory_usage': [[] for _ in thresholds],
                    'processing_time': [[] for _ in thresholds],
                    'speedup': [[] for _ in thresholds]
                }
            
            threshold_idx = thresholds.index(threshold)
            results['per_class'][class_name]['faithfulness'][threshold_idx].append(faithfulness)
            results['per_class'][class_name]['completeness'][threshold_idx].append(completeness)
            results['per_class'][class_name]['sensitivity'][threshold_idx].append(sensitivity)
            results['per_class'][class_name]['memory_usage'][threshold_idx].append(memory_reduction)
            results['per_class'][class_name]['processing_time'][threshold_idx].append(simplified_time)
            results['per_class'][class_name]['speedup'][threshold_idx].append(speedup)
    
    # Calculate overall average metrics for each threshold
    for threshold_idx, threshold in enumerate(thresholds):
        faith_values = []
        comp_values = []
        sens_values = []
        mem_values = []
        time_values = []
        speedup_values = []
        
        for class_name in results['per_class']:
            faith_values.extend(results['per_class'][class_name]['faithfulness'][threshold_idx])
            comp_values.extend(results['per_class'][class_name]['completeness'][threshold_idx])
            sens_values.extend(results['per_class'][class_name]['sensitivity'][threshold_idx])
            mem_values.extend(results['per_class'][class_name]['memory_usage'][threshold_idx])
            time_values.extend(results['per_class'][class_name]['processing_time'][threshold_idx])
            speedup_values.extend(results['per_class'][class_name]['speedup'][threshold_idx])
        
        # Store average metrics for this threshold
        results['metrics']['faithfulness'].append(np.mean(faith_values))
        results['metrics']['completeness'].append(np.mean(comp_values))
        results['metrics']['sensitivity'].append(np.mean(sens_values))
        results['metrics']['memory_usage'].append(np.mean(mem_values))
        results['metrics']['processing_time'].append(np.mean(time_values))
        results['metrics']['speedup'].append(np.mean(speedup_values))
    
    return results

def generate_report(results, output_dir, classes):
    """Generate a report of the benchmark results"""
    report_path = os.path.join(output_dir, 'benchmark_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Custom Dataset Benchmark Report\n\n")
        
        # Overall metrics
        f.write("## Overall Metrics\n\n")
        f.write("| Threshold | Faithfulness | Completeness | Sensitivity | Memory Reduction | Speedup |\n")
        f.write("|-----------|--------------|--------------|-------------|-----------------|--------|\n")
        
        for i, threshold in enumerate(results['thresholds']):
            f.write(f"| {threshold}% | {results['metrics']['faithfulness'][i]:.4f} | ")
            f.write(f"{results['metrics']['completeness'][i]:.4f} | ")
            f.write(f"{results['metrics']['sensitivity'][i]:.4f} | ")
            f.write(f"{results['metrics']['memory_usage'][i]:.2f}% | ")
            f.write(f"{results['metrics']['speedup'][i]:.2f}x |\n")
        
        f.write("\n")
        
        # Per-class metrics
        f.write("## Per-Class Metrics\n\n")
        
        for class_name in classes:
            if class_name not in results['per_class']:
                continue
                
            f.write(f"### Class: {class_name}\n\n")
            f.write("| Threshold | Faithfulness | Completeness | Sensitivity | Memory Reduction | Speedup |\n")
            f.write("|-----------|--------------|--------------|-------------|-----------------|--------|\n")
            
            for i, threshold in enumerate(results['thresholds']):
                class_data = results['per_class'][class_name]
                
                faith_mean = np.mean(class_data['faithfulness'][i])
                comp_mean = np.mean(class_data['completeness'][i])
                sens_mean = np.mean(class_data['sensitivity'][i])
                mem_mean = np.mean(class_data['memory_usage'][i])
                speedup_mean = np.mean(class_data['speedup'][i])
                
                f.write(f"| {threshold}% | {faith_mean:.4f} | ")
                f.write(f"{comp_mean:.4f} | ")
                f.write(f"{sens_mean:.4f} | ")
                f.write(f"{mem_mean:.2f}% | ")
                f.write(f"{speedup_mean:.2f}x |\n")
            
            f.write("\n")
    
    print(f"Report saved to {report_path}")
    
    # Generate visualization plots
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot overall metrics
    metrics = ['faithfulness', 'completeness', 'sensitivity', 'memory_usage', 'speedup']
    titles = ['Faithfulness', 'Completeness', 'Sensitivity', 'Memory Reduction (%)', 'Speedup (x)']
    
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(10, 6))
        plt.plot(results['thresholds'], results['metrics'][metric], 'o-', linewidth=2)
        plt.title(f'{title} vs Threshold')
        plt.xlabel('Threshold (%)')
        plt.ylabel(title)
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, f'{metric}_overall.png'))
        plt.close()
    
    # Plot per-class metrics
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(10, 6))
        
        for class_name in results['per_class']:
            class_values = []
            
            for i in range(len(results['thresholds'])):
                class_values.append(np.mean(results['per_class'][class_name][metric][i]))
            
            plt.plot(results['thresholds'], class_values, 'o-', linewidth=2, label=class_name)
        
        plt.title(f'{title} vs Threshold (Per Class)')
        plt.xlabel('Threshold (%)')
        plt.ylabel(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f'{metric}_per_class.png'))
        plt.close()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = ExplainableModel()
    
    # Load dataset
    dataset, classes = load_custom_dataset(args.data_dir, args.num_samples)
    
    # Calculate metrics
    results = calculate_metrics(model, dataset, args.thresholds, args.output_dir)
    
    # Generate report
    generate_report(results, args.output_dir, classes)
    
    print("Benchmark complete!")

if __name__ == '__main__':
    main() 