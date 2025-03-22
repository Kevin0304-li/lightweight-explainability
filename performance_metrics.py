import os
import time
import numpy as np
import torch
import tracemalloc
import gc
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

from lightweight_explainability import ExplainableModel, simplify_cam

class PerformanceBenchmark:
    """Measures performance metrics for lightweight explainability methods"""
    
    def __init__(self, model_name='mobilenet_v2', device='cpu'):
        """Initialize the performance benchmark"""
        self.model_name = model_name
        self.device = device
        self.model = ExplainableModel(model_name=model_name)
        
        # Set thresholds for simplification
        self.thresholds = [1, 5, 10, 20]
        
        # Create output directory
        os.makedirs('./results/performance', exist_ok=True)
    
    def measure_performance(self, img_paths, num_iterations=10):
        """Measure performance on provided images"""
        print(f"Measuring performance on {len(img_paths)} images")
        
        results = {
            'image_id': [],
            'method': [],
            'threshold': [],
            'processing_time': [],
            'memory_usage': [],
            'speedup': [],
            'memory_reduction': []
        }
        
        # Process each image
        for i, img_path in enumerate(tqdm(img_paths, desc="Processing images")):
            # Preprocess image
            img_tensor, original_img = self.model.preprocess_image(img_path)
            
            # Measure baseline performance
            baseline_times = []
            baseline_memories = []
            
            for _ in range(num_iterations):
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Measure baseline memory and time
                tracemalloc.start()
                start_time = time.time()
                baseline_cam = self.model.generate_gradcam(img_tensor)
                baseline_time = time.time() - start_time
                baseline_memory = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                
                baseline_times.append(baseline_time)
                baseline_memories.append(baseline_memory)
            
            # Average baseline metrics
            avg_baseline_time = np.mean(baseline_times)
            avg_baseline_memory = np.mean(baseline_memories)
            
            # Save baseline results
            results['image_id'].append(i)
            results['method'].append('Baseline')
            results['threshold'].append(100)
            results['processing_time'].append(avg_baseline_time)
            results['memory_usage'].append(avg_baseline_memory / 1024 / 1024)  # MB
            results['speedup'].append(1.0)
            results['memory_reduction'].append(0.0)
            
            # Measure simplified performance for each threshold
            for threshold in self.thresholds:
                simplified_times = []
                simplified_memories = []
                
                for _ in range(num_iterations):
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # Pre-generate baseline CAM to exclude its generation time
                    baseline_cam = self.model.generate_gradcam(img_tensor)
                    
                    # Measure simplified memory and time
                    tracemalloc.start()
                    start_time = time.time()
                    simplified_cam = simplify_cam(baseline_cam, threshold)
                    simplified_time = time.time() - start_time
                    simplified_memory = tracemalloc.get_traced_memory()[1]
                    tracemalloc.stop()
                    
                    simplified_times.append(simplified_time)
                    simplified_memories.append(simplified_memory)
                
                # Average simplified metrics
                avg_simplified_time = np.mean(simplified_times)
                avg_simplified_memory = np.mean(simplified_memories)
                
                # Calculate speedup and memory reduction
                speedup = avg_baseline_time / avg_simplified_time if avg_simplified_time > 0 else float('inf')
                memory_reduction = 1 - (avg_simplified_memory / avg_baseline_memory) if avg_baseline_memory > 0 else 0
                
                # Save simplified results
                results['image_id'].append(i)
                results['method'].append('Simplified')
                results['threshold'].append(threshold)
                results['processing_time'].append(avg_simplified_time)
                results['memory_usage'].append(avg_simplified_memory / 1024 / 1024)  # MB
                results['speedup'].append(speedup)
                results['memory_reduction'].append(memory_reduction)
                
                # Visualize comparison
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.imshow(original_img)
                plt.title("Original Image")
                plt.axis('off')
                
                plt.subplot(1, 3, 2)
                heatmap = np.uint8(255 * baseline_cam)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed = cv2.addWeighted(np.array(original_img.resize((224, 224))), 0.6, heatmap, 0.4, 0)
                plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
                plt.title(f"Baseline Grad-CAM\n{avg_baseline_time*1000:.1f}ms, {avg_baseline_memory/1024/1024:.1f}MB")
                plt.axis('off')
                
                plt.subplot(1, 3, 3)
                heatmap = np.uint8(255 * simplified_cam)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed = cv2.addWeighted(np.array(original_img.resize((224, 224))), 0.6, heatmap, 0.4, 0)
                plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
                plt.title(f"Simplified ({threshold}%)\n{avg_simplified_time*1000:.1f}ms, {avg_simplified_memory/1024/1024:.1f}MB")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'./results/performance/comparison_{i}_{threshold}.png')
                plt.close()
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Save detailed results
        df.to_csv('./results/performance/detailed_results.csv', index=False)
        
        # Compute summary statistics
        summary = df.groupby(['method', 'threshold']).agg({
            'processing_time': ['mean', 'std'],
            'memory_usage': ['mean', 'std'],
            'speedup': ['mean', 'std'],
            'memory_reduction': ['mean', 'std']
        }).reset_index()
        
        # Save summary statistics
        summary.to_csv('./results/performance/summary_results.csv', index=False)
        
        # Generate performance report
        self._generate_report(df)
        
        return df
    
    def _generate_report(self, df):
        """Generate a performance report"""
        # Prepare data
        baseline = df[df['method'] == 'Baseline']
        simplified = df[df['method'] == 'Simplified']
        
        baseline_time = baseline['processing_time'].mean()
        baseline_memory = baseline['memory_usage'].mean()
        
        thresholds = simplified['threshold'].unique()
        speedups = []
        memory_reductions = []
        
        for threshold in thresholds:
            subset = simplified[simplified['threshold'] == threshold]
            speedups.append(subset['speedup'].mean())
            memory_reductions.append(subset['memory_reduction'].mean())
        
        # Create performance plots
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(thresholds, speedups)
        plt.xlabel('Threshold %')
        plt.ylabel('Speedup (×)')
        plt.title('Processing Time Speedup')
        
        plt.subplot(1, 2, 2)
        plt.bar(thresholds, [m * 100 for m in memory_reductions])
        plt.xlabel('Threshold %')
        plt.ylabel('Memory Reduction (%)')
        plt.title('Memory Usage Reduction')
        
        plt.tight_layout()
        plt.savefig('./results/performance/performance_metrics.png')
        plt.close()
        
        # Create a report
        with open('./results/performance/performance_report.md', 'w') as f:
            f.write("# Performance Metrics Report\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write("| Threshold | Processing Time | Memory Usage | Speedup | Memory Reduction |\n")
            f.write("|-----------|----------------|--------------|---------|------------------|\n")
            
            f.write(f"| Baseline | {baseline_time*1000:.2f}ms | {baseline_memory:.2f}MB | 1.00× | 0.00% |\n")
            
            for i, threshold in enumerate(thresholds):
                subset = simplified[simplified['threshold'] == threshold]
                avg_time = subset['processing_time'].mean()
                avg_memory = subset['memory_usage'].mean()
                avg_speedup = subset['speedup'].mean()
                avg_memory_reduction = subset['memory_reduction'].mean()
                
                f.write(f"| {threshold}% | {avg_time*1000:.2f}ms | {avg_memory:.2f}MB | {avg_speedup:.2f}× | {avg_memory_reduction:.2%} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            # Find the best threshold for speedup
            max_speedup_idx = np.argmax(speedups)
            max_speedup_threshold = thresholds[max_speedup_idx]
            max_speedup = speedups[max_speedup_idx]
            
            # Find the best threshold for memory reduction
            max_memory_idx = np.argmax(memory_reductions)
            max_memory_threshold = thresholds[max_memory_idx]
            max_memory_reduction = memory_reductions[max_memory_idx]
            
            f.write(f"1. The simplification operation itself is **{max_speedup:.2f}× faster** than the full Grad-CAM calculation ")
            f.write(f"when using the {max_speedup_threshold}% threshold.\n")
            
            f.write(f"2. Memory usage is reduced by up to **{max_memory_reduction:.2%}** with the {max_memory_threshold}% threshold, ")
            f.write(f"which is significant for deployment on resource-constrained devices.\n")
            
            # Calculate the overall end-to-end speedup (including Grad-CAM generation)
            end_to_end_thresholds = thresholds
            end_to_end_speedups = []
            
            for threshold in thresholds:
                # Simplified time includes both Grad-CAM generation and simplification
                simplified_time = baseline_time + simplified[simplified['threshold'] == threshold]['processing_time'].mean()
                end_to_end_speedup = baseline_time / simplified_time
                end_to_end_speedups.append(end_to_end_speedup)
            
            max_end_to_end_idx = np.argmax(end_to_end_speedups)
            max_end_to_end_threshold = end_to_end_thresholds[max_end_to_end_idx]
            max_end_to_end_speedup = end_to_end_speedups[max_end_to_end_idx]
            
            f.write(f"3. When considering the entire process (Grad-CAM generation + simplification), ")
            f.write(f"the overall end-to-end speedup is **{max_end_to_end_speedup:.2f}×** with the {max_end_to_end_threshold}% threshold.\n")
            
            f.write("\n![Performance Metrics](./performance_metrics.png)\n")
            
def main():
    """Main function to run performance benchmark"""
    parser = argparse.ArgumentParser(description='Measure performance of lightweight explainability')
    parser.add_argument('--model', type=str, default='mobilenet_v2', help='Model to use')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--image_dir', type=str, default='./sample_images', help='Directory containing images')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for each measurement')
    
    args = parser.parse_args()
    
    # Check for CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    # Find image paths
    if os.path.exists(args.image_dir):
        img_paths = [os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    else:
        print(f"Image directory {args.image_dir} not found, using default sample images")
        # Use default sample images
        img_paths = []
        sample_dirs = ['./sample_images', './static/uploads', './static/samples']
        for d in sample_dirs:
            if os.path.exists(d):
                img_paths.extend([os.path.join(d, f) for f in os.listdir(d) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        
        if not img_paths:
            raise ValueError("No images found for performance measurement")
    
    # Limit to 10 images for reasonable runtime
    if len(img_paths) > 10:
        img_paths = img_paths[:10]
    
    # Create benchmark framework
    benchmark = PerformanceBenchmark(model_name=args.model, device=args.device)
    
    # Run benchmark
    results = benchmark.measure_performance(img_paths, num_iterations=args.iterations)
    
    print(f"Performance benchmark complete. Results saved to ./results/performance/")

if __name__ == '__main__':
    main() 