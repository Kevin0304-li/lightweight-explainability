import os
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd
import json

from lightweight_explainability import ExplainableModel, preprocess_image

class MobileDeviceBenchmark:
    """Simulate performance on mobile devices by applying appropriate throttling"""
    
    def __init__(self, device_type='raspberry_pi'):
        self.device_type = device_type
        
        # Device performance factors (CPU, memory, I/O relative to desktop)
        # These are approximations based on typical performance
        self.device_profiles = {
            'raspberry_pi': {
                'cpu_factor': 0.15,  # 15% of desktop CPU performance
                'memory_limit': 1024,  # 1GB RAM
                'io_factor': 0.3,  # 30% of desktop I/O performance
                'name': 'Raspberry Pi 4'
            },
            'android': {
                'cpu_factor': 0.3,  # 30% of desktop CPU performance
                'memory_limit': 4096,  # 4GB RAM
                'io_factor': 0.5,  # 50% of desktop I/O performance
                'name': 'Mid-range Android'
            },
            'iphone': {
                'cpu_factor': 0.45,  # 45% of desktop CPU performance
                'memory_limit': 6144,  # 6GB RAM
                'io_factor': 0.6,  # 60% of desktop I/O performance
                'name': 'iPhone 13'
            }
        }
        
        # Use provided device type or default to Raspberry Pi
        if device_type not in self.device_profiles:
            print(f"Unknown device type: {device_type}. Using 'raspberry_pi' profile.")
            self.device_type = 'raspberry_pi'
            
        self.profile = self.device_profiles[self.device_type]
        print(f"Using device profile: {self.profile['name']}")
        
        # Initialize model
        self.model = ExplainableModel()
    
    def simulate_execution_time(self, baseline_time):
        """Simulate execution time on the target device
        
        Args:
            baseline_time: Time measured on desktop
            
        Returns:
            simulated_time: Estimated time on target device
        """
        # Apply CPU factor and add some variability
        cpu_time = baseline_time / self.profile['cpu_factor']
        
        # Add random variation (±10%)
        variation = np.random.uniform(0.9, 1.1)
        
        return cpu_time * variation
    
    def simulate_memory_usage(self, baseline_memory_mb):
        """Simulate memory constraints on the target device
        
        Args:
            baseline_memory_mb: Memory usage in MB on desktop
            
        Returns:
            can_run: Whether the operation can run on the device
            adjusted_memory: Adjusted memory usage on target device
        """
        # Memory usage is generally similar, but with some overhead
        adjusted_memory = baseline_memory_mb * 1.2  # 20% overhead for mobile
        
        # Check if it exceeds device memory
        can_run = adjusted_memory < self.profile['memory_limit']
        
        return can_run, adjusted_memory
    
    def run_benchmark(self, data_dir='./examples', num_samples=5, thresholds=[1, 5, 10, 20], output_dir='./results/mobile'):
        """Run benchmark on simulated mobile device
        
        Args:
            data_dir: Directory containing test images
            num_samples: Number of samples to test
            thresholds: List of threshold percentages to test
            output_dir: Directory to save results
            
        Returns:
            results: Dictionary with benchmark results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        device_dir = os.path.join(output_dir, self.device_type)
        os.makedirs(device_dir, exist_ok=True)
        
        # Load images
        images = []
        if os.path.isdir(data_dir):
            # Find all image files in directory
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        images.append(file_path)
        
        # Randomly select subset
        if len(images) > num_samples:
            import random
            images = random.sample(images, num_samples)
        
        print(f"Testing with {len(images)} images")
        
        # Initialize results structure
        results = {
            'device': self.profile['name'],
            'thresholds': thresholds,
            'images': [],
            'summary': {
                'baseline_time': [],
                'simulated_baseline_time': [],
                'simplified_times': {t: [] for t in thresholds},
                'simulated_simplified_times': {t: [] for t in thresholds},
                'speedup_factors': {t: [] for t in thresholds},
                'memory_usage': [],
                'memory_reduction': {t: [] for t in thresholds}
            }
        }
        
        # Process each image
        for img_path in tqdm(images, desc=f"Benchmarking on {self.profile['name']}"):
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess_image(img)
            
            img_results = {
                'path': img_path,
                'baseline': {},
                'simplified': {}
            }
            
            # Generate baseline Grad-CAM and measure time
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            start_time = time.time()
            _, _ = self.model.predict(img_tensor)
            baseline_cam = self.model.generate_gradcam(img_tensor)
            baseline_time = time.time() - start_time
            
            # Simulate on target device
            simulated_baseline_time = self.simulate_execution_time(baseline_time)
            
            # Measure memory usage
            import psutil
            process = psutil.Process(os.getpid())
            baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Check if operation can run on device
            can_run_baseline, adjusted_baseline_memory = self.simulate_memory_usage(baseline_memory)
            
            img_results['baseline'] = {
                'time': baseline_time,
                'simulated_time': simulated_baseline_time,
                'memory_mb': baseline_memory,
                'simulated_memory_mb': adjusted_baseline_memory,
                'can_run_on_device': can_run_baseline
            }
            
            # Store in summary
            results['summary']['baseline_time'].append(baseline_time)
            results['summary']['simulated_baseline_time'].append(simulated_baseline_time)
            results['summary']['memory_usage'].append(baseline_memory)
            
            # Test each threshold
            img_results['simplified'] = {}
            
            for threshold in thresholds:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Generate simplified Grad-CAM and measure time
                start_time = time.time()
                simplified_cam, _ = self.model.simplified_gradcam(img_tensor, threshold_pct=threshold)
                simplified_time = time.time() - start_time
                
                # Simulate on target device
                simulated_simplified_time = self.simulate_execution_time(simplified_time)
                
                # Calculate speedup factor
                speedup = simulated_baseline_time / simulated_simplified_time if simulated_simplified_time > 0 else float('inf')
                
                # Measure memory usage
                process = psutil.Process(os.getpid())
                simplified_memory = process.memory_info().rss / (1024 * 1024)  # MB
                
                # Calculate memory reduction
                memory_reduction = (baseline_memory - simplified_memory) / baseline_memory * 100 if baseline_memory > 0 else 0
                
                # Check if operation can run on device
                can_run_simplified, adjusted_simplified_memory = self.simulate_memory_usage(simplified_memory)
                
                # Store results for this threshold
                img_results['simplified'][threshold] = {
                    'time': simplified_time,
                    'simulated_time': simulated_simplified_time,
                    'memory_mb': simplified_memory,
                    'simulated_memory_mb': adjusted_simplified_memory,
                    'memory_reduction_pct': memory_reduction,
                    'speedup_factor': speedup,
                    'can_run_on_device': can_run_simplified
                }
                
                # Store in summary
                results['summary']['simplified_times'][threshold].append(simplified_time)
                results['summary']['simulated_simplified_times'][threshold].append(simulated_simplified_time)
                results['summary']['speedup_factors'][threshold].append(speedup)
                results['summary']['memory_reduction'][threshold].append(memory_reduction)
            
            # Add image results to overall results
            results['images'].append(img_results)
        
        # Calculate summary statistics
        self.generate_report(results, device_dir)
        
        # Save raw results
        results_file = os.path.join(device_dir, 'benchmark_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def generate_report(self, results, output_dir):
        """Generate a report of the benchmark results
        
        Args:
            results: Dictionary with benchmark results
            output_dir: Directory to save report
        """
        report_path = os.path.join(output_dir, 'mobile_benchmark_report.md')
        
        with open(report_path, 'w') as f:
            f.write(f"# Mobile Benchmark Report: {results['device']}\n\n")
            
            # Summary table
            f.write("## Summary Results\n\n")
            f.write("| Metric | Baseline | ")
            for t in results['thresholds']:
                f.write(f"Threshold {t}% | ")
            f.write("\n")
            
            f.write("|--------|----------|")
            for _ in results['thresholds']:
                f.write("------------|")
            f.write("\n")
            
            # Average execution time
            avg_baseline_time = np.mean(results['summary']['simulated_baseline_time'])
            f.write(f"| Execution Time (s) | {avg_baseline_time:.4f} | ")
            
            for t in results['thresholds']:
                avg_time = np.mean(results['summary']['simulated_simplified_times'][t])
                f.write(f"{avg_time:.4f} | ")
            f.write("\n")
            
            # Speedup factor
            f.write(f"| Speedup Factor | 1.00x | ")
            for t in results['thresholds']:
                avg_speedup = np.mean(results['summary']['speedup_factors'][t])
                f.write(f"{avg_speedup:.2f}x | ")
            f.write("\n")
            
            # Memory reduction
            f.write(f"| Memory Reduction | 0% | ")
            for t in results['thresholds']:
                avg_reduction = np.mean(results['summary']['memory_reduction'][t])
                f.write(f"{avg_reduction:.2f}% | ")
            f.write("\n")
            
            # Check if all operations can run
            can_run_baseline = all(img['baseline']['can_run_on_device'] for img in results['images'])
            f.write(f"| Can Run on Device | {'Yes' if can_run_baseline else 'No'} | ")
            
            for t in results['thresholds']:
                can_run_threshold = all(img['simplified'][t]['can_run_on_device'] for img in results['images'])
                f.write(f"{'Yes' if can_run_threshold else 'No'} | ")
            f.write("\n\n")
            
            # Create plots directory
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Create execution time plot
            plt.figure(figsize=(10, 6))
            
            # Baseline as horizontal line
            plt.axhline(y=avg_baseline_time, color='r', linestyle='-', label='Baseline')
            
            # Simplified times
            x = results['thresholds']
            y = [np.mean(results['summary']['simulated_simplified_times'][t]) for t in x]
            
            plt.plot(x, y, 'o-', linewidth=2)
            plt.title(f'Execution Time on {results["device"]}')
            plt.xlabel('Threshold (%)')
            plt.ylabel('Time (seconds)')
            plt.grid(True)
            plt.legend()
            
            time_plot_path = os.path.join(plots_dir, 'execution_time.png')
            plt.savefig(time_plot_path)
            plt.close()
            
            # Add plot to report
            f.write(f"## Execution Time\n\n")
            f.write(f"![Execution Time](plots/execution_time.png)\n\n")
            
            # Create speedup plot
            plt.figure(figsize=(10, 6))
            
            y_speedup = [np.mean(results['summary']['speedup_factors'][t]) for t in x]
            
            plt.plot(x, y_speedup, 'o-', linewidth=2)
            plt.title(f'Speedup Factor on {results["device"]}')
            plt.xlabel('Threshold (%)')
            plt.ylabel('Speedup (x)')
            plt.grid(True)
            
            speedup_plot_path = os.path.join(plots_dir, 'speedup_factor.png')
            plt.savefig(speedup_plot_path)
            plt.close()
            
            # Add plot to report
            f.write(f"## Speedup Factor\n\n")
            f.write(f"![Speedup Factor](plots/speedup_factor.png)\n\n")
            
            # Create memory reduction plot
            plt.figure(figsize=(10, 6))
            
            y_memory = [np.mean(results['summary']['memory_reduction'][t]) for t in x]
            
            plt.plot(x, y_memory, 'o-', linewidth=2)
            plt.title(f'Memory Reduction on {results["device"]}')
            plt.xlabel('Threshold (%)')
            plt.ylabel('Memory Reduction (%)')
            plt.grid(True)
            
            memory_plot_path = os.path.join(plots_dir, 'memory_reduction.png')
            plt.savefig(memory_plot_path)
            plt.close()
            
            # Add plot to report
            f.write(f"## Memory Reduction\n\n")
            f.write(f"![Memory Reduction](plots/memory_reduction.png)\n\n")
            
            # Add findings and recommendations
            f.write("## Findings and Recommendations\n\n")
            
            # Find best threshold (balancing speed and memory)
            balanced_scores = {}
            for t in results['thresholds']:
                speedup = np.mean(results['summary']['speedup_factors'][t])
                memory_reduction = np.mean(results['summary']['memory_reduction'][t])
                # Simple balanced score (normalize and weight)
                balanced_scores[t] = speedup * 0.7 + memory_reduction/100 * 0.3
                
            best_threshold = max(balanced_scores, key=balanced_scores.get)
            
            f.write(f"- Best threshold for {results['device']}: **{best_threshold}%**\n")
            f.write(f"  - Provides {np.mean(results['summary']['speedup_factors'][best_threshold]):.2f}x speedup\n")
            f.write(f"  - Reduces memory usage by {np.mean(results['summary']['memory_reduction'][best_threshold]):.2f}%\n\n")
            
            # Check if baseline can run
            if not can_run_baseline:
                f.write(f"- **Warning**: Baseline method is too demanding for {results['device']}\n")
                
                # Find thresholds that can run
                viable_thresholds = []
                for t in results['thresholds']:
                    if all(img['simplified'][t]['can_run_on_device'] for img in results['images']):
                        viable_thresholds.append(t)
                
                if viable_thresholds:
                    f.write(f"- Recommended thresholds for this device: {', '.join(map(str, viable_thresholds))}%\n\n")
                else:
                    f.write(f"- None of the tested thresholds are viable for this device\n\n")
            
            # Conclusion
            f.write("### Conclusion\n\n")
            f.write(f"The simplified explainability approach with threshold {best_threshold}% provides the best balance ")
            f.write(f"of speed and quality for {results['device']}. ")
            
            if avg_baseline_time > 1.0:
                f.write("The baseline method is too slow for real-time applications on this device. ")
            
            best_avg_time = np.mean(results['summary']['simulated_simplified_times'][best_threshold])
            if best_avg_time < 0.1:
                f.write("The simplified method is fast enough for real-time applications.")
            elif best_avg_time < 0.5:
                f.write("The simplified method is acceptable for interactive applications.")
            else:
                f.write("The simplified method may still be too slow for real-time use on this device.")
        
        print(f"Report saved to {report_path}")
        
def main():
    parser = argparse.ArgumentParser(description='Mobile device benchmarking')
    parser.add_argument('--device', type=str, default='raspberry_pi', choices=['raspberry_pi', 'android', 'iphone'],
                        help='Device type to simulate')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of images to test')
    parser.add_argument('--thresholds', type=str, default='1,5,10,20', help='Comma-separated list of thresholds')
    parser.add_argument('--output_dir', type=str, default='./results/mobile', help='Output directory')
    parser.add_argument('--data_dir', type=str, default='./examples', help='Directory with test images')
    args = parser.parse_args()
    
    # Parse thresholds
    thresholds = [int(t) for t in args.thresholds.split(',')]
    
    # Create and run benchmark
    benchmark = MobileDeviceBenchmark(device_type=args.device)
    results = benchmark.run_benchmark(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        thresholds=thresholds,
        output_dir=args.output_dir
    )
    
    print("Mobile benchmark complete!")
    
if __name__ == '__main__':
    main() 