import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random
import concurrent.futures
from typing import Dict, List, Tuple

from lightweight_explainability import ExplainableModel, simplify_cam, show_heatmap, compare_heatmaps

# Create necessary directories
os.makedirs('./results/benchmark', exist_ok=True)
os.makedirs('./results/benchmark/comparisons', exist_ok=True)

def collect_test_images(sample_dir: str = './sample_images', num_images: int = 20) -> List[str]:
    """Collect paths to test images"""
    if not os.path.exists(sample_dir):
        raise ValueError(f"Sample directory {sample_dir} does not exist")
    
    image_files = [f for f in os.listdir(sample_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) < num_images:
        print(f"Warning: Only {len(image_files)} images available, less than requested {num_images}")
        return [os.path.join(sample_dir, f) for f in image_files]
    
    # Select a subset if we have more than needed
    selected = random.sample(image_files, num_images)
    return [os.path.join(sample_dir, f) for f in selected]

def benchmark_single_image(model: ExplainableModel, img_path: str) -> Dict:
    """Run benchmark on a single image, measuring baseline vs simplified times"""
    # Preprocess image
    img_tensor, img = model.preprocess_image(img_path)
    
    # Benchmark baseline Grad-CAM
    start_time = time.time()
    cam = model.generate_gradcam(img_tensor)
    baseline_time = time.time() - start_time
    
    # Benchmark simplified Grad-CAM (10% threshold)
    start_time = time.time()
    simplified_10 = simplify_cam(cam, top_percent=10)
    simplified_10_time = time.time() - start_time
    
    # Benchmark simplified Grad-CAM (5% threshold)
    start_time = time.time()
    simplified_5 = simplify_cam(cam, top_percent=5)
    simplified_5_time = time.time() - start_time
    
    # Compute percentage of active pixels
    total_pixels = cam.shape[0] * cam.shape[1]
    active_baseline = np.sum(cam > 0.2) / total_pixels * 100
    active_10 = np.sum(simplified_10 > 0) / total_pixels * 100
    active_5 = np.sum(simplified_5 > 0) / total_pixels * 100
    
    return {
        'img_path': img_path,
        'baseline_time': baseline_time,
        'simplified_10_time': simplified_10_time,
        'simplified_5_time': simplified_5_time,
        'active_baseline': active_baseline,
        'active_10': active_10,
        'active_5': active_5,
        'cam': cam,
        'simplified_10': simplified_10,
        'simplified_5': simplified_5,
        'img': img
    }

def run_benchmark(img_paths: List[str], model_name: str = 'mobilenet_v2') -> Dict:
    """Run benchmark on multiple images"""
    print(f"Running benchmark on {len(img_paths)} images using {model_name}...")
    
    # Initialize model
    model = ExplainableModel(model_name)
    
    # Process images
    results = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(benchmark_single_image, model, path) for path in img_paths]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                print(f"Processed image {i+1}/{len(img_paths)}: {os.path.basename(result['img_path'])}")
            except Exception as e:
                print(f"Error processing image: {e}")
    
    # Aggregate results
    baseline_times = [r['baseline_time'] for r in results]
    simplified_10_times = [r['simplified_10_time'] for r in results]
    simplified_5_times = [r['simplified_5_time'] for r in results]
    
    active_baseline = [r['active_baseline'] for r in results]
    active_10 = [r['active_10'] for r in results]
    active_5 = [r['active_5'] for r in results]
    
    summary = {
        'model': model_name,
        'num_images': len(img_paths),
        'baseline_time': {
            'mean': np.mean(baseline_times),
            'std': np.std(baseline_times),
            'total': np.sum(baseline_times)
        },
        'simplified_10_time': {
            'mean': np.mean(simplified_10_times),
            'std': np.std(simplified_10_times),
            'total': np.sum(simplified_10_times)
        },
        'simplified_5_time': {
            'mean': np.mean(simplified_5_times),
            'std': np.std(simplified_5_times),
            'total': np.sum(simplified_5_times)
        },
        'active_pixels': {
            'baseline': np.mean(active_baseline),
            'simplified_10': np.mean(active_10),
            'simplified_5': np.mean(active_5)
        },
        'detailed_results': results
    }
    
    return summary

def visualize_benchmark_results(results: Dict):
    """Create visualizations of benchmark results"""
    # Time comparison
    plt.figure(figsize=(10, 6))
    times = [
        results['baseline_time']['mean'], 
        results['simplified_10_time']['mean'], 
        results['simplified_5_time']['mean']
    ]
    labels = ['Baseline Grad-CAM', 'Simplified (10%)', 'Simplified (5%)']
    
    plt.bar(labels, times)
    plt.title(f'Average Processing Time Comparison ({results["model"]})')
    plt.ylabel('Time (seconds)')
    plt.savefig('./results/benchmark/time_comparison.png')
    plt.close()
    
    # Active pixels comparison
    plt.figure(figsize=(10, 6))
    pixels = [
        results['active_pixels']['baseline'],
        results['active_pixels']['simplified_10'],
        results['active_pixels']['simplified_5']
    ]
    
    plt.bar(labels, pixels)
    plt.title(f'Average Percentage of Active Pixels ({results["model"]})')
    plt.ylabel('% of Image')
    plt.savefig('./results/benchmark/active_pixels_comparison.png')
    plt.close()
    
    # Save side-by-side comparison images for a subset of results
    for i, result in enumerate(results['detailed_results'][:5]):  # First 5 images
        img_name = os.path.basename(result['img_path']).split('.')[0]
        
        # Side-by-side comparison: Baseline vs Simplified 10%
        compare_heatmaps(
            result['img'], result['cam'],
            result['img'], result['simplified_10'],
            title1="Baseline Grad-CAM",
            title2="Simplified (10%)",
            save_path=f'./results/benchmark/comparisons/{img_name}_comparison_10.png'
        )
        
        # Side-by-side comparison: Baseline vs Simplified 5%
        compare_heatmaps(
            result['img'], result['cam'],
            result['img'], result['simplified_5'],
            title1="Baseline Grad-CAM",
            title2="Simplified (5%)",
            save_path=f'./results/benchmark/comparisons/{img_name}_comparison_5.png'
        )
        
        print(f"Saved comparison visualizations for {img_name}")

def create_benchmark_report(results: Dict):
    """Create a textual report of benchmark results"""
    report_path = './results/benchmark/benchmark_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("LIGHTWEIGHT EXPLAINABILITY BENCHMARK REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Model: {results['model']}\n")
        f.write(f"Number of images: {results['num_images']}\n\n")
        
        f.write("TIME METRICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Baseline Grad-CAM:\n")
        f.write(f"  - Total time: {results['baseline_time']['total']:.4f} seconds\n")
        f.write(f"  - Mean time per image: {results['baseline_time']['mean']:.4f} seconds\n")
        f.write(f"  - Standard deviation: {results['baseline_time']['std']:.4f} seconds\n\n")
        
        f.write(f"Simplified Grad-CAM (10%):\n")
        f.write(f"  - Total time: {results['simplified_10_time']['total']:.4f} seconds\n")
        f.write(f"  - Mean time per image: {results['simplified_10_time']['mean']:.4f} seconds\n")
        f.write(f"  - Standard deviation: {results['simplified_10_time']['std']:.4f} seconds\n\n")
        
        f.write(f"Simplified Grad-CAM (5%):\n")
        f.write(f"  - Total time: {results['simplified_5_time']['total']:.4f} seconds\n")
        f.write(f"  - Mean time per image: {results['simplified_5_time']['mean']:.4f} seconds\n")
        f.write(f"  - Standard deviation: {results['simplified_5_time']['std']:.4f} seconds\n\n")
        
        f.write("ACTIVE PIXELS METRICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Baseline Grad-CAM: {results['active_pixels']['baseline']:.2f}% of image\n")
        f.write(f"Simplified Grad-CAM (10%): {results['active_pixels']['simplified_10']:.2f}% of image\n")
        f.write(f"Simplified Grad-CAM (5%): {results['active_pixels']['simplified_5']:.2f}% of image\n\n")
        
        f.write("SPEEDUP METRICS:\n")
        f.write("-" * 50 + "\n")
        baseline_total = results['baseline_time']['total']
        simplified_10_total = results['simplified_10_time']['total']
        simplified_5_total = results['simplified_5_time']['total']
        
        f.write(f"Simplified (10%) speedup over baseline: {baseline_total / simplified_10_total:.2f}x\n")
        f.write(f"Simplified (5%) speedup over baseline: {baseline_total / simplified_5_total:.2f}x\n\n")
        
        f.write("INDIVIDUAL IMAGE RESULTS:\n")
        f.write("-" * 50 + "\n")
        for i, result in enumerate(results['detailed_results']):
            img_name = os.path.basename(result['img_path'])
            f.write(f"Image {i+1}: {img_name}\n")
            f.write(f"  - Baseline time: {result['baseline_time']:.4f} seconds\n")
            f.write(f"  - Simplified (10%) time: {result['simplified_10_time']:.4f} seconds\n")
            f.write(f"  - Simplified (5%) time: {result['simplified_5_time']:.4f} seconds\n")
            f.write(f"  - Active pixels (baseline): {result['active_baseline']:.2f}%\n")
            f.write(f"  - Active pixels (10%): {result['active_10']:.2f}%\n")
            f.write(f"  - Active pixels (5%): {result['active_5']:.2f}%\n\n")
    
    print(f"Benchmark report saved to {report_path}")

def run_complete_benchmark(num_images: int = 20, model_name: str = 'mobilenet_v2'):
    """Run a complete benchmark with visualizations and report"""
    # Collect test images
    img_paths = collect_test_images(num_images=num_images)
    
    # Run benchmark
    start_time = time.time()
    results = run_benchmark(img_paths, model_name)
    total_time = time.time() - start_time
    
    print(f"Benchmark completed in {total_time:.2f} seconds")
    
    # Create visualizations
    visualize_benchmark_results(results)
    
    # Create report
    create_benchmark_report(results)
    
    # Return results for further analysis
    return results

if __name__ == "__main__":
    # Run benchmark with default settings (20 images, MobileNetV2)
    results = run_complete_benchmark()
    
    print("\nBenchmark Summary:")
    print(f"Baseline Grad-CAM: {results['baseline_time']['mean']:.4f} seconds per image")
    print(f"Simplified (10%): {results['simplified_10_time']['mean']:.4f} seconds per image")
    print(f"Simplified (5%): {results['simplified_5_time']['mean']:.4f} seconds per image")
    print(f"Check ./results/benchmark/ for detailed results and visualizations") 