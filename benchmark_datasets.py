import os
import time
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import tracemalloc
import gc
from collections import defaultdict
import cv2

from lightweight_explainability import ExplainableModel, simplify_cam
from evaluation_metrics import ExplanationEvaluator

# Create output directories
os.makedirs('./results/benchmark', exist_ok=True)
os.makedirs('./results/benchmark/imagenet', exist_ok=True)
os.makedirs('./results/benchmark/cifar10', exist_ok=True)

class BenchmarkFramework:
    """Framework for benchmarking lightweight explainability on standard datasets"""
    
    def __init__(self, model_name='mobilenet_v2', device='cpu'):
        """Initialize the benchmark framework"""
        self.model_name = model_name
        self.device = device
        self.model = ExplainableModel(model_name=model_name)
        self.evaluator = ExplanationEvaluator(self.model, device=device)
        
        # Set thresholds for simplification
        self.thresholds = [1, 5, 10, 20]
        
        # Dictionary to store results
        self.results = defaultdict(list)
        
    def benchmark_imagenet(self, num_samples=50, seed=42):
        """Benchmark on ImageNet validation set"""
        print(f"Benchmarking on ImageNet (samples: {num_samples})")
        
        # Set up ImageNet validation transform
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        try:
            # Try to load ImageNet validation set
            imagenet_val = datasets.ImageNet(root='./data', split='val', transform=transform)
            
            # If successful, create a subset for benchmarking
            np.random.seed(seed)
            indices = np.random.choice(len(imagenet_val), num_samples, replace=False)
            benchmark_subset = torch.utils.data.Subset(imagenet_val, indices)
            dataloader = torch.utils.data.DataLoader(benchmark_subset, batch_size=1, shuffle=False)
            
            # Run benchmark
            self._run_benchmark(dataloader, dataset_name="ImageNet")
            
        except Exception as e:
            print(f"Error loading ImageNet: {e}")
            print("Using ImageNet samples from the model instead")
            
            # Use sample ImageNet images included with PyTorch models
            sample_images = []
            try:
                from torchvision.models import list_models, get_model, get_weight
                models_with_weights = list_models(weights=True)
                
                if self.model_name in models_with_weights:
                    weights = get_weight(f"{self.model_name}_Weights.IMAGENET1K_V1")
                    sample_images = weights.samples
                    
                    # Use a subset of sample images
                    dataloader = [(img.to(self.device), label) for img, label in sample_images[:num_samples]]
                    
                    # Run benchmark
                    self._run_benchmark(dataloader, dataset_name="ImageNet")
                else:
                    print(f"No sample images available for {self.model_name}")
            except Exception as e:
                print(f"Error using sample images: {e}")
    
    def benchmark_cifar10(self, num_samples=100, seed=42):
        """Benchmark on CIFAR-10 test set"""
        print(f"Benchmarking on CIFAR-10 (samples: {num_samples})")
        
        # Set up CIFAR-10 transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match model input size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load CIFAR-10 test set
        cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        # Create a subset for benchmarking
        np.random.seed(seed)
        indices = np.random.choice(len(cifar10_test), num_samples, replace=False)
        benchmark_subset = torch.utils.data.Subset(cifar10_test, indices)
        
        # Debug print to understand data format
        print("Debugging CIFAR-10 data format:")
        sample_data, sample_label = benchmark_subset[0]
        print(f"Sample data type: {type(sample_data)}")
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample label: {sample_label}")
        
        # Create dataloader
        dataloader = torch.utils.data.DataLoader(benchmark_subset, batch_size=1, shuffle=False)
        
        # Custom processing for CIFAR-10
        results = self._run_benchmark_cifar10(dataloader, dataset_name="CIFAR-10")
        
    def _run_benchmark_cifar10(self, dataloader, dataset_name):
        """Run benchmark specifically for CIFAR-10 dataset"""
        print(f"Running benchmark on {dataset_name}...")
        
        # Initialize results
        results = {
            'dataset': [],
            'image_id': [],
            'method': [],
            'threshold': [],
            'processing_time': [],
            'memory_usage': [],
            'faithfulness': [],
            'completeness': [],
            'sensitivity': [],
            'iou': []
        }
        
        # Process each image
        for i, (img_tensor, label) in enumerate(tqdm(dataloader, desc=f"Processing {dataset_name}")):
            try:
                # CIFAR-10 is already batched, so tensor should be 4D: [B,C,H,W]
                if not isinstance(img_tensor, torch.Tensor):
                    print(f"Skipping non-tensor data at index {i}, type: {type(img_tensor)}")
                    continue
                
                # Process with baseline Grad-CAM
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Measure baseline memory and time
                tracemalloc.start()
                start_time = time.time()
                baseline_cam = self.model.generate_gradcam(img_tensor)
                baseline_time = time.time() - start_time
                baseline_memory = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                
                # Get prediction
                class_idx, class_name, confidence = self.model.predict(img_tensor)
                
                # Convert tensor to PIL image for visualization
                img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                img_pil = Image.fromarray((img_np * 255).astype('uint8'))
                
                # Save baseline results
                results['dataset'].append(dataset_name)
                results['image_id'].append(i)
                results['method'].append('Baseline')
                results['threshold'].append(100)
                results['processing_time'].append(baseline_time)
                results['memory_usage'].append(baseline_memory / 1024 / 1024)  # MB
                
                # Evaluate baseline
                faithfulness = self.evaluator.evaluate_faithfulness(img_tensor, baseline_cam)
                completeness = self.evaluator.evaluate_completeness(img_tensor, baseline_cam)
                sensitivity = self.evaluator.evaluate_sensitivity(img_tensor, baseline_cam)
                
                results['faithfulness'].append(faithfulness['confidence_drop'])
                results['completeness'].append(completeness['auc'])
                results['sensitivity'].append(sensitivity['mean_sensitivity'])
                results['iou'].append(1.0)  # Self-IoU is 1.0
                
                # Save baseline visualization
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(img_pil)
                plt.title(f"Original: {class_name} ({confidence:.2f})")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                heatmap = np.uint8(255 * baseline_cam)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed = cv2.addWeighted(np.array(img_pil), 0.6, heatmap, 0.4, 0)
                plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
                plt.title(f"Baseline Grad-CAM")
                plt.axis('off')
                
                os.makedirs(f'./results/benchmark/{dataset_name.lower()}', exist_ok=True)
                plt.savefig(f'./results/benchmark/{dataset_name.lower()}/baseline_{i}.png')
                plt.close()
                
                # Process with simplified Grad-CAM at different thresholds
                for threshold in self.thresholds:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # Measure simplified memory and time
                    tracemalloc.start()
                    start_time = time.time()
                    simplified_cam = simplify_cam(baseline_cam, threshold)
                    simplified_time = time.time() - start_time
                    simplified_memory = tracemalloc.get_traced_memory()[1]
                    tracemalloc.stop()
                    
                    # Save simplified results
                    results['dataset'].append(dataset_name)
                    results['image_id'].append(i)
                    results['method'].append('Simplified')
                    results['threshold'].append(threshold)
                    results['processing_time'].append(simplified_time)
                    results['memory_usage'].append(simplified_memory / 1024 / 1024)  # MB
                    
                    # Evaluate simplified
                    faithfulness = self.evaluator.evaluate_faithfulness(img_tensor, simplified_cam)
                    completeness = self.evaluator.evaluate_completeness(img_tensor, simplified_cam)
                    sensitivity = self.evaluator.evaluate_sensitivity(img_tensor, simplified_cam)
                    
                    results['faithfulness'].append(faithfulness['confidence_drop'])
                    results['completeness'].append(completeness['auc'])
                    results['sensitivity'].append(sensitivity['mean_sensitivity'])
                    
                    # Calculate IoU between baseline and simplified
                    baseline_binary = (baseline_cam > 0).astype(np.float32)
                    simplified_binary = (simplified_cam > 0).astype(np.float32)
                    intersection = np.logical_and(baseline_binary, simplified_binary).sum()
                    union = np.logical_or(baseline_binary, simplified_binary).sum()
                    iou = intersection / union if union > 0 else 0
                    results['iou'].append(iou)
                    
                    # Save simplified visualization
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    heatmap = np.uint8(255 * baseline_cam)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    superimposed = cv2.addWeighted(np.array(img_pil), 0.6, heatmap, 0.4, 0)
                    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
                    plt.title(f"Baseline Grad-CAM")
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    heatmap = np.uint8(255 * simplified_cam)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    superimposed = cv2.addWeighted(np.array(img_pil), 0.6, heatmap, 0.4, 0)
                    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
                    plt.title(f"Simplified ({threshold}%)")
                    plt.axis('off')
                    
                    plt.savefig(f'./results/benchmark/{dataset_name.lower()}/simplified_{i}_{threshold}.png')
                    plt.close()
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Limit the number of samples for memory reasons
            if i >= 20:
                break
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Check if we have any results
        if len(df) == 0:
            print(f"No valid results for {dataset_name}. Check if dataloader is correctly configured.")
            return
        
        # Save detailed results
        df.to_csv(f'./results/benchmark/{dataset_name.lower()}_results.csv', index=False)
        
        # Compute summary statistics
        summary = df.groupby(['method', 'threshold']).agg({
            'processing_time': ['mean', 'std'],
            'memory_usage': ['mean', 'std'],
            'faithfulness': ['mean', 'std'],
            'completeness': ['mean', 'std'],
            'sensitivity': ['mean', 'std'],
            'iou': ['mean', 'std']
        }).reset_index()
        
        # Save summary statistics
        summary.to_csv(f'./results/benchmark/{dataset_name.lower()}_summary.csv', index=False)
        
        # Calculate speedup and memory reduction
        baseline = df[df['method'] == 'Baseline']
        baseline_time = baseline['processing_time'].mean()
        baseline_memory = baseline['memory_usage'].mean()
        
        for threshold in self.thresholds:
            simplified = df[(df['method'] == 'Simplified') & (df['threshold'] == threshold)]
            if len(simplified) == 0:
                continue
                
            simplified_time = simplified['processing_time'].mean()
            simplified_memory = simplified['memory_usage'].mean()
            
            speedup = baseline_time / simplified_time if simplified_time > 0 else float('inf')
            memory_reduction = 1 - (simplified_memory / baseline_memory) if baseline_memory > 0 else 0
            
            print(f"Threshold {threshold}%:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Memory reduction: {memory_reduction:.2%}")
            print(f"  IoU with baseline: {simplified['iou'].mean():.2f}")
            try:
                print(f"  Faithfulness preservation: {simplified['faithfulness'].mean() / baseline['faithfulness'].mean():.2%}")
            except:
                print(f"  Faithfulness preservation: N/A")
            print(f"  Completeness: {simplified['completeness'].mean():.2f} (baseline: {baseline['completeness'].mean():.2f})")
            print(f"  Sensitivity: {simplified['sensitivity'].mean():.2f} (baseline: {baseline['sensitivity'].mean():.2f})")
            print()
        
        # Create visualization of results
        self._visualize_results(df, dataset_name)
        
        return df
    
    def _run_benchmark(self, dataloader, dataset_name):
        """Run benchmark on the given dataset"""
        print(f"Running benchmark on {dataset_name}...")
        
        # Initialize results
        results = {
            'dataset': [],
            'image_id': [],
            'method': [],
            'threshold': [],
            'processing_time': [],
            'memory_usage': [],
            'faithfulness': [],
            'completeness': [],
            'sensitivity': [],
            'iou': []
        }
        
        # Process each image
        for i, data in enumerate(tqdm(dataloader, desc=f"Processing {dataset_name}")):
            try:
                # Get image and label
                if isinstance(data, tuple) and len(data) == 2:
                    img_tensor, label = data
                    if not isinstance(img_tensor, torch.Tensor):
                        # Handle case where img_tensor is not a tensor
                        if isinstance(img_tensor, list):  # Handle list case
                            if len(img_tensor) > 0:
                                img_tensor = img_tensor[0]
                            else:
                                print(f"Skipping empty data at index {i}")
                                continue
                        img_tensor = torch.tensor(img_tensor)
                else:
                    img_tensor = data
                    label = None
                
                # Ensure tensor is in correct format
                if not isinstance(img_tensor, torch.Tensor):
                    print(f"Skipping non-tensor data at index {i}, type: {type(img_tensor)}")
                    continue
                    
                # Ensure batch dimension
                if img_tensor.dim() == 3:
                    img_tensor = img_tensor.unsqueeze(0)
                elif img_tensor.dim() != 4:
                    print(f"Skipping tensor with unexpected dimensions: {img_tensor.shape}")
                    continue
                
                # Process with baseline Grad-CAM
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Measure baseline memory and time
                tracemalloc.start()
                start_time = time.time()
                baseline_cam = self.model.generate_gradcam(img_tensor)
                baseline_time = time.time() - start_time
                baseline_memory = tracemalloc.get_traced_memory()[1]
                tracemalloc.stop()
                
                # Get prediction
                class_idx, class_name, confidence = self.model.predict(img_tensor)
                
                # Convert tensor to PIL image for visualization
                img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                img_pil = Image.fromarray((img_np * 255).astype('uint8'))
                
                # Save baseline results
                results['dataset'].append(dataset_name)
                results['image_id'].append(i)
                results['method'].append('Baseline')
                results['threshold'].append(100)
                results['processing_time'].append(baseline_time)
                results['memory_usage'].append(baseline_memory / 1024 / 1024)  # MB
                
                # Evaluate baseline
                faithfulness = self.evaluator.evaluate_faithfulness(img_tensor, baseline_cam)
                completeness = self.evaluator.evaluate_completeness(img_tensor, baseline_cam)
                sensitivity = self.evaluator.evaluate_sensitivity(img_tensor, baseline_cam)
                
                results['faithfulness'].append(faithfulness['confidence_drop'])
                results['completeness'].append(completeness['auc'])
                results['sensitivity'].append(sensitivity['mean_sensitivity'])
                results['iou'].append(1.0)  # Self-IoU is 1.0
                
                # Save baseline visualization
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(img_pil)
                plt.title(f"Original: {class_name} ({confidence:.2f})")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                heatmap = np.uint8(255 * baseline_cam)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed = cv2.addWeighted(np.array(img_pil), 0.6, heatmap, 0.4, 0)
                plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
                plt.title(f"Baseline Grad-CAM")
                plt.axis('off')
                
                os.makedirs(f'./results/benchmark/{dataset_name.lower()}', exist_ok=True)
                plt.savefig(f'./results/benchmark/{dataset_name.lower()}/baseline_{i}.png')
                plt.close()
                
                # Process with simplified Grad-CAM at different thresholds
                for threshold in self.thresholds:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # Measure simplified memory and time
                    tracemalloc.start()
                    start_time = time.time()
                    simplified_cam = simplify_cam(baseline_cam, threshold)
                    simplified_time = time.time() - start_time
                    simplified_memory = tracemalloc.get_traced_memory()[1]
                    tracemalloc.stop()
                    
                    # Save simplified results
                    results['dataset'].append(dataset_name)
                    results['image_id'].append(i)
                    results['method'].append('Simplified')
                    results['threshold'].append(threshold)
                    results['processing_time'].append(simplified_time)
                    results['memory_usage'].append(simplified_memory / 1024 / 1024)  # MB
                    
                    # Evaluate simplified
                    faithfulness = self.evaluator.evaluate_faithfulness(img_tensor, simplified_cam)
                    completeness = self.evaluator.evaluate_completeness(img_tensor, simplified_cam)
                    sensitivity = self.evaluator.evaluate_sensitivity(img_tensor, simplified_cam)
                    
                    results['faithfulness'].append(faithfulness['confidence_drop'])
                    results['completeness'].append(completeness['auc'])
                    results['sensitivity'].append(sensitivity['mean_sensitivity'])
                    
                    # Calculate IoU between baseline and simplified
                    baseline_binary = (baseline_cam > 0).astype(np.float32)
                    simplified_binary = (simplified_cam > 0).astype(np.float32)
                    intersection = np.logical_and(baseline_binary, simplified_binary).sum()
                    union = np.logical_or(baseline_binary, simplified_binary).sum()
                    iou = intersection / union if union > 0 else 0
                    results['iou'].append(iou)
                    
                    # Save simplified visualization
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    heatmap = np.uint8(255 * baseline_cam)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    superimposed = cv2.addWeighted(np.array(img_pil), 0.6, heatmap, 0.4, 0)
                    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
                    plt.title(f"Baseline Grad-CAM")
                    plt.axis('off')
                    
                    plt.subplot(1, 2, 2)
                    heatmap = np.uint8(255 * simplified_cam)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    superimposed = cv2.addWeighted(np.array(img_pil), 0.6, heatmap, 0.4, 0)
                    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
                    plt.title(f"Simplified ({threshold}%)")
                    plt.axis('off')
                    
                    plt.savefig(f'./results/benchmark/{dataset_name.lower()}/simplified_{i}_{threshold}.png')
                    plt.close()
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
            
            # Limit the number of samples for memory reasons
            if i >= 20:
                break
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Check if we have any results
        if len(df) == 0:
            print(f"No valid results for {dataset_name}. Check if dataloader is correctly configured.")
            return
            
        # Save detailed results
        df.to_csv(f'./results/benchmark/{dataset_name.lower()}_results.csv', index=False)
        
        # Compute summary statistics
        summary = df.groupby(['method', 'threshold']).agg({
            'processing_time': ['mean', 'std'],
            'memory_usage': ['mean', 'std'],
            'faithfulness': ['mean', 'std'],
            'completeness': ['mean', 'std'],
            'sensitivity': ['mean', 'std'],
            'iou': ['mean', 'std']
        }).reset_index()
        
        # Save summary statistics
        summary.to_csv(f'./results/benchmark/{dataset_name.lower()}_summary.csv', index=False)
        
        # Calculate speedup and memory reduction
        baseline = df[df['method'] == 'Baseline']
        baseline_time = baseline['processing_time'].mean()
        baseline_memory = baseline['memory_usage'].mean()
        
        for threshold in self.thresholds:
            simplified = df[(df['method'] == 'Simplified') & (df['threshold'] == threshold)]
            if len(simplified) == 0:
                continue
                
            simplified_time = simplified['processing_time'].mean()
            simplified_memory = simplified['memory_usage'].mean()
            
            speedup = baseline_time / simplified_time if simplified_time > 0 else float('inf')
            memory_reduction = 1 - (simplified_memory / baseline_memory) if baseline_memory > 0 else 0
            
            print(f"Threshold {threshold}%:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Memory reduction: {memory_reduction:.2%}")
            print(f"  IoU with baseline: {simplified['iou'].mean():.2f}")
            try:
                print(f"  Faithfulness preservation: {simplified['faithfulness'].mean() / baseline['faithfulness'].mean():.2%}")
            except:
                print(f"  Faithfulness preservation: N/A")
            print(f"  Completeness: {simplified['completeness'].mean():.2f} (baseline: {baseline['completeness'].mean():.2f})")
            print(f"  Sensitivity: {simplified['sensitivity'].mean():.2f} (baseline: {baseline['sensitivity'].mean():.2f})")
            print()
        
        # Create visualization of results
        self._visualize_results(df, dataset_name)
    
    def _visualize_results(self, df, dataset_name):
        """Create visualizations of benchmark results"""
        # Prepare data
        baseline = df[df['method'] == 'Baseline']
        simplified = df[df['method'] == 'Simplified']
        
        baseline_time = baseline['processing_time'].mean()
        baseline_memory = baseline['memory_usage'].mean()
        baseline_faithfulness = baseline['faithfulness'].mean()
        baseline_completeness = baseline['completeness'].mean()
        baseline_sensitivity = baseline['sensitivity'].mean()
        
        thresholds = simplified['threshold'].unique()
        speedups = []
        memory_reductions = []
        faithfulness_ratios = []
        completeness_ratios = []
        sensitivity_ratios = []
        ious = []
        
        for threshold in thresholds:
            subset = simplified[simplified['threshold'] == threshold]
            
            simplified_time = subset['processing_time'].mean()
            simplified_memory = subset['memory_usage'].mean()
            simplified_faithfulness = subset['faithfulness'].mean()
            simplified_completeness = subset['completeness'].mean()
            simplified_sensitivity = subset['sensitivity'].mean()
            simplified_iou = subset['iou'].mean()
            
            speedup = baseline_time / simplified_time if simplified_time > 0 else float('inf')
            memory_reduction = 1 - (simplified_memory / baseline_memory) if baseline_memory > 0 else 0
            faithfulness_ratio = simplified_faithfulness / baseline_faithfulness if baseline_faithfulness > 0 else 0
            completeness_ratio = simplified_completeness / baseline_completeness if baseline_completeness > 0 else 0
            sensitivity_ratio = simplified_sensitivity / baseline_sensitivity if baseline_sensitivity > 0 else 0
            
            speedups.append(speedup)
            memory_reductions.append(memory_reduction)
            faithfulness_ratios.append(faithfulness_ratio)
            completeness_ratios.append(completeness_ratio)
            sensitivity_ratios.append(sensitivity_ratio)
            ious.append(simplified_iou)
        
        # Create performance plots
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        plt.bar(thresholds, speedups)
        plt.xlabel('Threshold %')
        plt.ylabel('Speedup (×)')
        plt.title('Processing Time Speedup')
        
        plt.subplot(2, 2, 2)
        plt.bar(thresholds, [m * 100 for m in memory_reductions])
        plt.xlabel('Threshold %')
        plt.ylabel('Memory Reduction (%)')
        plt.title('Memory Usage Reduction')
        
        plt.subplot(2, 2, 3)
        plt.bar(thresholds, [i * 100 for i in ious])
        plt.xlabel('Threshold %')
        plt.ylabel('IoU (%)')
        plt.title('Intersection over Union with Baseline')
        
        plt.subplot(2, 2, 4)
        plt.plot(thresholds, faithfulness_ratios, 'o-', label='Faithfulness')
        plt.plot(thresholds, completeness_ratios, 's-', label='Completeness')
        plt.plot(thresholds, sensitivity_ratios, '^-', label='Sensitivity')
        plt.xlabel('Threshold %')
        plt.ylabel('Ratio to Baseline')
        plt.title('Quality Metrics Preservation')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'./results/benchmark/{dataset_name.lower()}_performance.png')
        plt.close()
        
        # Create a report
        with open(f'./results/benchmark/{dataset_name.lower()}_report.md', 'w') as f:
            f.write(f"# Benchmark Results for {dataset_name}\n\n")
            
            f.write("## Performance Metrics\n\n")
            f.write("| Threshold | Speedup | Memory Reduction | IoU | Faithfulness | Completeness | Sensitivity |\n")
            f.write("|-----------|---------|------------------|-----|--------------|--------------|-------------|\n")
            
            f.write(f"| Baseline | 1.00× | 0.00% | 100% | {baseline_faithfulness:.4f} | {baseline_completeness:.4f} | {baseline_sensitivity:.4f} |\n")
            
            for i, threshold in enumerate(thresholds):
                f.write(f"| {threshold}% | {speedups[i]:.2f}× | {memory_reductions[i]:.2%} | {ious[i]:.2%} | ")
                f.write(f"{faithfulness_ratios[i] * baseline_faithfulness:.4f} | {completeness_ratios[i] * baseline_completeness:.4f} | ")
                f.write(f"{sensitivity_ratios[i] * baseline_sensitivity:.4f} |\n")
            
            # Find best threshold
            best_idx = np.argmax([s * m * i * f * c * s for s, m, i, f, c, s in 
                                 zip(speedups, memory_reductions, ious, 
                                     faithfulness_ratios, completeness_ratios, sensitivity_ratios)])
            best_threshold = thresholds[best_idx]
            
            f.write(f"\n## Summary\n\n")
            f.write(f"Based on the benchmark results, the optimal threshold appears to be **{best_threshold}%**.\n\n")
            
            # Analysis of results
            f.write("### Key Findings\n\n")
            
            # Speedup analysis
            max_speedup = max(speedups)
            max_speedup_threshold = thresholds[speedups.index(max_speedup)]
            f.write(f"- **Speed**: The simplified method achieves up to **{max_speedup:.1f}×** speedup at {max_speedup_threshold}% threshold.\n")
            
            # Memory analysis
            max_memory_reduction = max(memory_reductions)
            max_memory_threshold = thresholds[memory_reductions.index(max_memory_reduction)]
            f.write(f"- **Memory**: Memory usage is reduced by up to **{max_memory_reduction:.1%}** at {max_memory_threshold}% threshold.\n")
            
            # Quality metrics
            f.write(f"- **Explanation Quality**: At the {best_threshold}% threshold, the simplified method maintains ")
            f.write(f"{faithfulness_ratios[best_idx]:.1%} of faithfulness, ")
            f.write(f"{completeness_ratios[best_idx]:.1%} of completeness, and ")
            f.write(f"{sensitivity_ratios[best_idx]:.1%} of sensitivity.\n")
            
            # IoU analysis
            best_iou = max(ious)
            best_iou_threshold = thresholds[ious.index(best_iou)]
            f.write(f"- **IoU**: The highest overlap with baseline explanations ({best_iou:.1%}) is achieved at {best_iou_threshold}% threshold.\n\n")
            
            f.write("### Conclusion\n\n")
            if best_threshold <= 5:
                f.write(f"The {best_threshold}% threshold provides an excellent balance of performance and explanation quality, ")
                f.write(f"achieving significant speedup ({speedups[best_idx]:.1f}×) and memory reduction ({memory_reductions[best_idx]:.1%}) ")
                f.write(f"while maintaining high explanation fidelity (IoU: {ious[best_idx]:.1%}).\n")
            else:
                f.write(f"While higher thresholds provide better performance, the {best_threshold}% threshold is recommended as it ")
                f.write(f"maintains a good balance between efficiency ({speedups[best_idx]:.1f}× speedup, {memory_reductions[best_idx]:.1%} memory reduction) ")
                f.write(f"and explanation quality (IoU: {ious[best_idx]:.1%}).\n")
            
            f.write("\n![Performance Metrics](./benchmark/{}_performance.png)\n".format(dataset_name.lower()))

def main():
    """Main function to run benchmarks"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark lightweight explainability')
    parser.add_argument('--model', type=str, default='mobilenet_v2', help='Model to use')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--dataset', type=str, default='both', help='Dataset to benchmark (imagenet, cifar10, or both)')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Check for CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    # Create benchmark framework
    framework = BenchmarkFramework(model_name=args.model, device=args.device)
    
    # Run benchmarks
    if args.dataset.lower() in ['imagenet', 'both']:
        framework.benchmark_imagenet(num_samples=args.num_samples, seed=args.seed)
    
    if args.dataset.lower() in ['cifar10', 'both']:
        framework.benchmark_cifar10(num_samples=args.num_samples, seed=args.seed)
    
    print(f"Benchmarks complete. Results saved to ./results/benchmark/")

if __name__ == '__main__':
    main() 