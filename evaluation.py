import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import json
import random
from typing import Dict, List, Tuple
import qrcode
from datetime import datetime

from lightweight_explainability import ExplainableModel, simplify_cam, show_heatmap, compare_heatmaps, generate_text_explanation

# Create directories for evaluation results
os.makedirs('./results/evaluation', exist_ok=True)
os.makedirs('./results/evaluation/survey_images', exist_ok=True)
os.makedirs('./results/evaluation/metrics', exist_ok=True)

def evaluate_image(model: ExplainableModel, img_path: str, top_percents: List[float] = [5, 10, 15]) -> Dict:
    """Evaluate an image with different simplification thresholds"""
    # Preprocess image
    img_tensor, img = model.preprocess_image(img_path)
    
    # Get prediction
    class_idx, class_name, confidence = model.predict(img_tensor)
    
    # Generate baseline Grad-CAM
    start_time = time.time()
    cam = model.generate_gradcam(img_tensor)
    baseline_time = time.time() - start_time
    
    # Calculate number of non-zero pixels in baseline
    total_pixels = cam.shape[0] * cam.shape[1]
    active_baseline = np.sum(cam > 0.2) / total_pixels * 100
    
    # Evaluate each simplification threshold
    simplified_results = {}
    
    for top_percent in top_percents:
        # Generate simplified Grad-CAM
        start_time = time.time()
        simplified = simplify_cam(cam, top_percent)
        simplified_time = time.time() - start_time
        
        # Calculate active pixels
        active_simplified = np.sum(simplified > 0) / total_pixels * 100
        
        # Generate explanation
        start_time = time.time()
        explanation = generate_text_explanation(simplified, class_name, confidence, img)
        explanation_time = time.time() - start_time
        
        # Calculate reduction in active pixels
        pixel_reduction = 100 * (1 - active_simplified / active_baseline) if active_baseline > 0 else 0
        
        simplified_results[str(top_percent)] = {
            'time': simplified_time,
            'active_pixels': active_simplified,
            'explanation_time': explanation_time,
            'explanation': explanation,
            'pixel_reduction': pixel_reduction,
            'cam': simplified
        }
    
    return {
        'img_path': img_path,
        'class_name': class_name,
        'confidence': confidence,
        'baseline': {
            'time': baseline_time,
            'active_pixels': active_baseline,
            'cam': cam
        },
        'simplified': simplified_results
    }

def prepare_survey_image(img_path: str, result: Dict, survey_id: str):
    """Prepare comparison images for the survey"""
    # Load original image
    img = Image.open(img_path)
    
    # Get baseline CAM and the 10% simplified version
    baseline_cam = result['baseline']['cam']
    simplified_cam = result['simplified']['10']['cam']
    
    # Create side-by-side comparison
    compare_save_path = f'./results/evaluation/survey_images/{survey_id}_comparison.png'
    
    compare_heatmaps(
        img, baseline_cam,
        img, simplified_cam,
        title1="Traditional Heatmap Explanation",
        title2="Simplified Heatmap Explanation",
        save_path=compare_save_path
    )
    
    # Get explanations
    baseline_explanation = generate_text_explanation(baseline_cam, 
                                                    result['class_name'], 
                                                    result['confidence'], 
                                                    img)
    
    simplified_explanation = result['simplified']['10']['explanation']
    
    # Create description images with text
    baseline_text = f"Explanation A: {' '.join(baseline_explanation)}"
    simplified_text = f"Explanation B: {' '.join(simplified_explanation)}"
    
    # Create a document with images and explanations for the survey
    plt.figure(figsize=(12, 16))
    
    # Title
    plt.subplot(4, 1, 1)
    plt.text(0.5, 0.5, f"Image Explanation Comparison - ID: {survey_id}", 
             horizontalalignment='center', verticalalignment='center', 
             fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # Display comparison image
    plt.subplot(4, 1, 2)
    comparison_img = plt.imread(compare_save_path)
    plt.imshow(comparison_img)
    plt.axis('off')
    
    # Display explanations
    plt.subplot(4, 1, 3)
    plt.text(0.1, 0.8, baseline_text, fontsize=12, wrap=True)
    plt.text(0.1, 0.4, simplified_text, fontsize=12, wrap=True)
    plt.axis('off')
    
    # Generate QR code linking to survey
    survey_url = f"https://forms.gle/exampleSurveyLink{survey_id}"
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(survey_url)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    # Save QR code
    qr_path = f'./results/evaluation/survey_images/{survey_id}_qr.png'
    qr_img.save(qr_path)
    
    # Display QR code
    plt.subplot(4, 1, 4)
    plt.imshow(plt.imread(qr_path))
    plt.text(0.5, 0.9, "Scan to take survey", horizontalalignment='center', fontsize=14)
    plt.axis('off')
    
    # Save the complete survey image
    survey_doc_path = f'./results/evaluation/survey_images/{survey_id}_survey_doc.png'
    plt.tight_layout()
    plt.savefig(survey_doc_path)
    plt.close()
    
    print(f"Created survey document for image {img_path}: {survey_doc_path}")
    
    return {
        'survey_id': survey_id,
        'image_path': img_path,
        'class_name': result['class_name'],
        'survey_doc_path': survey_doc_path,
        'comparison_path': compare_save_path,
        'qr_path': qr_path,
        'baseline_explanation': baseline_explanation,
        'simplified_explanation': simplified_explanation
    }

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate quantitative metrics from evaluation results"""
    num_images = len(results)
    
    # Initialize metrics
    metrics = {
        'num_images': num_images,
        'baseline': {
            'avg_time': 0,
            'avg_active_pixels': 0,
        },
        'simplified': {}
    }
    
    # Process each threshold
    thresholds = list(results[0]['simplified'].keys())
    for threshold in thresholds:
        metrics['simplified'][threshold] = {
            'avg_time': 0,
            'avg_active_pixels': 0,
            'avg_explanation_time': 0,
            'avg_pixel_reduction': 0,
            'avg_explanation_length': 0
        }
    
    # Calculate averages
    for result in results:
        # Baseline metrics
        metrics['baseline']['avg_time'] += result['baseline']['time']
        metrics['baseline']['avg_active_pixels'] += result['baseline']['active_pixels']
        
        # Simplified metrics for each threshold
        for threshold in thresholds:
            metrics['simplified'][threshold]['avg_time'] += result['simplified'][threshold]['time']
            metrics['simplified'][threshold]['avg_active_pixels'] += result['simplified'][threshold]['active_pixels']
            metrics['simplified'][threshold]['avg_explanation_time'] += result['simplified'][threshold]['explanation_time']
            metrics['simplified'][threshold]['avg_pixel_reduction'] += result['simplified'][threshold]['pixel_reduction']
            metrics['simplified'][threshold]['avg_explanation_length'] += len(' '.join(result['simplified'][threshold]['explanation']))
    
    # Compute averages
    metrics['baseline']['avg_time'] /= num_images
    metrics['baseline']['avg_active_pixels'] /= num_images
    
    for threshold in thresholds:
        metrics['simplified'][threshold]['avg_time'] /= num_images
        metrics['simplified'][threshold]['avg_active_pixels'] /= num_images
        metrics['simplified'][threshold]['avg_explanation_time'] /= num_images
        metrics['simplified'][threshold]['avg_pixel_reduction'] /= num_images
        metrics['simplified'][threshold]['avg_explanation_length'] /= num_images
    
    # Calculate speedups
    baseline_time = metrics['baseline']['avg_time']
    for threshold in thresholds:
        simplified_time = metrics['simplified'][threshold]['avg_time']
        metrics['simplified'][threshold]['speedup'] = baseline_time / simplified_time if simplified_time > 0 else 0
    
    return metrics

def visualize_metrics(metrics: Dict):
    """Create visualizations of the metrics"""
    # Get thresholds
    thresholds = list(metrics['simplified'].keys())
    threshold_vals = [float(t) for t in thresholds]
    
    # Time comparison
    plt.figure(figsize=(12, 6))
    
    # Add baseline as bar 0
    plt.bar(0, metrics['baseline']['avg_time'], label='Baseline')
    
    # Add simplified times
    for i, threshold in enumerate(thresholds, 1):
        plt.bar(i, metrics['simplified'][threshold]['avg_time'], 
                label=f'Simplified ({threshold}%)')
    
    plt.title('Average Processing Time')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(thresholds) + 1), 
               ['Baseline'] + [f'{t}%' for t in threshold_vals])
    plt.legend()
    plt.savefig('./results/evaluation/metrics/time_comparison.png')
    plt.close()
    
    # Active pixels comparison
    plt.figure(figsize=(12, 6))
    
    # Add baseline as bar 0
    plt.bar(0, metrics['baseline']['avg_active_pixels'], label='Baseline')
    
    # Add simplified active pixels
    for i, threshold in enumerate(thresholds, 1):
        plt.bar(i, metrics['simplified'][threshold]['avg_active_pixels'], 
                label=f'Simplified ({threshold}%)')
    
    plt.title('Average Percentage of Active Pixels')
    plt.ylabel('% of Image')
    plt.xticks(range(len(thresholds) + 1), 
               ['Baseline'] + [f'{t}%' for t in threshold_vals])
    plt.legend()
    plt.savefig('./results/evaluation/metrics/active_pixels_comparison.png')
    plt.close()
    
    # Pixel reduction comparison
    plt.figure(figsize=(12, 6))
    
    # Add simplified pixel reductions
    pixel_reductions = [metrics['simplified'][threshold]['avg_pixel_reduction'] 
                        for threshold in thresholds]
    
    plt.bar(range(len(thresholds)), pixel_reductions, 
            tick_label=[f'{t}%' for t in threshold_vals])
    
    plt.title('Average Reduction in Active Pixels')
    plt.ylabel('% Reduction')
    plt.savefig('./results/evaluation/metrics/pixel_reduction.png')
    plt.close()
    
    # Speedup comparison
    plt.figure(figsize=(12, 6))
    
    # Add simplified speedups
    speedups = [metrics['simplified'][threshold]['speedup'] 
                for threshold in thresholds]
    
    plt.bar(range(len(thresholds)), speedups, 
            tick_label=[f'{t}%' for t in threshold_vals])
    
    plt.title('Average Speedup Over Baseline')
    plt.ylabel('Speedup Factor (×)')
    plt.savefig('./results/evaluation/metrics/speedup_comparison.png')
    plt.close()
    
    # Explanation length comparison
    plt.figure(figsize=(12, 6))
    
    # Add simplified explanation lengths
    expl_lengths = [metrics['simplified'][threshold]['avg_explanation_length'] 
                   for threshold in thresholds]
    
    plt.bar(range(len(thresholds)), expl_lengths, 
            tick_label=[f'{t}%' for t in threshold_vals])
    
    plt.title('Average Explanation Length (characters)')
    plt.ylabel('Length')
    plt.savefig('./results/evaluation/metrics/explanation_length.png')
    plt.close()

def create_evaluation_report(metrics: Dict):
    """Create a textual report of evaluation results"""
    report_path = './results/evaluation/metrics/evaluation_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("LIGHTWEIGHT EXPLAINABILITY EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Number of images evaluated: {metrics['num_images']}\n\n")
        
        f.write("BASELINE METRICS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Average processing time: {metrics['baseline']['avg_time']:.4f} seconds\n")
        f.write(f"Average active pixels: {metrics['baseline']['avg_active_pixels']:.2f}% of image\n\n")
        
        f.write("SIMPLIFIED METRICS:\n")
        f.write("-" * 50 + "\n")
        
        for threshold in metrics['simplified']:
            f.write(f"Simplified ({threshold}%):\n")
            f.write(f"  - Average processing time: {metrics['simplified'][threshold]['avg_time']:.4f} seconds\n")
            f.write(f"  - Average active pixels: {metrics['simplified'][threshold]['avg_active_pixels']:.2f}% of image\n")
            f.write(f"  - Average explanation time: {metrics['simplified'][threshold]['avg_explanation_time']:.4f} seconds\n")
            f.write(f"  - Average pixel reduction: {metrics['simplified'][threshold]['avg_pixel_reduction']:.2f}%\n")
            f.write(f"  - Average explanation length: {metrics['simplified'][threshold]['avg_explanation_length']:.1f} characters\n")
            f.write(f"  - Speedup over baseline: {metrics['simplified'][threshold]['speedup']:.2f}x\n\n")
        
        f.write("CONCLUSIONS:\n")
        f.write("-" * 50 + "\n")
        
        # Find best threshold for balance of speed vs. detail
        thresholds = list(metrics['simplified'].keys())
        best_threshold = max(thresholds, 
                           key=lambda t: metrics['simplified'][t]['speedup'] * (1 - metrics['simplified'][t]['avg_active_pixels'] / 100))
        
        f.write(f"Best threshold for balance of speed and clarity: {best_threshold}%\n")
        f.write(f"This threshold provides a {metrics['simplified'][best_threshold]['speedup']:.2f}x speedup\n")
        f.write(f"while reducing active pixels by {metrics['simplified'][best_threshold]['avg_pixel_reduction']:.2f}%\n\n")
        
        f.write("RECOMMENDATIONS FOR USER STUDY:\n")
        f.write("-" * 50 + "\n")
        f.write("Based on the quantitative metrics, prepare a user study comparing:\n")
        f.write("1. Standard Grad-CAM visualization and explanation\n")
        f.write(f"2. Simplified ({best_threshold}%) visualization and enhanced explanation\n\n")
        f.write("Ask users to rate:\n")
        f.write("- Which explanation is clearer?\n")
        f.write("- Which helps you better understand the model's decision?\n")
        f.write("- Which explanation would you prefer to use in practice?\n")
    
    print(f"Evaluation report saved to {report_path}")

def prepare_survey_materials(results: List[Dict], num_samples: int = 5):
    """Prepare survey materials for qualitative evaluation"""
    # Select a subset of images for the survey
    if len(results) > num_samples:
        survey_results = random.sample(results, num_samples)
    else:
        survey_results = results
    
    # Generate unique survey IDs
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    survey_materials = []
    
    for i, result in enumerate(survey_results):
        survey_id = f"{timestamp}_{i+1}"
        survey_material = prepare_survey_image(result['img_path'], result, survey_id)
        survey_materials.append(survey_material)
    
    # Save survey materials metadata
    survey_metadata = {
        'timestamp': timestamp,
        'num_samples': len(survey_materials),
        'materials': survey_materials
    }
    
    with open('./results/evaluation/survey_images/survey_metadata.json', 'w') as f:
        # Convert lists to strings for JSON serialization
        for material in survey_metadata['materials']:
            material['baseline_explanation'] = ' '.join(material['baseline_explanation'])
            material['simplified_explanation'] = ' '.join(material['simplified_explanation'])
        json.dump(survey_metadata, f, indent=2)
    
    print(f"Prepared survey materials for {len(survey_materials)} images")
    print("Survey metadata saved to ./results/evaluation/survey_images/survey_metadata.json")
    
    # Create a survey instruction document
    create_survey_instructions(survey_metadata)
    
    return survey_metadata

def create_survey_instructions(survey_metadata: Dict):
    """Create instructions for the survey"""
    instructions_path = './results/evaluation/survey_images/survey_instructions.txt'
    
    with open(instructions_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("IMAGE EXPLANATION EVALUATION SURVEY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("INSTRUCTIONS:\n")
        f.write("-" * 50 + "\n")
        f.write("This survey aims to evaluate different approaches to explaining image classifications.\n\n")
        
        f.write("For each image, you will see:\n")
        f.write("1. The original image\n")
        f.write("2. Two different visualizations showing where the AI model is focusing\n")
        f.write("3. Two different text explanations of what the model is looking at\n\n")
        
        f.write("Please answer the following questions for each image:\n\n")
        
        f.write("1. Which visualization is clearer and easier to understand?\n")
        f.write("   A) Traditional heatmap (left)\n")
        f.write("   B) Simplified heatmap (right)\n\n")
        
        f.write("2. Which text explanation better helps you understand the model's decision?\n")
        f.write("   A) Explanation A\n")
        f.write("   B) Explanation B\n\n")
        
        f.write("3. On a scale of 1-5, how helpful is explanation A? (1=not helpful, 5=very helpful)\n\n")
        
        f.write("4. On a scale of 1-5, how helpful is explanation B? (1=not helpful, 5=very helpful)\n\n")
        
        f.write("5. Which approach would you prefer to use in practice?\n")
        f.write("   A) Traditional approach (left visualization + Explanation A)\n")
        f.write("   B) Simplified approach (right visualization + Explanation B)\n\n")
        
        f.write("6. Any additional comments or suggestions?\n\n")
        
        f.write("SURVEY MATERIALS:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Survey ID: {survey_metadata['timestamp']}\n")
        f.write(f"Number of images: {survey_metadata['num_samples']}\n\n")
        
        f.write("Image IDs for this survey:\n")
        for material in survey_metadata['materials']:
            f.write(f"- {material['survey_id']}: {os.path.basename(material['image_path'])} ({material['class_name']})\n")
    
    print(f"Survey instructions saved to {instructions_path}")

def run_evaluation(img_paths: List[str], model_name: str = 'mobilenet_v2'):
    """Run a complete evaluation"""
    # Initialize model
    model = ExplainableModel(model_name)
    
    # Evaluate each image
    results = []
    for img_path in img_paths:
        print(f"Evaluating {img_path}...")
        result = evaluate_image(model, img_path)
        results.append(result)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(results)
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_metrics(metrics)
    
    # Create report
    print("Creating evaluation report...")
    create_evaluation_report(metrics)
    
    # Prepare survey materials
    print("Preparing survey materials...")
    survey_metadata = prepare_survey_materials(results)
    
    print("Evaluation complete!")
    return metrics, survey_metadata

if __name__ == "__main__":
    # Check for sample images
    sample_dir = './sample_images'
    if not os.path.exists(sample_dir):
        print(f"Sample directory {sample_dir} not found")
        exit(1)
    
    # Collect image paths
    img_paths = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not img_paths:
        print("No images found in sample directory")
        exit(1)
    
    # Run evaluation
    metrics, survey_metadata = run_evaluation(img_paths) 