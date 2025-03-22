import os
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
from PIL import Image
import time
import concurrent.futures

from lightweight_explainability import ExplainableModel, analyze_image, analyze_batch, compare_models, run_on_image

# Create necessary directories
os.makedirs('./sample_images', exist_ok=True)
os.makedirs('./results/model_comparisons', exist_ok=True)
os.makedirs('./results/batch_processing', exist_ok=True)

# Sample images to download for testing
sample_images = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/1280px-Felis_catus-cat_on_snow.jpg",
        "filename": "cat.jpg",
        "id": "cat_sample"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Poodle_black_puppy_portrait.jpg/1280px-Poodle_black_puppy_portrait.jpg",
        "filename": "dog.jpg", 
        "id": "dog_sample"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/42/Shaqi_jrvej.jpg/1280px-Shaqi_jrvej.jpg",
        "filename": "car.jpg",
        "id": "car_sample"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Goldfinch_male_breeding_plumage.jpg/1280px-Goldfinch_male_breeding_plumage.jpg",
        "filename": "bird.jpg",
        "id": "bird_sample"
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Pepperoni_pizza.jpg/1280px-Pepperoni_pizza.jpg",
        "filename": "pizza.jpg",
        "id": "pizza_sample"
    }
]

def download_sample_images():
    """Download sample images if they don't exist already"""
    print("\n" + "="*50)
    print("DOWNLOADING SAMPLE IMAGES")
    print("="*50)
    
    for image in sample_images:
        filepath = os.path.join('./sample_images', image['filename'])
        if not os.path.exists(filepath):
            print(f"Downloading {image['filename']}...")
            try:
                urllib.request.urlretrieve(image['url'], filepath)
                print(f"Downloaded {image['filename']}")
            except Exception as e:
                print(f"Failed to download {image['filename']}: {e}")
        else:
            print(f"Image {image['filename']} already exists")

def demonstrate_single_image():
    """Demonstrate analysis of a single image"""
    print("\n" + "="*50)
    print("SINGLE IMAGE ANALYSIS")
    print("="*50)
    
    # Choose a sample image
    filepath = os.path.join('./sample_images', 'cat.jpg')
    if not os.path.exists(filepath):
        print(f"Image {filepath} not found")
        return
        
    print(f"Analyzing {filepath}...")
    
    # Use the backward-compatible function for simple cases
    cam, simplified, explanation = run_on_image(filepath, 'cat_demo')
    
    print(f"Explanation: {' '.join(explanation)}")
    print("-"*50)

def demonstrate_model_comparison():
    """Compare different models on the same image"""
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    # Choose a sample image
    filepath = os.path.join('./sample_images', 'dog.jpg')
    if not os.path.exists(filepath):
        print(f"Image {filepath} not found")
        return
        
    print(f"Comparing models on {filepath}...")
    
    # Compare MobileNetV2 with ResNet18
    models_to_compare = ['mobilenet_v2', 'resnet18']
    results = compare_models(
        filepath, 
        'dog_model_comparison',
        models_to_compare
    )
    
    # Print explanations from different models
    for model_name in models_to_compare:
        print(f"\n{model_name} explanation: {' '.join(results[model_name]['explanation'])}")
    
    print("-"*50)

def demonstrate_batch_processing():
    """Demonstrate batch processing of multiple images"""
    print("\n" + "="*50)
    print("BATCH PROCESSING")
    print("="*50)
    
    # Get all available images
    img_paths = []
    img_ids = []
    
    for image in sample_images:
        filepath = os.path.join('./sample_images', image['filename'])
        if os.path.exists(filepath):
            img_paths.append(filepath)
            img_ids.append(f"batch_{image['id']}")
    
    if not img_paths:
        print("No images available for batch processing")
        return
        
    print(f"Processing {len(img_paths)} images in batch...")
    
    # Create model
    model = ExplainableModel('mobilenet_v2')
    
    # Process in batch
    start_time = time.time()
    results = analyze_batch(model, img_paths, img_ids)
    total_time = time.time() - start_time
    
    # Calculate average time per image
    avg_time = total_time / len(img_paths)
    
    print(f"Batch processing completed in {total_time:.2f} seconds")
    print(f"Average time per image: {avg_time:.2f} seconds")
    
    # Print class names and processing times
    for result in results:
        print(f"{result['image_id']}: {result['class_name']} - {result['times']['total']:.2f}s")
    
    print("-"*50)

def compare_processing_times():
    """Compare sequential vs batch processing times"""
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    
    # Get available images
    img_paths = []
    img_ids = []
    
    for image in sample_images[:3]:  # Use first 3 images for comparison
        filepath = os.path.join('./sample_images', image['filename'])
        if os.path.exists(filepath):
            img_paths.append(filepath)
            img_ids.append(f"perf_{image['id']}")
    
    if len(img_paths) < 2:
        print("Not enough images for performance comparison")
        return
    
    # Create model
    model = ExplainableModel('mobilenet_v2')
    
    # Sequential processing
    print("Running sequential processing...")
    seq_start = time.time()
    seq_results = []
    for path, img_id in zip(img_paths, img_ids):
        result = analyze_image(model, path, img_id)
        seq_results.append(result)
    seq_time = time.time() - seq_start
    
    # Batch processing
    print("Running batch processing...")
    batch_start = time.time()
    batch_results = analyze_batch(model, img_paths, img_ids)
    batch_time = time.time() - batch_start
    
    # Compare times
    print(f"Sequential processing time: {seq_time:.2f} seconds")
    print(f"Batch processing time: {batch_time:.2f} seconds")
    print(f"Speedup: {seq_time/batch_time:.2f}x")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(['Sequential', 'Batch'], [seq_time, batch_time])
    plt.title('Processing Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.savefig('./results/batch_processing/performance_comparison.png')
    plt.show()
    
    print("-"*50)

def run_comprehensive_demo():
    """Run a comprehensive demo of all features"""
    # First, download any missing sample images
    download_sample_images()
    
    # Demonstrate each feature
    demonstrate_single_image()
    demonstrate_model_comparison()
    demonstrate_batch_processing()
    compare_processing_times()
    
    print("\nDemo completed! Check the 'results' directory for output files.")

if __name__ == "__main__":
    run_comprehensive_demo()
