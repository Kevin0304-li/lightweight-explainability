import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor

from transformer_explainability import TransformerExplainer
from universal_explainability import UniversalExplainer

def main():
    """Demonstrate transformer explainability with ViT model"""
    # Create output directory
    os.makedirs('./results/transformer_examples', exist_ok=True)
    
    # Load ViT model and feature extractor
    print("Loading ViT model...")
    model_name = "google/vit-base-patch16-224"
    try:
        model = ViTForImageClassification.from_pretrained(model_name)
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please run: pip install transformers")
        return
    
    model.eval()
    
    # Load sample image
    img_path = "../sample_images/dog.jpg"
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        print("Please provide a valid image path")
        return
    
    # Load and preprocess image
    image = Image.open(img_path).convert("RGB")
    
    # Process image with feature extractor
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    
    # Method 1: Using specialized TransformerExplainer
    print("Generating explanation with TransformerExplainer...")
    explainer = TransformerExplainer(model, patch_size=16)
    explanation = explainer.explain(pixel_values, threshold_percent=10)
    
    # Visualize explanation
    vis_img = explainer.visualize_explanation(
        image, explanation, 
        title="ViT Explanation",
        save_path="./results/transformer_examples/vit_explanation_direct.png"
    )
    
    # Method 2: Using UniversalExplainer
    print("Generating explanation with UniversalExplainer...")
    universal_explainer = UniversalExplainer(model, model_type="transformer", patch_size=16)
    universal_explanation = universal_explainer.explain(pixel_values, threshold_percent=10)
    
    # Visualize explanation
    universal_vis_img = universal_explainer.visualize_explanation(
        image, universal_explanation,
        title="Universal Explainer - ViT",
        save_path="./results/transformer_examples/vit_explanation_universal.png"
    )
    
    # Run benchmark with different thresholds
    print("Running benchmark with different thresholds...")
    benchmark_results = universal_explainer.benchmark(
        pixel_values, thresholds=[5, 10, 20], num_runs=3
    )
    
    # Visualize benchmark results
    universal_explainer.visualize_benchmark(
        benchmark_results,
        save_path="./results/transformer_examples/vit_benchmark.png"
    )
    
    # Print performance summary
    print("Performance Summary:")
    print(universal_explainer.get_performance_summary())
    
    print("Done! Results saved in ./results/transformer_examples/")

if __name__ == "__main__":
    main() 