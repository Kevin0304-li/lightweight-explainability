#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of thresholding limitations and adaptation strategies.

This script showcases common failure modes of thresholding in explainability methods
and demonstrates how different adaptation strategies can help overcome these limitations.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the universal explainer
try:
    from universal_explainability import UniversalExplainer, multi_level_threshold, simplify_importance_map
except ImportError:
    print("UniversalExplainer not found. Make sure you've created the universal_explainability.py module.")
    sys.exit(1)

# Create output directory
os.makedirs("./results/thresholding_limitations", exist_ok=True)

def load_sample_cnn():
    """Load a pre-trained CNN model and sample image."""
    # Load pre-trained MobileNetV2 model
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    
    # Load and preprocess sample image
    img_path = "./sample_images/dog.jpg"
    if not os.path.exists(img_path):
        print(f"Sample image not found at {img_path}")
        # Create a folder for sample images
        os.makedirs("./sample_images", exist_ok=True)
        # Download a sample image or use a solid color
        img = Image.new('RGB', (224, 224), color='white')
        img.save(img_path)
        print(f"Created blank sample image at {img_path}")
    
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(img).unsqueeze(0)
    
    return model, input_tensor, img

def load_sample_transformer():
    """Load a pre-trained Vision Transformer model and sample image."""
    try:
        from transformers import ViTForImageClassification, ViTFeatureExtractor
        
        # Load pre-trained ViT model
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        model.eval()
        
        # Load and preprocess sample image
        img_path = "./sample_images/dog.jpg"
        if not os.path.exists(img_path):
            print(f"Sample image not found at {img_path}")
            # Create a folder for sample images
            os.makedirs("./sample_images", exist_ok=True)
            # Create a blank image
            img = Image.new('RGB', (224, 224), color='white')
            img.save(img_path)
            print(f"Created blank sample image at {img_path}")
            
        img = Image.open(img_path).convert('RGB')
        inputs = feature_extractor(images=img, return_tensors="pt")
        
        return model, inputs, img
    except ImportError:
        print("transformers package not found. Vision Transformer example will be skipped.")
        return None, None, None

def demonstrate_information_loss():
    """Demonstrate information loss due to aggressive thresholding."""
    print("\n1. Demonstrating Information Loss...")
    
    model, input_tensor, img = load_sample_cnn()
    
    # Create a standard explainer
    explainer = UniversalExplainer(model)
    
    # Create a figure to compare different thresholds
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show original image
    axs[0, 0].imshow(img)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')
    
    # Test different thresholds
    thresholds = [1, 5, 10, 20, 50]
    
    for i, threshold in enumerate(thresholds):
        row, col = (i+1) // 3, (i+1) % 3
        
        # Generate explanation with current threshold
        explanation = explainer.explain(input_tensor, threshold_percent=threshold)
        
        # Visualize the explanation
        heatmap = explanation['heatmap']
        
        # Superimpose heatmap on original image
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
        heatmap_img = heatmap_img.resize(img.size, Image.BILINEAR)
        heatmap_np = np.array(heatmap_img)
        
        # Create a colored heatmap
        colored_heatmap = np.zeros((heatmap_np.shape[0], heatmap_np.shape[1], 3), dtype=np.float32)
        colored_heatmap[..., 0] = heatmap_np  # Red channel
        
        # Convert original image to numpy array
        img_np = np.array(img).astype(np.float32) / 255
        
        # Blend original image with heatmap
        alpha = 0.7
        blended = (1-alpha) * img_np + alpha * colored_heatmap
        blended = np.clip(blended, 0, 1)
        
        # Display the result
        axs[row, col].imshow(blended)
        axs[row, col].set_title(f"{threshold}% Threshold")
        axs[row, col].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("./results/thresholding_limitations/information_loss.png")
    plt.close()
    
    print("✓ Information loss demonstration complete.")

def create_synthetic_attention_map():
    """Create a synthetic attention map with both strong and subtle features."""
    # Create base map with random noise
    attention_map = np.random.rand(16, 16) * 0.05
    
    # Add a strong feature
    x, y = 5, 5
    attention_map[x-1:x+2, y-1:y+2] = 0.8
    
    # Add a medium feature
    x, y = 12, 9
    attention_map[x-1:x+2, y-1:y+2] = 0.4
    
    # Add several subtle but important features
    subtle_features = [(3, 14), (7, 12), (10, 2)]
    for x, y in subtle_features:
        attention_map[x, y] = 0.25
    
    # Add relational context between features
    for i, (x1, y1) in enumerate([(5, 5), (12, 9)] + subtle_features):
        for j, (x2, y2) in enumerate([(5, 5), (12, 9)] + subtle_features):
            if i != j:
                # Draw faint connections between features
                steps = max(abs(x2-x1), abs(y2-y1))
                for t in range(1, steps):
                    xt = x1 + int((x2-x1) * t/steps)
                    yt = y1 + int((y2-y1) * t/steps)
                    attention_map[xt, yt] = max(attention_map[xt, yt], 0.15)
    
    # Normalize
    attention_map = attention_map / attention_map.max()
    
    return attention_map

def demonstrate_architectural_challenges():
    """Demonstrate architecture-specific challenges with thresholding."""
    print("\n2. Demonstrating Architecture-Specific Challenges...")
    
    # Create a figure for comparison
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    
    # 1. CNN with distributed features
    print("  - CNN challenges...")
    model, input_tensor, img = load_sample_cnn()
    
    # Create a standard explainer
    explainer = UniversalExplainer(model)
    
    # Generate baseline explanation
    explanation = explainer.explain(input_tensor, threshold_percent=5)
    
    # Visualize standard explanation
    heatmap = explanation['heatmap']
    
    # Show heatmap
    axs[0, 0].imshow(heatmap, cmap='jet')
    axs[0, 0].set_title("CNN: Standard 5% Threshold")
    axs[0, 0].axis('off')
    
    # Visualization with connectivity preservation
    explainer.preserve_connectivity = True
    explanation = explainer.explain(input_tensor, threshold_percent=5)
    
    # Apply connectivity preservation manually
    preserved_heatmap = np.zeros_like(heatmap)
    regions = np.zeros_like(heatmap, dtype=int)
    
    # Simple connectivity analysis (simplified version of what's in UniversalExplainer)
    binary_map = heatmap > 0
    from scipy import ndimage
    labeled, num_features = ndimage.label(binary_map)
    
    for i in range(1, num_features + 1):
        region_size = np.sum(labeled == i)
        if region_size > 5:  # Keep regions larger than this threshold
            preserved_heatmap[labeled == i] = heatmap[labeled == i]
    
    axs[0, 1].imshow(preserved_heatmap, cmap='jet')
    axs[0, 1].set_title("CNN: With Connectivity Preservation")
    axs[0, 1].axis('off')
    
    # Dynamic thresholding
    explainer.preserve_connectivity = False
    explainer.use_dynamic_threshold = True
    explainer.min_coverage = 0.2
    explanation = explainer.explain(input_tensor)
    
    dynamic_heatmap = explanation['heatmap']
    
    axs[0, 2].imshow(dynamic_heatmap, cmap='jet')
    axs[0, 2].set_title(f"CNN: Dynamic Threshold ({explanation['threshold_percent']:.1f}%)")
    axs[0, 2].axis('off')
    
    # 2. Transformer attention distortion
    print("  - Transformer challenges...")
    
    # Use synthetic attention map for transformer
    attention_map = create_synthetic_attention_map()
    
    # Display original attention map
    axs[1, 0].imshow(attention_map, cmap='viridis')
    axs[1, 0].set_title("Transformer: Full Attention Map")
    axs[1, 0].axis('off')
    
    # Apply standard thresholding
    threshold = 0.3
    thresholded_map = attention_map.copy()
    thresholded_map[thresholded_map < threshold] = 0
    
    axs[1, 1].imshow(thresholded_map, cmap='viridis')
    axs[1, 1].set_title(f"Transformer: Simple Threshold ({threshold:.1f})")
    axs[1, 1].axis('off')
    
    # Apply relationship-preserving thresholding
    relational_map = attention_map.copy()
    threshold = 0.3
    mask = relational_map >= threshold
    
    # For each important position, also keep its connected positions
    for i in range(attention_map.shape[0]):
        for j in range(attention_map.shape[1]):
            if mask[i, j]:
                # Find neighbor indices
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < attention_map.shape[0] and 
                            0 <= nj < attention_map.shape[1] and
                            attention_map[ni, nj] > 0.15):  # Connection threshold
                            relational_map[ni, nj] = max(relational_map[ni, nj], attention_map[ni, nj])
    
    axs[1, 2].imshow(relational_map, cmap='viridis')
    axs[1, 2].set_title("Transformer: Relationship Preserved")
    axs[1, 2].axis('off')
    
    # 3. RNN sequential context loss
    print("  - RNN challenges...")
    
    # Create synthetic temporal importance
    seq_length = 50
    temporal_importance = np.zeros(seq_length)
    # Add some important timesteps
    important_timesteps = [10, 25, 40]
    for t in important_timesteps:
        temporal_importance[t] = 1.0
    
    # Add some medium importance
    medium_timesteps = [5, 15, 30, 45]
    for t in medium_timesteps:
        temporal_importance[t] = 0.5
    
    # Add connecting context
    for t1, t2 in zip(important_timesteps[:-1], important_timesteps[1:]):
        for t in range(t1+1, t2):
            # Create a bridge between important timesteps
            temporal_importance[t] = max(temporal_importance[t], 0.2)
    
    # Normalize
    temporal_importance = temporal_importance / temporal_importance.max()
    
    # Plot original importance
    axs[2, 0].bar(range(seq_length), temporal_importance, color='blue')
    axs[2, 0].set_title("RNN: Full Temporal Importance")
    axs[2, 0].set_xlabel("Timestep")
    axs[2, 0].set_ylabel("Importance")
    
    # Apply standard thresholding
    threshold = 0.4
    thresholded_importance = temporal_importance.copy()
    thresholded_importance[thresholded_importance < threshold] = 0
    
    axs[2, 1].bar(range(seq_length), thresholded_importance, color='blue')
    axs[2, 1].set_title(f"RNN: Simple Threshold ({threshold:.1f})")
    axs[2, 1].set_xlabel("Timestep")
    axs[2, 1].set_ylabel("Importance")
    
    # Apply temporal smoothing
    smoothed_importance = gaussian_filter1d(temporal_importance, sigma=2.0)
    smoothed_importance = smoothed_importance / smoothed_importance.max()  # Renormalize
    
    # Apply threshold to smoothed importance
    smoothed_thresholded = smoothed_importance.copy()
    smoothed_thresholded[smoothed_thresholded < threshold] = 0
    
    axs[2, 2].bar(range(seq_length), smoothed_thresholded, color='blue')
    axs[2, 2].set_title("RNN: Temporal Smoothing + Threshold")
    axs[2, 2].set_xlabel("Timestep")
    axs[2, 2].set_ylabel("Importance")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("./results/thresholding_limitations/architectural_challenges.png")
    plt.close()
    
    print("✓ Architectural challenges demonstration complete.")

def demonstrate_domain_specific_problems():
    """Demonstrate domain-specific problems with thresholding."""
    print("\n3. Demonstrating Domain-Specific Problems...")
    
    # Create a figure for comparison
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Medical imaging example - synthetic tumor image
    # Create a synthetic medical image with a small tumor
    medical_img = np.zeros((100, 100))
    # Add background texture
    medical_img += np.random.rand(100, 100) * 0.1
    # Add organ-like structure
    for i in range(100):
        for j in range(100):
            dist = np.sqrt((i-50)**2 + (j-50)**2)
            if dist < 40:
                medical_img[i, j] += 0.5 + 0.1 * np.random.rand()
    
    # Add small tumor
    tumor_x, tumor_y = 70, 30
    tumor_size = 3
    for i in range(tumor_x-tumor_size, tumor_x+tumor_size+1):
        for j in range(tumor_y-tumor_size, tumor_y+tumor_size+1):
            if 0 <= i < 100 and 0 <= j < 100:
                dist = np.sqrt((i-tumor_x)**2 + (j-tumor_y)**2)
                if dist < tumor_size:
                    medical_img[i, j] += 0.3
    
    # Create synthetic importance map
    importance_map = np.zeros_like(medical_img)
    # Main organ gets medium importance
    for i in range(100):
        for j in range(100):
            dist = np.sqrt((i-50)**2 + (j-50)**2)
            if dist < 40:
                importance_map[i, j] = 0.5 - dist/80
    
    # Tumor gets high importance but small area
    for i in range(tumor_x-tumor_size, tumor_x+tumor_size+1):
        for j in range(tumor_y-tumor_size, tumor_y+tumor_size+1):
            if 0 <= i < 100 and 0 <= j < 100:
                dist = np.sqrt((i-tumor_x)**2 + (j-tumor_y)**2)
                if dist < tumor_size:
                    importance_map[i, j] = 0.9
    
    # Display medical image
    axs[0, 0].imshow(medical_img, cmap='gray')
    axs[0, 0].set_title("Medical Image")
    axs[0, 0].axis('off')
    
    # Display full importance map
    axs[0, 1].imshow(importance_map, cmap='jet')
    axs[0, 1].set_title("Full Importance Map")
    axs[0, 1].axis('off')
    
    # Apply standard thresholding
    threshold = 0.7
    thresholded_map = importance_map.copy()
    thresholded_map[thresholded_map < threshold] = 0
    
    axs[0, 2].imshow(thresholded_map, cmap='jet')
    axs[0, 2].set_title(f"Thresholded Map ({threshold:.1f})")
    axs[0, 2].axis('off')
    
    # 2. Financial time series example
    # Create synthetic financial data
    time_steps = 100
    price = np.zeros(time_steps)
    price[0] = 100
    
    # Generate random walk with a trend change
    for i in range(1, time_steps):
        if i < 25:
            price[i] = price[i-1] + np.random.normal(0.1, 0.5)
        elif i < 30:  # Early signal before major event
            price[i] = price[i-1] + np.random.normal(-0.3, 0.5)
        elif i == 30:  # Major drop
            price[i] = price[i-1] - 5
        elif i < 70:
            price[i] = price[i-1] + np.random.normal(-0.2, 0.5)
        else:
            price[i] = price[i-1] + np.random.normal(0.15, 0.5)
    
    # Create synthetic importance map
    temporal_importance = np.zeros(time_steps)
    
    # Major event gets highest importance
    temporal_importance[30] = 1.0
    
    # Recovery period gets medium importance
    temporal_importance[70:75] = 0.6
    
    # Early signal gets low but crucial importance
    temporal_importance[25:30] = 0.3
    
    # Display time series
    axs[1, 0].plot(range(time_steps), price)
    axs[1, 0].set_title("Financial Time Series")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Price")
    
    # Display full importance
    axs[1, 1].bar(range(time_steps), temporal_importance)
    axs[1, 1].set_title("Full Temporal Importance")
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("Importance")
    
    # Apply standard thresholding that would miss early signals
    threshold = 0.5
    thresholded_importance = temporal_importance.copy()
    thresholded_importance[thresholded_importance < threshold] = 0
    
    axs[1, 2].bar(range(time_steps), thresholded_importance)
    axs[1, 2].set_title(f"Thresholded Importance ({threshold:.1f})")
    axs[1, 2].set_xlabel("Time")
    axs[1, 2].set_ylabel("Importance")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("./results/thresholding_limitations/domain_specific_problems.png")
    plt.close()
    
    print("✓ Domain-specific problems demonstration complete.")

def demonstrate_adaptive_strategies():
    """Demonstrate adaptive strategies to overcome thresholding limitations."""
    print("\n4. Demonstrating Adaptive Strategies...")
    
    # Create a figure for comparison
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Dynamic thresholding
    print("  - Dynamic thresholding...")
    model, input_tensor, img = load_sample_cnn()
    
    # Create standard explainer
    explainer = UniversalExplainer(model)
    
    # Generate explanation with fixed threshold
    fixed_explanation = explainer.explain(input_tensor, threshold_percent=10)
    fixed_heatmap = fixed_explanation['heatmap']
    
    # Enable dynamic thresholding
    explainer.use_dynamic_threshold = True
    explainer.min_coverage = 0.2
    
    # Generate explanation with dynamic threshold
    dynamic_explanation = explainer.explain(input_tensor)
    dynamic_heatmap = dynamic_explanation['heatmap']
    dynamic_threshold = dynamic_explanation.get('threshold_percent', 'unknown')
    
    # Display original image
    axs[0, 0].imshow(img)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis('off')
    
    # Display fixed threshold heatmap
    axs[0, 1].imshow(fixed_heatmap, cmap='jet')
    axs[0, 1].set_title("Fixed Threshold (10%)")
    axs[0, 1].axis('off')
    
    # Display dynamic threshold heatmap
    axs[0, 2].imshow(dynamic_heatmap, cmap='jet')
    axs[0, 2].set_title(f"Dynamic Threshold ({dynamic_threshold:.1f}%)")
    axs[0, 2].axis('off')
    
    # 2. Multi-level thresholding
    print("  - Multi-level thresholding...")
    
    # Generate raw explanation with low threshold to get raw heatmap
    explainer.use_dynamic_threshold = False
    raw_explanation = explainer.explain(input_tensor, threshold_percent=1)
    
    # Get raw heatmap
    if 'raw_heatmap' in raw_explanation:
        raw_heatmap = raw_explanation['raw_heatmap']
    else:
        raw_heatmap = raw_explanation['heatmap']
    
    # Apply multi-level thresholding
    multi_level = multi_level_threshold(raw_heatmap, thresholds=[5, 10, 20, 50])
    
    # Create a combined visualization
    combined = np.zeros_like(raw_heatmap)
    
    # Color coding different threshold levels
    colors = [
        [1.0, 0.0, 0.0],  # Red for 5%
        [0.0, 1.0, 0.0],  # Green for 10%
        [0.0, 0.0, 1.0],  # Blue for 20% 
        [1.0, 1.0, 0.0]   # Yellow for 50%
    ]
    
    multi_level_img = np.zeros((raw_heatmap.shape[0], raw_heatmap.shape[1], 3))
    
    # Apply color coding
    thresholds = [5, 10, 20, 50]
    for i, threshold in enumerate(thresholds):
        level_key = f"level_{threshold}"
        level_map = multi_level[level_key]
        
        mask = level_map > 0
        for c in range(3):
            multi_level_img[..., c][mask] = colors[i][c]
    
    # Display multi-level visualization
    axs[1, 0].imshow(multi_level_img)
    axs[1, 0].set_title("Multi-level Thresholding")
    axs[1, 0].axis('off')
    
    # Create legend
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=colors[i], label=f"{thresholds[i]}% Threshold") 
        for i in range(len(thresholds))
    ]
    axs[1, 0].legend(handles=patches, loc='lower right')
    
    # 3. Confidence scoring
    print("  - Confidence scoring...")
    
    # Generate explanations with different thresholds and get confidence scores
    thresholds = [5, 10, 20, 30, 50]
    confidence_scores = []
    
    for threshold in thresholds:
        explanation = explainer.explain(input_tensor, threshold_percent=threshold, return_confidence=True)
        confidence_scores.append(explanation.get('confidence', 0))
    
    # Plot confidence scores
    axs[1, 1].bar(thresholds, confidence_scores, color='green')
    axs[1, 1].set_title("Explanation Confidence by Threshold")
    axs[1, 1].set_xlabel("Threshold (%)")
    axs[1, 1].set_ylabel("Confidence Score")
    axs[1, 1].set_ylim(0, 1)
    
    # Get the optimal threshold with highest confidence
    optimal_threshold = thresholds[np.argmax(confidence_scores)]
    
    # Generate explanation with optimal threshold
    optimal_explanation = explainer.explain(input_tensor, threshold_percent=optimal_threshold)
    optimal_heatmap = optimal_explanation['heatmap']
    
    # Display optimal threshold heatmap
    axs[1, 2].imshow(optimal_heatmap, cmap='jet')
    axs[1, 2].set_title(f"Optimal Threshold ({optimal_threshold}%)")
    axs[1, 2].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("./results/thresholding_limitations/adaptive_strategies.png")
    plt.close()
    
    print("✓ Adaptive strategies demonstration complete.")

def run_all_demonstrations():
    """Run all demonstration functions."""
    print("Starting thresholding limitations demos...")
    
    demonstrate_information_loss()
    demonstrate_architectural_challenges()
    demonstrate_domain_specific_problems()
    demonstrate_adaptive_strategies()
    
    print("\nAll demonstrations completed successfully!")
    print("Results saved in ./results/thresholding_limitations/")

if __name__ == "__main__":
    run_all_demonstrations() 