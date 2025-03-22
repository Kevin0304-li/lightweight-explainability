#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Dataset Explainability Example

This example demonstrates how to generate and visualize
explanations across different types of datasets.
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from visualization module
try:
    from src.visualization.multi_dataset_examples import (
        create_multi_dataset_figure,
        set_publication_style,
        DATASETS,
        SAMPLE_IMAGES
    )
except ImportError:
    print("Error: Could not import from multi_dataset_examples.py")
    print("Make sure you've run the script from the project root directory.")
    sys.exit(1)

def run_example():
    """Run the multi-dataset explanation example."""
    print("Running multi-dataset explainability example...")
    
    # Check if sample images exist
    for dataset, path in SAMPLE_IMAGES.items():
        if not os.path.exists(path):
            print(f"Warning: Sample image {path} for {dataset} dataset doesn't exist.")
            print("Please run 'python src/visualization/multi_dataset_examples.py' first to create sample images.")
            return
    
    # Load the sample images
    original_images = {}
    for dataset, path in SAMPLE_IMAGES.items():
        try:
            original_images[dataset] = Image.open(path).convert('RGB')
            print(f"Loaded {dataset} image from {path}")
        except Exception as e:
            print(f"Error loading {dataset} image: {e}")
    
    if not original_images:
        print("Error: No images could be loaded.")
        return
    
    # Create simulated explanations
    explanations = {}
    for dataset, img in original_images.items():
        # Create a synthetic heatmap centered on the image
        h, w = img.size[1], img.size[0]
        heatmap = np.zeros((h, w))
        
        # Center heatmap with radial gradient
        center_y, center_x = h//2, w//2
        radius = min(h, w)//3
        
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                if dist < radius:
                    heatmap[i, j] = max(0, 1 - dist/radius)
        
        explanations[dataset] = heatmap
    
    # Create the visualization
    output_path = os.path.join("results", "visualizations", "example_visualization.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig = create_multi_dataset_figure(
        explanations,
        original_images,
        save_path=output_path
    )
    
    print(f"Created visualization at {output_path}")
    print("Example complete!")

if __name__ == "__main__":
    run_example() 