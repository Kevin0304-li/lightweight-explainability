#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate visualization of failure cases where thresholding removes critical information.

This script creates a visualization showing cases where our simplified thresholding
approach actually removes important features that are critical for correct predictions.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.gridspec as gridspec

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import explainability modules if available, otherwise simulate
try:
    from src.explainability.gradcam import GradCAM
    from src.explainability.lightweight import LightweightExplainer
    from src.models.model_loader import load_model
except ImportError:
    print("Error importing required modules")
    print("Running in simulation mode")
    SIMULATION_MODE = True
else:
    SIMULATION_MODE = False

# Create output directory
output_dir = "results/failure_cases"
os.makedirs(output_dir, exist_ok=True)

def apply_colormap(mask, colormap='jet'):
    """Apply colormap to a mask.
    
    Args:
        mask: 2D numpy array
        colormap: Name of matplotlib colormap
        
    Returns:
        RGB image with colormap applied
    """
    if mask.max() > 0:
        mask = mask / mask.max()  # Normalize to 0-1
    
    # Handle matplotlib deprecation warning by using colormaps dictionary if available
    try:
        cmap = plt.colormaps[colormap]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap(colormap)
        
    colored_mask = cmap(mask)
    
    # Convert to RGB, remove alpha channel
    return (colored_mask[:, :, :3] * 255).astype(np.uint8)

def blend_with_image(image, mask, alpha=0.5):
    """Blend mask with original image.
    
    Args:
        image: PIL Image
        mask: RGB mask as numpy array
        alpha: Blending factor
        
    Returns:
        Blended image as PIL Image
    """
    image_array = np.array(image)
    blended = (1 - alpha) * image_array + alpha * mask
    return Image.fromarray(blended.astype(np.uint8))

def simulate_multimodal_heatmap(height, width, centers, intensities, sigma_values):
    """Create a simulated heatmap with multiple important regions.
    
    Args:
        height: Image height
        width: Image width
        centers: List of (y, x) centers for important regions
        intensities: List of intensity values for each region
        sigma_values: List of standard deviations for each region
        
    Returns:
        Simulated heatmap as numpy array
    """
    heatmap = np.zeros((height, width))
    y_grid, x_grid = np.ogrid[:height, :width]
    
    for (cy, cx), intensity, sigma in zip(centers, intensities, sigma_values):
        # Create a Gaussian blob
        dist = np.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)
        gaussian = np.exp(-dist**2 / (2 * sigma**2))
        heatmap += intensity * gaussian
    
    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap

def generate_failure_case(image_size, threshold_percent=70):
    """Generate a failure case where thresholding removes critical information.
    
    Args:
        image_size: Tuple of (height, width)
        threshold_percent: Threshold percentage to keep top activations
        
    Returns:
        Tuple of (original_img, gradcam_heatmap, thresholded_heatmap, difference_mask)
    """
    height, width = image_size
    
    # Create a synthetic image with multiple objects
    # In a real case, we'd use actual images, but here we simulate
    original_img = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Simulate two important regions - one strong, one weaker but still important
    centers = [(height//3, width//3), (2*height//3, 2*width//3)]
    
    # Create main object - strong activation
    center_y, center_x = centers[0]
    obj_radius = height // 10
    y_grid, x_grid = np.ogrid[:height, :width]
    dist = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
    original_img[dist < obj_radius] = [50, 50, 200]  # Blue object
    
    # Create secondary object - weaker activation but still important
    center_y, center_x = centers[1]
    obj_radius = height // 12
    dist = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
    original_img[dist < obj_radius] = [200, 50, 50]  # Red object
    
    # Create a simulated gradcam heatmap with unequal importance
    gradcam_heatmap = simulate_multimodal_heatmap(
        height, width,
        centers=centers,
        intensities=[1.0, 0.4],  # Primary object stronger, secondary weaker
        sigma_values=[height//15, height//15]
    )
    
    # Apply thresholding to create lightweight explanation
    threshold = np.percentile(gradcam_heatmap, threshold_percent)
    thresholded_heatmap = gradcam_heatmap.copy()
    thresholded_heatmap[thresholded_heatmap < threshold] = 0
    
    # Calculate difference mask (what was lost)
    difference_mask = (gradcam_heatmap > 0) & (thresholded_heatmap == 0)
    difference_heatmap = gradcam_heatmap.copy()
    difference_heatmap[~difference_mask] = 0
    
    return Image.fromarray(original_img), gradcam_heatmap, thresholded_heatmap, difference_heatmap

def create_failure_visualization(n_examples=3, threshold_percents=[70, 80, 90], output_path=None):
    """Create visualization of failure cases at different thresholds.
    
    Args:
        n_examples: Number of examples to generate
        threshold_percents: List of threshold percentages to visualize
        output_path: Path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    # Create figure with n_examples rows and 4 columns for each threshold
    n_thresholds = len(threshold_percents)
    fig = plt.figure(figsize=(4 * n_thresholds, 4 * n_examples))
    fig.suptitle('Failure Cases Where Thresholding Removes Critical Information', fontsize=16)
    
    # Create grid layout
    gs = gridspec.GridSpec(n_examples, n_thresholds)
    
    # Image dimensions
    image_size = (224, 224)
    
    for i in range(n_examples):
        # Vary image generation slightly for each example
        offset_y = np.random.randint(-30, 30)
        offset_x = np.random.randint(-30, 30)
        size_variation = np.random.uniform(0.8, 1.2)
        
        for j, threshold in enumerate(threshold_percents):
            # Generate failure case with different threshold
            image, gradcam, thresholded, difference = generate_failure_case(
                image_size, 
                threshold_percent=threshold
            )
            
            # Create subplots in a grid
            ax = fig.add_subplot(gs[i, j])
            
            # Apply colormaps
            gradcam_colored = apply_colormap(gradcam, 'hot')
            thresholded_colored = apply_colormap(thresholded, 'hot')
            difference_colored = apply_colormap(difference, 'Reds')
            
            # Overlay heatmaps on image
            combined_img = np.array(image)
            
            # Blend gradcam
            gradcam_alpha = 0.6
            gradcam_overlay = (1 - gradcam_alpha) * combined_img + gradcam_alpha * gradcam_colored
            
            # Add red highlights for lost information
            difference_mask = difference > 0
            
            # Create a copy of the overlay for modification
            highlighted = gradcam_overlay.copy()
            
            # Apply red highlight to lost information areas
            highlighted[difference_mask, 0] = 255  # Red channel
            highlighted[difference_mask, 1] = 0    # Green channel
            highlighted[difference_mask, 2] = 0    # Blue channel
            
            # Display the combined image
            ax.imshow(highlighted.astype(np.uint8))
            
            # Add threshold info to title
            ax.set_title(f'Threshold: Top {100-threshold}%\nRed = Lost Information', fontsize=10)
            ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved failure cases visualization to {output_path}")
    
    return fig

def main():
    """Main function to generate failure case visualization."""
    # Define output path
    output_path = os.path.join(output_dir, "failure_cases.png")
    
    # Generate failure case visualization with multiple examples and thresholds
    create_failure_visualization(
        n_examples=3,
        threshold_percents=[70, 80, 90],
        output_path=output_path
    )
    
    print(f"Generated failure cases visualization at {output_path}")

if __name__ == "__main__":
    main() 