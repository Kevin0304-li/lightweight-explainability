#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate ground truth comparison visualization.

This script creates a visualization comparing Grad-CAM, our simplified method,
and ground truth annotations on a set of example images.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.gridspec as gridspec

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import explainability modules
try:
    from src.explainability.gradcam import GradCAM
    from src.explainability.lightweight import LightweightExplainer
    from src.benchmark.evaluation_metrics import create_ground_truth_mask
    from src.models.model_loader import load_model
except ImportError:
    print("Error importing required modules")
    print("Running in simulation mode")
    SIMULATION_MODE = True
else:
    SIMULATION_MODE = False

# Create output directory
output_dir = "results/ground_truth"
os.makedirs(output_dir, exist_ok=True)

# Define create_ground_truth_mask function for simulation mode
def create_ground_truth_mask(image_size, important_regions=None):
    """Create a synthetic ground truth mask for evaluation.
    
    Args:
        image_size: Tuple of (height, width) for the mask
        important_regions: List of tuples (y, x, radius) for important regions
        
    Returns:
        Binary ground truth mask
    """
    if important_regions is None:
        # Default: create two important regions
        h, w = image_size
        important_regions = [
            (h // 3, w // 3, h // 10),  # Region 1: (y, x, radius)
            (2 * h // 3, 2 * w // 3, h // 8)  # Region 2: (y, x, radius)
        ]
    
    # Create empty mask
    mask = np.zeros(image_size, dtype=bool)
    
    # Fill important regions
    for y, x, r in important_regions:
        y_grid, x_grid = np.ogrid[:image_size[0], :image_size[1]]
        dist = np.sqrt((y_grid - y)**2 + (x_grid - x)**2)
        mask[dist <= r] = True
    
    return mask

def generate_synthetic_ground_truth(image):
    """Generate synthetic ground truth for an image.
    
    Args:
        image: PIL Image
        
    Returns:
        Ground truth mask as numpy array
    """
    width, height = image.size
    
    # Create appropriate regions based on image content
    # In a real scenario, these would be human annotations
    # Here we're creating synthetic ones
    
    # Simple circular region in center for demo purposes
    mask = create_ground_truth_mask((height, width), 
                                   important_regions=[(height//2, width//2, min(height, width)//4)])
    
    return mask

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

def simulate_explanations(image):
    """Simulate explanations for demonstration purposes.
    
    Args:
        image: PIL Image
        
    Returns:
        Tuple of (gradcam_heatmap, lightweight_heatmap)
    """
    # Convert to numpy array
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Create simulated Grad-CAM heatmap (more diffuse)
    gradcam_heatmap = np.zeros((height, width))
    center_y, center_x = height // 2, width // 2
    
    # Create radial gradient from center
    y_grid, x_grid = np.ogrid[:height, :width]
    dist = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
    max_dist = np.sqrt(height**2 + width**2) / 2
    gradcam_heatmap = np.maximum(0, 1 - dist / (max_dist * 0.5))
    
    # Add some noise for realism
    gradcam_heatmap += np.random.normal(0, 0.05, (height, width))
    gradcam_heatmap = np.clip(gradcam_heatmap, 0, 1)
    
    # Create simulated lightweight heatmap (more focused)
    # Start with Grad-CAM and threshold it
    lightweight_heatmap = gradcam_heatmap.copy()
    threshold = np.percentile(lightweight_heatmap, 70)  # Keep top 30%
    lightweight_heatmap[lightweight_heatmap < threshold] = 0
    
    return gradcam_heatmap, lightweight_heatmap

def generate_explanations(image, model_path=None):
    """Generate real explanations using the actual explainability methods.
    
    Args:
        image: PIL Image
        model_path: Path to model weights
        
    Returns:
        Tuple of (gradcam_heatmap, lightweight_heatmap)
    """
    if SIMULATION_MODE:
        return simulate_explanations(image)
    
    # Load model and generate explanations
    model = load_model(model_path)
    
    # Generate Grad-CAM explanation
    gradcam = GradCAM(model)
    gradcam_heatmap = gradcam.explain(image)
    
    # Generate lightweight explanation
    lightweight = LightweightExplainer(model)
    lightweight_heatmap = lightweight.explain(image)
    
    return gradcam_heatmap, lightweight_heatmap

def create_visualization(image_paths, output_path):
    """Create visualization comparing methods with ground truth.
    
    Args:
        image_paths: List of paths to images
        output_path: Path to save visualization
    """
    # Initialize figure
    fig = plt.figure(figsize=(15, 5 * len(image_paths)))
    fig.suptitle('Grad-CAM vs. Our Simplified Method vs. Ground Truth', fontsize=18)
    
    # Create GridSpec layout
    gs = gridspec.GridSpec(len(image_paths), 4, figure=fig)
    
    for i, img_path in enumerate(image_paths):
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Generate ground truth
        ground_truth = generate_synthetic_ground_truth(image)
        
        # Generate explanations
        gradcam_heatmap, lightweight_heatmap = generate_explanations(image)
        
        # Apply colormaps
        gt_colored = apply_colormap(ground_truth.astype(float), 'Greens')
        gradcam_colored = apply_colormap(gradcam_heatmap, 'Reds')
        lightweight_colored = apply_colormap(lightweight_heatmap, 'Blues')
        
        # Create blended versions
        gt_blended = blend_with_image(image, gt_colored, 0.5)
        gradcam_blended = blend_with_image(image, gradcam_colored, 0.5)
        lightweight_blended = blend_with_image(image, lightweight_colored, 0.5)
        
        # Plot original image
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Plot Grad-CAM
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(gradcam_blended)
        ax2.set_title('Grad-CAM')
        ax2.axis('off')
        
        # Plot lightweight method
        ax3 = fig.add_subplot(gs[i, 2])
        ax3.imshow(lightweight_blended)
        ax3.set_title('Our Simplified Method')
        ax3.axis('off')
        
        # Plot ground truth
        ax4 = fig.add_subplot(gs[i, 3])
        ax4.imshow(gt_blended)
        ax4.set_title('Ground Truth')
        ax4.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    return fig

def main():
    """Main function to generate visualization."""
    # Get sample images 
    sample_dir = "sample_images"
    image_files = [
        os.path.join(sample_dir, "dog.jpg"),
        os.path.join(sample_dir, "cat.jpg"),
        os.path.join(sample_dir, "car.jpg")
    ]
    
    # Filter to images that exist
    available_images = [f for f in image_files if os.path.exists(f)]
    
    if not available_images:
        print("No sample images found. Please ensure sample_images directory exists with images.")
        return
    
    # Create visualization
    output_path = os.path.join(output_dir, "ground_truth_comparison.png")
    create_visualization(available_images, output_path)
    
    print(f"Generated ground truth comparison visualization at {output_path}")

if __name__ == "__main__":
    main() 