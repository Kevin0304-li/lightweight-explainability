import os
import sys
import shutil
from PIL import Image
import numpy as np
import argparse

def create_color_image(width, height, color):
    """Create a simple colored image"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    return Image.fromarray(img)

def create_examples(output_dir, num_images_per_class=10):
    """Create a simple dataset with class directories"""
    classes = ['animal', 'vehicle', 'furniture', 'food']
    colors = {
        'animal': (120, 180, 120),  # Green
        'vehicle': (100, 100, 180),  # Blue
        'furniture': (160, 120, 120),  # Brown
        'food': (180, 120, 180)  # Purple
    }
    
    print(f"Creating example images in {output_dir}")
    
    # Create the main directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create class directories and example images
    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(num_images_per_class):
            # Create a simple colored image
            img = create_color_image(224, 224, colors[class_name])
            
            # Save the image
            img_path = os.path.join(class_dir, f"{class_name}_{i+1}.jpg")
            img.save(img_path)
            
            print(f"Created {img_path}")
    
    print(f"Created {num_images_per_class * len(classes)} example images")

def reorganize_existing_examples(output_dir):
    """Move existing example images into class directories"""
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist")
        return
        
    # Create class directories if they don't exist
    classes = ['animal', 'vehicle', 'furniture', 'food']
    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # Find existing example images
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        
        # Skip directories
        if os.path.isdir(filepath):
            continue
            
        # Check if it's an image file
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        # Determine class from filename
        for class_name in classes:
            if class_name in filename.lower():
                # Move to appropriate class directory
                destination = os.path.join(output_dir, class_name, filename)
                print(f"Moving {filename} to {class_name} directory")
                shutil.move(filepath, destination)
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate example images for benchmark testing")
    parser.add_argument("--output_dir", type=str, default="examples", help="Output directory")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images per class")
    parser.add_argument("--reorganize", action="store_true", help="Reorganize existing examples")
    args = parser.parse_args()
    
    if args.reorganize:
        reorganize_existing_examples(args.output_dir)
    else:
        create_examples(args.output_dir, args.num_images) 