import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json
import shutil
from torchvision.datasets import ImageFolder

class CustomFourClassDataset(Dataset):
    """
    Custom 4-class dataset for benchmarking lightweight explainability
    Classes: animal, vehicle, furniture, food
    """
    
    def __init__(self, root_dir="custom_dataset", transform=None, download=False, 
                 num_samples_per_class=25, validation_split=0.2):
        """
        Initialize the dataset
        
        Args:
            root_dir: Directory to store/load the dataset
            transform: Image transformations
            download: Whether to download sample images
            num_samples_per_class: Number of images per class
            validation_split: Percentage of data for validation
        """
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = ['animal', 'vehicle', 'furniture', 'food']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Create directory structure
        if download and not os.path.exists(root_dir):
            self._create_dataset(num_samples_per_class)
            
        # Load dataset
        self.samples = []
        self.targets = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append(img_path)
                        self.targets.append(self.class_to_idx[class_name])
                        
        # Create segmentation masks for pointing game evaluation
        self.masks_dir = os.path.join(root_dir, 'masks')
        os.makedirs(self.masks_dir, exist_ok=True)
        
        # Create metadata file with annotations
        self.metadata_file = os.path.join(root_dir, 'metadata.json')
        if not os.path.exists(self.metadata_file):
            self._create_metadata()
            
        # Load metadata
        with open(self.metadata_file, 'r') as f:
            self.metadata = json.load(f)
            
    def _create_dataset(self, num_samples_per_class):
        """Create the dataset structure and download images from ImageNet"""
        from torchvision.datasets import ImageNet
        import random
        
        print(f"Creating custom 4-class dataset with {num_samples_per_class} samples per class")
        
        # Create directories
        os.makedirs(self.root_dir, exist_ok=True)
        for class_name in self.classes:
            os.makedirs(os.path.join(self.root_dir, class_name), exist_ok=True)
            
        # Map our classes to ImageNet classes
        imagenet_mappings = {
            'animal': ['n02123045', 'n02123159', 'n02124075', 'n02127052', 'n01503061', 'n01514859'],  # cats, dogs, birds
            'vehicle': ['n02701002', 'n02814533', 'n02930766', 'n03100240', 'n03790512', 'n04285008'],  # cars, buses, trains
            'furniture': ['n03001627', 'n03179701', 'n04099969', 'n03337140', 'n03761084', 'n03982430'],  # chairs, tables
            'food': ['n07753592', 'n07747607', 'n07614500', 'n07745940', 'n07749582', 'n07730033']  # food items
        }
        
        # For each class, copy images from pre-downloaded ImageNet
        try:
            imagenet_path = os.environ.get('IMAGENET_PATH', './imagenet')
            if not os.path.exists(imagenet_path):
                print("ImageNet dataset not found. Using random images instead.")
                self._create_random_images(num_samples_per_class)
                return
                
            for target_class, imagenet_classes in imagenet_mappings.items():
                samples_count = 0
                for imagenet_class in imagenet_classes:
                    imagenet_class_dir = os.path.join(imagenet_path, 'train', imagenet_class)
                    if os.path.exists(imagenet_class_dir):
                        images = [f for f in os.listdir(imagenet_class_dir) if f.endswith(('.JPEG', '.jpg', '.png'))]
                        selected_images = random.sample(images, min(num_samples_per_class // len(imagenet_classes) + 1, len(images)))
                        
                        for img in selected_images:
                            if samples_count >= num_samples_per_class:
                                break
                                
                            src = os.path.join(imagenet_class_dir, img)
                            dst = os.path.join(self.root_dir, target_class, f"{imagenet_class}_{img}")
                            shutil.copy(src, dst)
                            samples_count += 1
                            
            print(f"Created custom dataset with {sum(len(os.listdir(os.path.join(self.root_dir, c))) for c in self.classes)} images")
        except Exception as e:
            print(f"Error creating dataset from ImageNet: {e}")
            self._create_random_images(num_samples_per_class)
            
    def _create_random_images(self, num_samples_per_class):
        """Create random images for testing when no real images are available"""
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(num_samples_per_class):
                # Create a random color image
                img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                # Add a simple shape to make it more class-like
                if class_name == 'animal':
                    # Add oval shape
                    cv2.ellipse(img, (112, 112), (80, 60), 0, 0, 360, (120, 180, 120), -1)
                elif class_name == 'vehicle':
                    # Add rectangle
                    cv2.rectangle(img, (50, 70), (174, 154), (100, 100, 180), -1)
                elif class_name == 'furniture':
                    # Add chair-like shape
                    cv2.rectangle(img, (70, 90), (154, 180), (160, 120, 120), -1)
                    cv2.rectangle(img, (70, 50), (154, 90), (160, 120, 120), -1)
                elif class_name == 'food':
                    # Add circle
                    cv2.circle(img, (112, 112), 70, (120, 120, 180), -1)
                
                # Save the image
                img_path = os.path.join(class_dir, f"random_{i}.jpg")
                Image.fromarray(img).save(img_path)
                
    def _create_metadata(self):
        """Create metadata with bounding boxes and annotations for evaluation"""
        metadata = {"images": {}}
        
        for idx, img_path in enumerate(self.samples):
            img_name = os.path.basename(img_path)
            class_name = self.classes[self.targets[idx]]
            
            # Open image to get dimensions
            with Image.open(img_path) as img:
                width, height = img.size
                
            # Create a simple bounding box (60-80% of image)
            x_min = int(width * 0.1)
            y_min = int(height * 0.1)
            x_max = int(width * 0.9)
            y_max = int(height * 0.9)
            
            # Create a simple segmentation mask
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y_min:y_max, x_min:x_max] = 255
            
            # Save mask
            mask_path = os.path.join(self.masks_dir, img_name.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png'))
            Image.fromarray(mask).save(mask_path)
            
            # Add to metadata
            metadata["images"][img_name] = {
                "class": class_name,
                "bbox": [x_min, y_min, x_max, y_max],
                "mask": mask_path
            }
            
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        
        # Load image
        with Image.open(img_path).convert('RGB') as img:
            if self.transform:
                img = self.transform(img)
                
        # Get mask for pointing game evaluation
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(self.masks_dir, img_name.replace('.jpg', '_mask.png').replace('.jpeg', '_mask.png'))
        
        # Add metadata for evaluation
        metadata = {
            "path": img_path,
            "class": self.classes[target],
            "mask_path": mask_path
        }
                
        return img, target, metadata

def get_custom_dataset(batch_size=32, download=False):
    """Helper function to create and return data loaders for the custom dataset"""
    dataset = CustomFourClassDataset(download=download)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset.classes
    
if __name__ == "__main__":
    # Test the dataset creation
    import cv2
    
    # Create the dataset
    dataset = CustomFourClassDataset(download=True)
    
    # Display some statistics
    print(f"Dataset contains {len(dataset)} images")
    print(f"Classes: {dataset.classes}")
    
    # Show a sample image from each class
    for class_name in dataset.classes:
        class_idx = dataset.class_to_idx[class_name]
        samples = [i for i, t in enumerate(dataset.targets) if t == class_idx]
        
        if samples:
            img, _, metadata = dataset[samples[0]]
            img_path = metadata["path"]
            mask_path = metadata["mask_path"]
            
            # Load and display the image
            display_img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Overlay mask on image
            display_img[mask > 0] = display_img[mask > 0] * 0.7 + np.array([0, 0, 255]) * 0.3
            
            # Display class name on image
            cv2.putText(display_img, class_name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save the example
            output_dir = "examples"
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, f"{class_name}_example.jpg"), display_img)
            
    print("Dataset created and examples saved to 'examples' directory") 