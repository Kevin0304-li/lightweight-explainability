import os
import argparse
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch
import cv2
import qrcode
from datetime import datetime

from lightweight_explainability import ExplainableModel, preprocess_image

class UserStudy:
    """Class to simulate a user study for explainable AI"""
    
    def __init__(self, output_dir='./results/user_study', threshold_pcts=[5, 10, 20]):
        self.output_dir = output_dir
        self.threshold_pcts = threshold_pcts
        self.model = ExplainableModel()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'user_ratings': [],
            'localization_accuracy': [],
            'completion_time': [],
            'thresholds': threshold_pcts
        }
        
    def load_images(self, data_dir='./examples', num_samples=2):
        """Load sample images for the study"""
        images = []
        
        # Get list of class directories
        classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        for class_name in classes:
            class_dir = os.path.join(data_dir, class_name)
            files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Randomly select samples
            selected = random.sample(files, min(num_samples, len(files)))
            
            for file in selected:
                file_path = os.path.join(class_dir, file)
                img = Image.open(file_path).convert('RGB')
                
                images.append({
                    'image': img,
                    'path': file_path,
                    'class': class_name,
                    'id': f"{class_name}_{os.path.basename(file)}"
                })
        
        print(f"Loaded {len(images)} images for the user study")
        return images
    
    def generate_simulated_click(self, heatmap, noise_level=0.1):
        """Simulate a user click on the important area of the heatmap
        
        Args:
            heatmap: The heatmap array
            noise_level: Amount of random noise to add (0-1)
            
        Returns:
            (x, y): Coordinates of the click
        """
        # Find maximum activation point
        max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Add some noise to simulate user variation
        h, w = heatmap.shape
        noise_x = int(random.uniform(-w * noise_level, w * noise_level))
        noise_y = int(random.uniform(-h * noise_level, h * noise_level))
        
        click_x = max(0, min(w-1, max_x + noise_x))
        click_y = max(0, min(h-1, max_y + noise_y))
        
        return (click_x, click_y)
    
    def calculate_localization_accuracy(self, click, heatmap, threshold=0.5):
        """Calculate how accurate the user's click was
        
        Args:
            click: (x, y) coordinates of the click
            heatmap: The heatmap array
            threshold: Threshold for considering area as important
            
        Returns:
            accuracy: Score between 0-1
        """
        # Normalize heatmap
        norm_heatmap = heatmap / np.max(heatmap)
        
        # Create binary mask of important areas
        important_mask = norm_heatmap > threshold
        
        # Check if click is in important area
        x, y = click
        if y < 0 or y >= heatmap.shape[0] or x < 0 or x >= heatmap.shape[1]:
            return 0.0
        
        if important_mask[y, x]:
            # Calculate how close to the maximum value
            return norm_heatmap[y, x]
        else:
            # If not in important area, calculate distance to nearest important pixel
            if not np.any(important_mask):
                return 0.0
                
            # Get coordinates of important pixels
            important_coords = np.argwhere(important_mask)
            
            # Calculate distances to all important pixels
            distances = np.sqrt(np.sum(np.square(important_coords - np.array([y, x])), axis=1))
            min_distance = np.min(distances)
            
            # Convert distance to accuracy (higher distance = lower accuracy)
            max_possible_distance = np.sqrt(heatmap.shape[0]**2 + heatmap.shape[1]**2)
            distance_penalty = min_distance / max_possible_distance
            
            return max(0, 1 - distance_penalty)
    
    def simulate_user_rating(self, orig_heatmap, simplified_heatmap, ground_truth=None):
        """Simulate a user rating for explanation quality
        
        Args:
            orig_heatmap: Original Grad-CAM heatmap
            simplified_heatmap: Simplified heatmap
            ground_truth: Optional ground truth mask
            
        Returns:
            rating: Simulated user rating (1-5)
        """
        # Normalize heatmaps
        orig_norm = orig_heatmap / np.max(orig_heatmap) if np.max(orig_heatmap) > 0 else orig_heatmap
        simp_norm = simplified_heatmap / np.max(simplified_heatmap) if np.max(simplified_heatmap) > 0 else simplified_heatmap
        
        # Calculate structural similarity
        # Higher similarity = higher rating, as it preserves the original explanation
        mse = np.mean((orig_norm - simp_norm) ** 2)
        similarity = 1 - mse
        
        # Calculate compactness of simplified heatmap
        non_zero = np.count_nonzero(simp_norm)
        compactness = 1 - (non_zero / simp_norm.size)
        
        # If ground truth available, calculate accuracy
        if ground_truth is not None:
            # Calculate true positive rates for important pixels
            important_orig = orig_norm > 0.5
            important_simp = simp_norm > 0.5
            
            true_positives = np.logical_and(important_simp, ground_truth).sum()
            false_positives = np.logical_and(important_simp, np.logical_not(ground_truth)).sum()
            
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 0
                
            # Higher precision = higher rating
            accuracy_factor = precision
        else:
            # Without ground truth, assume neutral accuracy
            accuracy_factor = 0.5
        
        # Calculate final rating based on compactness and similarity
        # Users typically prefer explanations that are:
        # 1. Similar to the original (preserves key information)
        # 2. More compact (easier to understand)
        # 3. More accurate (if ground truth available)
        rating = 2 + similarity * 1.5 + compactness * 1.0 + accuracy_factor * 0.5
        
        # Clamp to 1-5 range and round to nearest 0.5
        rating = max(1, min(5, rating))
        rating = round(rating * 2) / 2
        
        return rating
    
    def run(self, num_images=5, participants=20):
        """Run the simulated user study
        
        Args:
            num_images: Number of images to use per class
            participants: Number of simulated participants
            
        Returns:
            results: Dictionary of study results
        """
        # Load images
        images = self.load_images(num_samples=num_images)
        
        # Create study ID
        study_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_dir = os.path.join(self.output_dir, study_id)
        os.makedirs(study_dir, exist_ok=True)
        
        # Create subdirectories
        images_dir = os.path.join(study_dir, 'images')
        results_dir = os.path.join(study_dir, 'results')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create QR code for the study
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(f"Explainable AI User Study: {study_id}")
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")
        qr_path = os.path.join(study_dir, 'study_qr.png')
        qr_img.save(qr_path)
        
        print(f"Created user study: {study_id}")
        print(f"QR code saved to: {qr_path}")
        
        # Initialize results for each threshold
        threshold_results = {t: {
            'ratings': [],
            'accuracy': [],
            'time': []
        } for t in self.threshold_pcts}
        
        # Process each image for each participant
        for participant_id in tqdm(range(participants), desc="Simulating participants"):
            # Randomly select images to show this participant
            participant_images = random.sample(images, min(5, len(images)))
            participant_thresholds = {}  # Track which threshold was shown for each image
            
            for i, sample in enumerate(participant_images):
                img = sample['image']
                img_id = sample['id']
                
                # Preprocess image for model
                img_tensor = preprocess_image(img)
                
                # Generate predictions and baseline CAM
                prediction, confidence = self.model.predict(img_tensor)
                baseline_cam = self.model.generate_gradcam(img_tensor)
                
                # Randomly select a threshold for this image-participant pair
                threshold = random.choice(self.threshold_pcts)
                participant_thresholds[img_id] = threshold
                
                # Generate simplified CAM
                start_time = time.time()
                simplified_cam = self.model.simplified_gradcam(img_tensor, threshold_pct=threshold)
                processing_time = time.time() - start_time
                
                # Save original and simplified images
                img_path = os.path.join(images_dir, f"p{participant_id}_img{i}_{img_id}.png")
                img.save(img_path)
                
                baseline_path = os.path.join(images_dir, f"p{participant_id}_img{i}_{img_id}_baseline.png")
                simplified_path = os.path.join(images_dir, f"p{participant_id}_img{i}_{img_id}_simplified_{threshold}.png")
                
                self.model.visualize_explanation(img, baseline_cam, 
                                               title=f"Baseline: {prediction}", 
                                               save_path=baseline_path)
                                               
                self.model.visualize_explanation(img, simplified_cam, 
                                               title=f"Simplified ({threshold}%): {prediction}", 
                                               save_path=simplified_path)
                
                # Simulate user interaction
                # 1. Simulate user click on what they think is important
                click = self.generate_simulated_click(simplified_cam)
                
                # 2. Calculate localization accuracy
                accuracy = self.calculate_localization_accuracy(click, baseline_cam)
                
                # 3. Simulate user rating
                rating = self.simulate_user_rating(baseline_cam, simplified_cam)
                
                # Record results for this image
                result = {
                    'participant_id': participant_id,
                    'image_id': img_id,
                    'prediction': prediction,
                    'confidence': confidence,
                    'threshold': threshold,
                    'click': click,
                    'accuracy': accuracy,
                    'rating': rating,
                    'processing_time': processing_time
                }
                
                # Save result to file
                result_path = os.path.join(results_dir, f"p{participant_id}_img{i}_{img_id}_result.txt")
                with open(result_path, 'w') as f:
                    for key, value in result.items():
                        f.write(f"{key}: {value}\n")
                
                # Aggregate results by threshold
                threshold_results[threshold]['ratings'].append(rating)
                threshold_results[threshold]['accuracy'].append(accuracy)
                threshold_results[threshold]['time'].append(processing_time)
        
        # Generate summary report
        self.generate_report(threshold_results, study_dir)
        
        return threshold_results
    
    def generate_report(self, results, output_dir):
        """Generate a report of the user study results"""
        report_path = os.path.join(output_dir, 'user_study_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# User Study Results\n\n")
            
            # Overall metrics
            f.write("## Overall Metrics\n\n")
            f.write("| Threshold | Average Rating | Localization Accuracy | Processing Time (s) |\n")
            f.write("|-----------|---------------|------------------------|---------------------|\n")
            
            for threshold in sorted(results.keys()):
                avg_rating = np.mean(results[threshold]['ratings'])
                avg_accuracy = np.mean(results[threshold]['accuracy'])
                avg_time = np.mean(results[threshold]['time'])
                
                f.write(f"| {threshold}% | {avg_rating:.2f} | {avg_accuracy:.4f} | {avg_time:.4f} |\n")
            
            f.write("\n")
            
            # Create plots directory
            plots_dir = os.path.join(output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot rating distribution
            plt.figure(figsize=(10, 6))
            for threshold in sorted(results.keys()):
                ratings = results[threshold]['ratings']
                # Plot histogram with KDE
                plt.hist(ratings, alpha=0.5, label=f"{threshold}% Threshold", bins=9)
            
            plt.xlabel('User Rating (1-5)')
            plt.ylabel('Count')
            plt.title('Distribution of User Ratings')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            rating_plot_path = os.path.join(plots_dir, 'rating_distribution.png')
            plt.savefig(rating_plot_path)
            plt.close()
            
            # Add plot to report
            f.write(f"## Rating Distribution\n\n")
            f.write(f"![Rating Distribution](plots/rating_distribution.png)\n\n")
            
            # Plot accuracy vs processing time
            plt.figure(figsize=(10, 6))
            for threshold in sorted(results.keys()):
                accuracy = results[threshold]['accuracy']
                times = results[threshold]['time']
                
                plt.scatter(times, accuracy, alpha=0.7, label=f"{threshold}% Threshold")
            
            plt.xlabel('Processing Time (s)')
            plt.ylabel('Localization Accuracy')
            plt.title('Accuracy vs Processing Time')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            accuracy_plot_path = os.path.join(plots_dir, 'accuracy_vs_time.png')
            plt.savefig(accuracy_plot_path)
            plt.close()
            
            # Add plot to report
            f.write(f"## Accuracy vs Processing Time\n\n")
            f.write(f"![Accuracy vs Time](plots/accuracy_vs_time.png)\n\n")
            
            # Plot average metrics
            thresholds = sorted(results.keys())
            avg_ratings = [np.mean(results[t]['ratings']) for t in thresholds]
            avg_accuracy = [np.mean(results[t]['accuracy']) for t in thresholds]
            
            plt.figure(figsize=(10, 6))
            
            plt.plot(thresholds, avg_ratings, 'o-', linewidth=2, label='Avg Rating')
            plt.plot(thresholds, avg_accuracy, 's-', linewidth=2, label='Avg Accuracy')
            
            plt.xlabel('Threshold (%)')
            plt.ylabel('Score')
            plt.title('Average Metrics by Threshold')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            metrics_plot_path = os.path.join(plots_dir, 'avg_metrics_by_threshold.png')
            plt.savefig(metrics_plot_path)
            plt.close()
            
            # Add plot to report
            f.write(f"## Average Metrics by Threshold\n\n")
            f.write(f"![Average Metrics](plots/avg_metrics_by_threshold.png)\n\n")
            
            # Add summary and conclusions
            f.write("## Summary and Conclusions\n\n")
            
            # Find best threshold by rating
            best_rating_threshold = max(thresholds, key=lambda t: np.mean(results[t]['ratings']))
            
            # Find best threshold by accuracy
            best_accuracy_threshold = max(thresholds, key=lambda t: np.mean(results[t]['accuracy']))
            
            # Find fastest threshold
            fastest_threshold = min(thresholds, key=lambda t: np.mean(results[t]['time']))
            
            f.write(f"- Best threshold for user satisfaction: **{best_rating_threshold}%** (avg rating: {np.mean(results[best_rating_threshold]['ratings']):.2f})\n")
            f.write(f"- Best threshold for localization accuracy: **{best_accuracy_threshold}%** (avg accuracy: {np.mean(results[best_accuracy_threshold]['accuracy']):.4f})\n")
            f.write(f"- Fastest processing: **{fastest_threshold}%** (avg time: {np.mean(results[fastest_threshold]['time']):.4f}s)\n\n")
            
            f.write("These results indicate the trade-offs between processing speed, accuracy, and user satisfaction ")
            f.write("when selecting a threshold for simplified Grad-CAM explanations.\n")
        
        print(f"Report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Run a simulated user study')
    parser.add_argument('--num_participants', type=int, default=20, 
                        help='Number of simulated participants')
    parser.add_argument('--num_images', type=int, default=3, 
                        help='Number of images per class')
    parser.add_argument('--thresholds', type=str, default='5,10,20', 
                        help='Comma-separated list of thresholds to test')
    parser.add_argument('--output_dir', type=str, default='./results/user_study', 
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Parse thresholds
    thresholds = [int(t) for t in args.thresholds.split(',')]
    
    # Create and run user study
    study = UserStudy(output_dir=args.output_dir, threshold_pcts=thresholds)
    results = study.run(num_images=args.num_images, participants=args.num_participants)
    
    print("User study simulation complete!")
    
if __name__ == '__main__':
    main() 