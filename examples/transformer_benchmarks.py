#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive benchmark for transformer explainability across different domains.

This script evaluates the performance and quality of transformer explainability
across different transformer architectures and domains:
1. Vision Transformers (ViT, Swin, DeiT)
2. Text Transformers (BERT, RoBERTa, DistilBERT)
3. Multimodal Transformers (CLIP)

Metrics:
- Processing time
- Memory usage
- Explanation quality via attention visualization
- Comparison of different attention head selection strategies
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import tracemalloc
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our transformer explainability module
from transformer_explainability import TransformerExplainer, simplify_attention_map

# Create output directory
output_dir = "results/transformer_benchmarks"
os.makedirs(output_dir, exist_ok=True)

# Try to import transformers library
try:
    import transformers
    from transformers import (
        ViTForImageClassification, 
        AutoFeatureExtractor,
        BertForSequenceClassification,
        AutoTokenizer,
        SwinForImageClassification,
        DeiTForImageClassification,
        RobertaForSequenceClassification,
        DistilBertForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Transformers library not found. Install with 'pip install transformers'")
    TRANSFORMERS_AVAILABLE = False

# Try to import CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    print("CLIP not available. Install with 'pip install git+https://github.com/openai/CLIP.git'")
    CLIP_AVAILABLE = False

def measure_memory_usage(func, *args, **kwargs):
    """Measure peak memory usage of a function."""
    tracemalloc.start()
    result = func(*args, **kwargs)
    peak_memory = tracemalloc.get_traced_memory()[1] / (1024 * 1024)  # in MB
    tracemalloc.stop()
    return result, peak_memory

def measure_execution_time(func, *args, **kwargs):
    """Measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    return result, execution_time

class TransformerBenchmark:
    """Benchmark for transformer explainability across architectures and domains."""
    
    def __init__(self):
        """Initialize benchmark with different transformer models."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.processors = {}
        self.explainers = {}
        
        # Initialize results tracking
        self.results = {
            'execution_time': defaultdict(list),
            'memory_usage': defaultdict(list),
            'head_analysis': defaultdict(list),
            'token_importance': defaultdict(list),
        }
        
        # Load models if available
        if TRANSFORMERS_AVAILABLE:
            self._load_vision_transformers()
            self._load_text_transformers()
        
        if CLIP_AVAILABLE:
            self._load_clip()
    
    def _load_vision_transformers(self):
        """Load different vision transformer models."""
        print("Loading Vision Transformer models...")
        
        try:
            # ViT base model
            vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            vit_processor = AutoFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
            self.models['vit'] = vit_model.to(self.device)
            self.processors['vit'] = vit_processor
            self.explainers['vit'] = TransformerExplainer(vit_model, patch_size=16, model_family='vit')
            print("  - ViT model loaded successfully")
        except Exception as e:
            print(f"  - Failed to load ViT model: {e}")
        
        try:
            # Swin Transformer
            swin_model = SwinForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
            swin_processor = AutoFeatureExtractor.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
            self.models['swin'] = swin_model.to(self.device)
            self.processors['swin'] = swin_processor
            self.explainers['swin'] = TransformerExplainer(swin_model, patch_size=4, model_family='swin')
            print("  - Swin Transformer loaded successfully")
        except Exception as e:
            print(f"  - Failed to load Swin Transformer: {e}")
        
        try:
            # DeiT model
            deit_model = DeiTForImageClassification.from_pretrained('facebook/deit-base-patch16-224')
            deit_processor = AutoFeatureExtractor.from_pretrained('facebook/deit-base-patch16-224')
            self.models['deit'] = deit_model.to(self.device)
            self.processors['deit'] = deit_processor
            self.explainers['deit'] = TransformerExplainer(deit_model, patch_size=16, model_family='deit')
            print("  - DeiT model loaded successfully")
        except Exception as e:
            print(f"  - Failed to load DeiT model: {e}")
    
    def _load_text_transformers(self):
        """Load different text transformer models."""
        print("Loading Text Transformer models...")
        
        try:
            # BERT model
            bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
            bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.models['bert'] = bert_model.to(self.device)
            self.processors['bert'] = bert_tokenizer
            self.explainers['bert'] = TransformerExplainer(bert_model, model_family='bert')
            print("  - BERT model loaded successfully")
        except Exception as e:
            print(f"  - Failed to load BERT model: {e}")
        
        try:
            # RoBERTa model
            roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base')
            roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
            self.models['roberta'] = roberta_model.to(self.device)
            self.processors['roberta'] = roberta_tokenizer
            self.explainers['roberta'] = TransformerExplainer(roberta_model, model_family='roberta')
            print("  - RoBERTa model loaded successfully")
        except Exception as e:
            print(f"  - Failed to load RoBERTa model: {e}")
        
        try:
            # DistilBERT model
            distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
            distilbert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.models['distilbert'] = distilbert_model.to(self.device)
            self.processors['distilbert'] = distilbert_tokenizer
            self.explainers['distilbert'] = TransformerExplainer(distilbert_model, model_family='distilbert')
            print("  - DistilBERT model loaded successfully")
        except Exception as e:
            print(f"  - Failed to load DistilBERT model: {e}")
    
    def _load_clip(self):
        """Load CLIP model."""
        print("Loading CLIP model...")
        
        try:
            clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.device)
            self.models['clip'] = clip_model
            self.processors['clip'] = clip_preprocess
            self.explainers['clip'] = TransformerExplainer(clip_model, patch_size=32, model_family='clip')
            print("  - CLIP model loaded successfully")
        except Exception as e:
            print(f"  - Failed to load CLIP model: {e}")
    
    def preprocess_image(self, image_path, model_name):
        """Preprocess image for specified model."""
        image = Image.open(image_path).convert('RGB')
        
        if model_name == 'clip':
            # CLIP has its own preprocessing
            if CLIP_AVAILABLE:
                inputs = self.processors[model_name](image).unsqueeze(0).to(self.device)
                return image, inputs
        else:
            # Use transformers processor
            if TRANSFORMERS_AVAILABLE:
                inputs = self.processors[model_name](images=image, return_tensors="pt").to(self.device)
                return image, inputs
        
        # Fallback to simple preprocessing
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        inputs = preprocess(image).unsqueeze(0).to(self.device)
        return image, inputs
    
    def preprocess_text(self, text, model_name):
        """Preprocess text for specified model."""
        if TRANSFORMERS_AVAILABLE and model_name in self.processors:
            inputs = self.processors[model_name](text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            return inputs
        return None
    
    def explain_vision_transformer(self, model_name, image_path, threshold_percent=10):
        """Generate explanation for vision transformer."""
        if model_name not in self.models or model_name not in self.explainers:
            print(f"Model {model_name} not available")
            return None
        
        # Preprocess image
        image, inputs = self.preprocess_image(image_path, model_name)
        
        # Generate explanation
        if model_name == 'clip':
            # For CLIP, we need to handle the image-text model differently
            explanation = self.explainers[model_name].explain(
                inputs, 
                target_class=None, 
                threshold_percent=threshold_percent,
                analyze_heads=True
            )
        else:
            # Standard vision transformer
            pixel_values = inputs.pixel_values if hasattr(inputs, 'pixel_values') else inputs
            explanation = self.explainers[model_name].explain(
                pixel_values, 
                target_class=None, 
                threshold_percent=threshold_percent,
                analyze_heads=True
            )
        
        return explanation, image
    
    def explain_text_transformer(self, model_name, text, threshold_percent=10):
        """Generate explanation for text transformer."""
        if model_name not in self.models or model_name not in self.explainers:
            print(f"Model {model_name} not available")
            return None
        
        # Preprocess text
        inputs = self.preprocess_text(text, model_name)
        if inputs is None:
            return None
        
        # Generate explanation
        explanation = self.explainers[model_name].explain(
            inputs, 
            target_class=None, 
            threshold_percent=threshold_percent,
            analyze_heads=True
        )
        
        # Analyze token importance
        tokens = self.processors[model_name].tokenize(text)
        token_analysis = self.explainers[model_name].analyze_text_attention(
            explanation.get('attention_maps', []), 
            tokens
        )
        
        explanation['token_analysis'] = token_analysis
        return explanation, tokens
    
    def benchmark_vision_transformers(self, image_paths):
        """Benchmark vision transformer models on multiple images."""
        vision_models = [model for model in self.models.keys() 
                        if model in ['vit', 'swin', 'deit', 'clip']]
        
        if not vision_models:
            print("No vision transformer models available")
            return
        
        print(f"Benchmarking {len(vision_models)} vision transformer models on {len(image_paths)} images...")
        
        for model_name in vision_models:
            print(f"\nBenchmarking {model_name.upper()}...")
            
            for image_path in tqdm(image_paths, desc=f"Processing images"):
                # Measure execution time and memory
                explanation_func = lambda: self.explain_vision_transformer(model_name, image_path)
                
                explanation_result, memory_usage = measure_memory_usage(explanation_func)
                if explanation_result is None:
                    continue
                    
                explanation, image = explanation_result
                execution_time = explanation.get('processing_time', 0)
                
                # Record results
                self.results['execution_time'][model_name].append(execution_time)
                self.results['memory_usage'][model_name].append(memory_usage)
                
                # Analyze attention heads
                head_analysis = explanation.get('head_analysis', None)
                if head_analysis and 'top_heads' in head_analysis:
                    self.results['head_analysis'][model_name].append(head_analysis)
        
        # Generate visualization of one example
        if image_paths:
            self._visualize_vision_transformer_comparison(image_paths[0])
    
    def benchmark_text_transformers(self, texts):
        """Benchmark text transformer models on multiple texts."""
        text_models = [model for model in self.models.keys() 
                      if model in ['bert', 'roberta', 'distilbert']]
        
        if not text_models:
            print("No text transformer models available")
            return
        
        print(f"Benchmarking {len(text_models)} text transformer models on {len(texts)} texts...")
        
        for model_name in text_models:
            print(f"\nBenchmarking {model_name.upper()}...")
            
            for text in tqdm(texts, desc=f"Processing texts"):
                # Measure execution time and memory
                explanation_func = lambda: self.explain_text_transformer(model_name, text)
                
                explanation_result, memory_usage = measure_memory_usage(explanation_func)
                if explanation_result is None:
                    continue
                    
                explanation, tokens = explanation_result
                execution_time = explanation.get('processing_time', 0)
                
                # Record results
                self.results['execution_time'][model_name].append(execution_time)
                self.results['memory_usage'][model_name].append(memory_usage)
                
                # Record token importance analysis
                token_analysis = explanation.get('token_analysis', None)
                if token_analysis and 'top_tokens' in token_analysis:
                    self.results['token_importance'][model_name].append(token_analysis)
        
        # Generate visualization of one example
        if texts:
            self._visualize_text_transformer_comparison(texts[0])
    
    def _visualize_vision_transformer_comparison(self, image_path):
        """Generate visualization comparing vision transformer explanations."""
        vision_models = [model for model in self.models.keys() 
                        if model in ['vit', 'swin', 'deit', 'clip']]
        
        if not vision_models:
            return
        
        # Set up plot
        fig, axes = plt.subplots(1, len(vision_models) + 1, figsize=(4 * (len(vision_models) + 1), 4))
        
        # Display original image
        original_image = Image.open(image_path).convert('RGB')
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Generate and display explanations
        for i, model_name in enumerate(vision_models):
            try:
                explanation_result = self.explain_vision_transformer(model_name, image_path)
                
                if explanation_result is None:
                    axes[i + 1].text(0.5, 0.5, "Failed to generate", ha='center', va='center')
                    axes[i + 1].set_title(f"{model_name.upper()}")
                    axes[i + 1].axis('off')
                    continue
                
                explanation, _ = explanation_result
                
                # Get attention heatmap
                if 'attention_heatmap' in explanation:
                    heatmap = explanation['attention_heatmap']
                    axes[i + 1].imshow(heatmap, cmap='jet')
                    
                    # Add performance info
                    processing_time = explanation.get('processing_time', 0)
                    memory_usage = np.mean(self.results['memory_usage'].get(model_name, [0]))
                    
                    title = f"{model_name.upper()}\n"
                    title += f"Time: {processing_time:.3f}s\n"
                    title += f"Memory: {memory_usage:.1f}MB"
                    axes[i + 1].set_title(title)
                else:
                    axes[i + 1].text(0.5, 0.5, "No heatmap available", ha='center', va='center')
                    axes[i + 1].set_title(f"{model_name.upper()}")
                
                axes[i + 1].axis('off')
                
            except Exception as e:
                print(f"Error visualizing {model_name}: {e}")
                axes[i + 1].text(0.5, 0.5, "Error", ha='center', va='center')
                axes[i + 1].set_title(f"{model_name.upper()}")
                axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'vision_transformer_comparison.png'), dpi=150)
        plt.close()
        
        print(f"Vision transformer comparison saved to {output_dir}/vision_transformer_comparison.png")
    
    def _visualize_text_transformer_comparison(self, text):
        """Generate visualization comparing text transformer explanations."""
        text_models = [model for model in self.models.keys() 
                      if model in ['bert', 'roberta', 'distilbert']]
        
        if not text_models:
            return
        
        # Visualize token importance for each model
        plt.figure(figsize=(12, 4 * len(text_models)))
        
        for i, model_name in enumerate(text_models):
            try:
                explanation_result = self.explain_text_transformer(model_name, text)
                
                if explanation_result is None:
                    continue
                
                explanation, tokens = explanation_result
                token_analysis = explanation.get('token_analysis', {})
                
                if 'top_tokens' in token_analysis:
                    # Extract token importances
                    tokens_list = [t[0] for t in token_analysis['top_tokens']]
                    importances = [t[1] for t in token_analysis['top_tokens']]
                    
                    # Plot as horizontal bar chart
                    plt.subplot(len(text_models), 1, i+1)
                    plt.barh(tokens_list, importances)
                    plt.xlabel('Importance')
                    
                    # Add performance info
                    processing_time = explanation.get('processing_time', 0)
                    memory_usage = np.mean(self.results['memory_usage'].get(model_name, [0]))
                    
                    title = f"{model_name.upper()} Token Importance\n"
                    title += f"Time: {processing_time:.3f}s - Memory: {memory_usage:.1f}MB"
                    plt.title(title)
                    plt.tight_layout()
            
            except Exception as e:
                print(f"Error visualizing {model_name}: {e}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'text_transformer_comparison.png'), dpi=150)
        plt.close()
        
        print(f"Text transformer comparison saved to {output_dir}/text_transformer_comparison.png")
    
    def generate_performance_report(self):
        """Generate performance report with tables and charts."""
        # Create DataFrame for performance data
        performance_data = []
        
        for model_name in self.models.keys():
            # Calculate average metrics
            avg_time = np.mean(self.results['execution_time'].get(model_name, [0]))
            avg_memory = np.mean(self.results['memory_usage'].get(model_name, [0]))
            
            # Determine model type
            if model_name in ['vit', 'swin', 'deit']:
                model_type = 'Vision Transformer'
            elif model_name in ['bert', 'roberta', 'distilbert']:
                model_type = 'Text Transformer'
            elif model_name == 'clip':
                model_type = 'Multimodal Transformer'
            else:
                model_type = 'Unknown'
            
            # Add to data
            performance_data.append({
                'Model': model_name.upper(),
                'Type': model_type,
                'Avg Time (s)': avg_time,
                'Avg Memory (MB)': avg_memory,
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(performance_data)
        
        # Save to CSV
        df.to_csv(os.path.join(output_dir, 'transformer_performance.csv'), index=False)
        
        # Create performance visualization
        plt.figure(figsize=(12, 8))
        
        # Time comparison
        plt.subplot(2, 1, 1)
        
        # Use seaborn for better styling
        sns.barplot(x='Model', y='Avg Time (s)', hue='Type', data=df)
        plt.title('Average Processing Time by Transformer Type')
        plt.yscale('log')  # Log scale often helps with time comparisons
        plt.grid(True, alpha=0.3)
        
        # Memory comparison
        plt.subplot(2, 1, 2)
        sns.barplot(x='Model', y='Avg Memory (MB)', hue='Type', data=df)
        plt.title('Average Memory Usage by Transformer Type')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'transformer_performance_comparison.png'), dpi=150)
        plt.close()
        
        # Save as markdown table
        with open(os.path.join(output_dir, 'transformer_performance.md'), 'w') as f:
            f.write("# Transformer Explainability Performance\n\n")
            
            # Print main metrics
            f.write("| Model | Type | Avg Time (s) | Avg Memory (MB) |\n")
            f.write("|-------|------|-------------|----------------|\n")
            
            for _, row in df.iterrows():
                f.write(f"| {row['Model']} | {row['Type']} | {row['Avg Time (s)']:.4f} | {row['Avg Memory (MB)']:.2f} |\n")
        
        print(f"Performance report saved to {output_dir}/transformer_performance.md")

def main():
    """Run transformer benchmarks."""
    # Check if transformers library is available
    if not TRANSFORMERS_AVAILABLE:
        print("Error: Transformers library is required for this benchmark.")
        print("Install with 'pip install transformers'")
        return
    
    # Initialize benchmark
    benchmark = TransformerBenchmark()
    
    # Test with sample images
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sample_images')
    image_paths = []
    
    # Check if directory exists
    if os.path.exists(sample_dir):
        # Get all image files
        for file in os.listdir(sample_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(sample_dir, file))
    
    if not image_paths:
        print("No sample images found. Please add images to the sample_images directory.")
        sample_image = "Please add sample images to run the benchmark"
    else:
        print(f"Found {len(image_paths)} sample images for benchmark.")
        # Limit to 5 images for faster testing
        image_paths = image_paths[:5]
        
        # Run vision transformer benchmarks
        benchmark.benchmark_vision_transformers(image_paths)
    
    # Skip text transformer benchmarks due to compatibility issues
    print("Skipping text transformer benchmarks due to compatibility issues")
    
    # Generate performance report with just vision results
    benchmark.generate_performance_report()
    
    print("Transformer benchmarks completed. Results saved to", output_dir)

if __name__ == "__main__":
    main() 