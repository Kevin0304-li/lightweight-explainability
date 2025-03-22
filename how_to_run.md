# How to Run the Lightweight Explainability Implementation

This guide provides instructions on how to run the various components of the Lightweight Explainability implementation.

## Prerequisites

Ensure you have all dependencies installed:

```
pip install -r requirements.txt
```

## Basic Usage

### Generate an Explanation for a Single Image

To generate a lightweight explanation for a single image, use:

```python
from lightweight_explainability import ExplainableModel, preprocess_image
from PIL import Image

# Initialize model
model = ExplainableModel()

# Load and preprocess image
img = Image.open('path/to/your/image.jpg').convert('RGB')
img_tensor = preprocess_image(img)

# Generate prediction
prediction, confidence = model.predict(img_tensor)
print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")

# Generate standard Grad-CAM
standard_cam = model.generate_gradcam(img_tensor)

# Generate simplified Grad-CAM with 10% threshold
simplified_cam = model.simplified_gradcam(img_tensor, threshold_pct=10)

# Visualize explanations
model.visualize_explanation(img, standard_cam, title="Standard Grad-CAM", save_path="standard.png")
model.visualize_explanation(img, simplified_cam, title="Simplified Grad-CAM (10%)", save_path="simplified.png")
```

## Benchmarking

### CIFAR-10 Benchmark

To benchmark on CIFAR-10 dataset:

```
python benchmark_cifar10.py --num_samples 10 --thresholds 1,5,10,20
```

Parameters:
- `--num_samples`: Number of images to test (default: 10)
- `--thresholds`: Comma-separated list of thresholds (default: 1,5,10,20)
- `--output_dir`: Directory to save results (default: ./results/cifar10)

### Custom Dataset Benchmark

To benchmark on a custom dataset:

```
python benchmark_custom_dataset.py --data_dir ./examples --num_samples 5
```

Parameters:
- `--data_dir`: Directory with custom dataset (default: ./examples)
- `--num_samples`: Number of samples per class (default: 5)
- `--thresholds`: Comma-separated list of thresholds (default: 1,5,10,20)

### Mobile Device Benchmark

To benchmark on simulated mobile devices:

```
python mobile_benchmark.py --device raspberry_pi --num_samples 5
```

Parameters:
- `--device`: Device to simulate (options: raspberry_pi, android, iphone)
- `--num_samples`: Number of samples to test (default: 5)
- `--thresholds`: Comma-separated list of thresholds (default: 1,5,10,20)

## User Study

To run a simulated user study:

```
python user_study.py --num_participants 20 --num_images 3
```

Parameters:
- `--num_participants`: Number of simulated participants (default: 20)
- `--num_images`: Number of images per class (default: 3)
- `--thresholds`: Comma-separated list of thresholds (default: 5,10,20)

## Creating a Custom Dataset

To create a custom 4-class dataset:

```
python custom_dataset.py --num_images 100 --output_dir ./examples
```

Parameters:
- `--num_images`: Total number of images (default: 100)
- `--output_dir`: Output directory (default: ./examples)

## Pointing Game Evaluation

To run pointing game evaluation:

```
python pointing_game.py --num_samples 50
```

Parameters:
- `--num_samples`: Number of samples to evaluate (default: 50)
- `--dataset`: Dataset to use (options: coco, imagenet, custom)

## Web Application

To run the web application for interactive exploration:

```
python app.py
```

Then open your browser to `http://localhost:5000`

## Viewing Results

After running benchmarks, you can find results in:
- CIFAR-10 results: `./results/cifar10/`
- Custom dataset results: `./results/custom_benchmark/`
- Mobile benchmark results: `./results/mobile/`
- User study results: `./results/user_study/`

Each results directory contains:
- Benchmark report (Markdown)
- Visualizations directory with plots
- Individual explanation images

## Additional Features

### Audit Mode

To enable audit mode:

```python
from lightweight_explainability import ExplainableModel

model = ExplainableModel()
model.set_audit_mode(True, log_dir="./audit_logs")

# Generate explanations as normal
# All explanations will be logged to the specified directory
```

### Dynamic Thresholding

For automatic threshold selection:

```python
from lightweight_explainability import ExplainableModel, preprocess_image
from PIL import Image

model = ExplainableModel()
img = Image.open('path/to/your/image.jpg').convert('RGB')
img_tensor = preprocess_image(img)

# Generate Grad-CAM with dynamic threshold
cam = model.generate_gradcam(img_tensor)
best_threshold = model.dynamic_threshold(cam, min_area_pct=5)
simplified_cam = model.simplified_gradcam(img_tensor, threshold_pct=best_threshold)

print(f"Automatically selected threshold: {best_threshold}%")
```

## Troubleshooting

- If you encounter memory errors, reduce `--num_samples` or batch size
- For CUDA out of memory, set `CUDA_VISIBLE_DEVICES=""` to use CPU only
- If images are not loading, check the path and file formats (supported: jpg, png, jpeg) 