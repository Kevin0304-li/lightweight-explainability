import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from rnn_explainability import RNNExplainer
from universal_explainability import UniversalExplainer

# Define a simple LSTM model for demonstration
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

def generate_synthetic_data(seq_length=20, batch_size=1, input_size=5):
    """Generate synthetic sequence data"""
    # Create a random sequence
    X = torch.randn(batch_size, seq_length, input_size)
    
    # Create a target based on the last few elements
    # This makes specific timesteps more important than others
    importance = torch.zeros(seq_length)
    importance[seq_length//2:seq_length//2+3] = 1.0  # Make middle elements important
    
    # For binary classification
    y = torch.sum(X[:, importance > 0.5, :]) > 0
    y = y.long()
    
    return X, y, importance

def main():
    """Demonstrate RNN explainability with LSTM model"""
    # Create output directory
    os.makedirs('./results/rnn_examples', exist_ok=True)
    
    # Set up model and data
    input_size = 5
    hidden_size = 20
    num_layers = 2
    num_classes = 2
    seq_length = 20
    
    # Initialize LSTM model
    model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)
    model.eval()
    
    # Generate synthetic data
    X, y, true_importance = generate_synthetic_data(seq_length, 1, input_size)
    print(f"Generated sequence with shape {X.shape}, target class: {y.item()}")
    print(f"True important timesteps: {torch.nonzero(true_importance).squeeze().tolist()}")
    
    # Method 1: Using specialized RNNExplainer
    print("Generating explanation with RNNExplainer...")
    explainer = RNNExplainer(model)
    explanation = explainer.explain(X, target_class=y.item(), threshold_percent=20)
    
    # Visualize explanation
    vis_img = explainer.visualize_explanation(
        X.numpy(), explanation,
        title="LSTM Temporal Importance",
        save_path="./results/rnn_examples/lstm_explanation_direct.png"
    )
    
    # Method 2: Using UniversalExplainer
    print("Generating explanation with UniversalExplainer...")
    universal_explainer = UniversalExplainer(model, model_type="rnn")
    universal_explanation = universal_explainer.explain(X, target_class=y.item(), threshold_percent=20)
    
    # Visualize explanation
    universal_vis_img = universal_explainer.visualize_explanation(
        X.numpy(), universal_explanation,
        title="Universal Explainer - LSTM",
        save_path="./results/rnn_examples/lstm_explanation_universal.png"
    )
    
    # Run benchmark with different thresholds
    print("Running benchmark with different thresholds...")
    benchmark_results = universal_explainer.benchmark(
        X, target_class=y.item(), thresholds=[10, 20, 30], num_runs=3
    )
    
    # Visualize benchmark results
    universal_explainer.visualize_benchmark(
        benchmark_results,
        save_path="./results/rnn_examples/lstm_benchmark.png"
    )
    
    # Print performance summary
    print("Performance Summary:")
    print(universal_explainer.get_performance_summary())
    
    # Compare with ground truth importance
    importance = explanation["importance"]
    top_timesteps = np.argsort(importance)[-3:]  # Top 3 timesteps
    true_top = torch.nonzero(true_importance).squeeze().tolist()
    
    print(f"Top important timesteps (predicted): {top_timesteps}")
    print(f"True important timesteps: {true_top}")
    
    # Calculate accuracy
    common = set(top_timesteps).intersection(set(true_top))
    accuracy = len(common) / len(true_top)
    print(f"Timestep importance accuracy: {accuracy:.2f}")
    
    print("Done! Results saved in ./results/rnn_examples/")

if __name__ == "__main__":
    main() 