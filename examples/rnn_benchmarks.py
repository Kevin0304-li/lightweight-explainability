#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive benchmark for RNN/LSTM explainability across different domains.

This script evaluates the performance and quality of RNN explainability
across different architectures and application domains:
1. Sequence Classification (LSTM, GRU, Bidirectional LSTM)
2. Time Series Forecasting
3. Text Classification
4. Multivariate Time Series Analysis

Metrics:
- Processing time
- Memory usage
- Explanation quality via temporal importance
- Stability across different sequence lengths
"""

import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tracemalloc
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our RNN explainability module
from rnn_explainability import RNNExplainer, simplify_temporal_importance

# Create output directory
output_dir = "results/rnn_benchmarks"
os.makedirs(output_dir, exist_ok=True)

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

# Define RNN models for benchmarking
class LSTMClassifier(nn.Module):
    """LSTM model for sequence classification"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRUClassifier(nn.Module):
    """GRU model for sequence classification"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM model for sequence classification"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # * 2 for bidirectional
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # * 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # * 2 for bidirectional
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesLSTM(nn.Module):
    """LSTM model for time series forecasting"""
    
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class RNNBenchmark:
    """Benchmark for RNN explainability across architectures and domains."""
    
    def __init__(self):
        """Initialize benchmark with different RNN models."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.explainers = {}
        
        # Initialize results tracking
        self.results = {
            'execution_time': defaultdict(list),
            'memory_usage': defaultdict(list),
            'sequence_length_impact': defaultdict(dict),
            'feature_importance': defaultdict(list),
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different RNN models."""
        print("Initializing RNN models...")
        
        # Common parameters
        input_dim = 10
        hidden_dim = 20
        num_layers = 2
        num_classes = 2
        
        # LSTM Classifier
        lstm_model = LSTMClassifier(input_dim, hidden_dim, num_layers, num_classes).to(self.device)
        self.models['lstm'] = lstm_model
        self.explainers['lstm'] = RNNExplainer(lstm_model)
        print("  - LSTM Classifier initialized")
        
        # GRU Classifier
        gru_model = GRUClassifier(input_dim, hidden_dim, num_layers, num_classes).to(self.device)
        self.models['gru'] = gru_model
        self.explainers['gru'] = RNNExplainer(gru_model)
        print("  - GRU Classifier initialized")
        
        # Bidirectional LSTM
        bilstm_model = BiLSTMClassifier(input_dim, hidden_dim, num_layers, num_classes).to(self.device)
        self.models['bilstm'] = bilstm_model
        self.explainers['bilstm'] = RNNExplainer(bilstm_model)
        print("  - Bidirectional LSTM initialized")
        
        # Time Series LSTM
        ts_lstm_model = TimeSeriesLSTM(input_dim, hidden_dim, num_layers, output_dim=1).to(self.device)
        self.models['ts_lstm'] = ts_lstm_model
        self.explainers['ts_lstm'] = RNNExplainer(ts_lstm_model)
        print("  - Time Series LSTM initialized")
    
    def generate_sequence_data(self, seq_length=20, num_samples=10):
        """Generate synthetic sequence data for classification."""
        # Create synthetic data with pattern
        x = torch.randn(num_samples, seq_length, 10)
        y = torch.zeros(num_samples, dtype=torch.long)
        
        # Add pattern: if sum of feature 0 in first half > second half, label = 1
        for i in range(num_samples):
            first_half = x[i, :seq_length//2, 0].sum()
            second_half = x[i, seq_length//2:, 0].sum()
            if first_half > second_half:
                y[i] = 1
                # Strengthen the pattern for class 1
                x[i, 5:10, 0] += 1.0
            else:
                # Strengthen the pattern for class 0
                x[i, seq_length-10:seq_length-5, 0] += 1.0
        
        return x.to(self.device), y.to(self.device)
    
    def generate_time_series_data(self, seq_length=50, num_samples=10):
        """Generate synthetic time series data for forecasting."""
        # Time steps
        t = torch.linspace(0, 4*np.pi, seq_length).to(self.device)
        
        # Create batch of sequences
        x = torch.zeros(num_samples, seq_length, 10).to(self.device)
        y = torch.zeros(num_samples, 1).to(self.device)
        
        for i in range(num_samples):
            # Main signal: sine wave with frequency variations
            freq = 1.0 + 0.2 * torch.randn(1).item()
            phase = 0.5 * torch.randn(1).item()
            
            # Create primary signal with random frequency and phase
            signal = torch.sin(freq * t + phase)
            
            # Add noise
            noise = 0.1 * torch.randn_like(signal)
            signal = signal + noise
            
            # Set target as the next value in sequence
            target = torch.sin(freq * (t[-1] + 0.1) + phase).unsqueeze(0)
            
            # Put signal in first feature, add random values for other features
            x[i, :, 0] = signal
            x[i, :, 1:] = 0.1 * torch.randn(seq_length, 9).to(self.device)
            
            y[i, 0] = target
        
        return x, y
    
    def explain_sequence(self, model_name, sequence, target, threshold_percent=10):
        """Generate explanation for sequence."""
        if model_name not in self.models or model_name not in self.explainers:
            print(f"Model {model_name} not available")
            return None
        
        # Generate explanation
        explanation = self.explainers[model_name].explain(
            sequence.unsqueeze(0), 
            target_class=target.item(),
            threshold_percent=threshold_percent
        )
        
        return explanation
    
    def benchmark_sequence_classification(self, num_samples=10):
        """Benchmark RNN models on sequence classification task."""
        # Models to evaluate
        classification_models = ['lstm', 'gru', 'bilstm']
        
        print(f"Benchmarking {len(classification_models)} RNN models on sequence classification...")
        
        # Generate data
        x, y = self.generate_sequence_data(num_samples=num_samples)
        
        for model_name in classification_models:
            print(f"\nBenchmarking {model_name.upper()}...")
            
            for i in tqdm(range(num_samples), desc=f"Processing sequences"):
                # Get sequence and target
                sequence = x[i]
                target = y[i]
                
                # Measure execution time and memory
                explanation_func = lambda: self.explain_sequence(model_name, sequence, target)
                
                explanation, memory_usage = measure_memory_usage(explanation_func)
                if explanation is None:
                    continue
                    
                execution_time = explanation.get('processing_time', 0)
                
                # Record results
                self.results['execution_time'][model_name].append(execution_time)
                self.results['memory_usage'][model_name].append(memory_usage)
                
                # Record feature importance (for multidimensional input)
                if 'feature_importance' in explanation:
                    self.results['feature_importance'][model_name].append(
                        explanation['feature_importance']
                    )
    
    def benchmark_time_series_forecasting(self, num_samples=10):
        """Benchmark time series forecasting models."""
        # Only use time series model
        print(f"\nBenchmarking time series LSTM model...")
        
        # Generate data
        x, y = self.generate_time_series_data(num_samples=num_samples)
        
        model_name = 'ts_lstm'
        
        for i in tqdm(range(num_samples), desc=f"Processing time series"):
            # Get sequence and target
            sequence = x[i]
            target = y[i]
            
            # Measure execution time and memory
            explanation_func = lambda: self.explain_sequence(model_name, sequence, target)
            
            explanation, memory_usage = measure_memory_usage(explanation_func)
            if explanation is None:
                continue
                
            execution_time = explanation.get('processing_time', 0)
            
            # Record results
            self.results['execution_time'][model_name].append(execution_time)
            self.results['memory_usage'][model_name].append(memory_usage)
    
    def benchmark_sequence_length_impact(self, model_name='lstm'):
        """Benchmark the impact of sequence length on explanation performance."""
        print(f"\nBenchmarking impact of sequence length on {model_name.upper()}...")
        
        # Sequence lengths to test
        sequence_lengths = [10, 20, 50, 100, 200, 500]
        
        for seq_len in sequence_lengths:
            print(f"  Testing sequence length: {seq_len}")
            
            # Generate data of this length
            x, y = self.generate_sequence_data(seq_length=seq_len, num_samples=5)
            
            execution_times = []
            memory_usages = []
            
            for i in range(len(x)):
                # Get sequence and target
                sequence = x[i]
                target = y[i]
                
                # Measure execution time and memory
                explanation_func = lambda: self.explain_sequence(model_name, sequence, target)
                
                explanation, memory_usage = measure_memory_usage(explanation_func)
                if explanation is None:
                    continue
                    
                execution_time = explanation.get('processing_time', 0)
                
                execution_times.append(execution_time)
                memory_usages.append(memory_usage)
            
            # Record average results for this sequence length
            if execution_times:
                self.results['sequence_length_impact'][model_name][seq_len] = {
                    'avg_time': np.mean(execution_times),
                    'avg_memory': np.mean(memory_usages)
                }
    
    def _visualize_sequence_explanation(self, model_name, sequence_length=50):
        """Generate visualization of sequence explanation."""
        # Generate a sample sequence
        x, y = self.generate_sequence_data(seq_length=sequence_length, num_samples=1)
        sequence = x[0]
        target = y[0]
        
        # Generate explanation
        explanation = self.explain_sequence(model_name, sequence, target)
        
        if explanation is None:
            return
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot original sequence values (first feature only)
        plt.subplot(2, 1, 1)
        plt.plot(sequence.cpu().numpy()[:, 0], label='Feature 0')
        plt.title(f"Input Sequence for {model_name.upper()} (Class {target.item()})")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot importance
        plt.subplot(2, 1, 2)
        
        # Get importance
        if 'importance' in explanation:
            importance = explanation['importance']
            simplified = explanation['simplified_importance']
            
            # Plot both raw and simplified importance
            plt.plot(importance, label='Raw Importance', color='blue', alpha=0.5)
            plt.bar(range(len(simplified)), simplified, color='red', alpha=0.7, label='Simplified')
            
            plt.title(f"Timestep Importance (Processing Time: {explanation['processing_time']:.3f}s)")
            plt.xlabel("Timestep")
            plt.ylabel("Importance")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_sequence_explanation.png'), dpi=150)
        plt.close()
        
        print(f"Sequence explanation visualization saved to {output_dir}/{model_name}_sequence_explanation.png")
    
    def generate_performance_report(self):
        """Generate performance report with tables and charts."""
        # Create DataFrame for performance data
        performance_data = []
        
        for model_name in self.models.keys():
            # Calculate average metrics
            avg_time = np.mean(self.results['execution_time'].get(model_name, [0]))
            avg_memory = np.mean(self.results['memory_usage'].get(model_name, [0]))
            
            # Determine model type
            if model_name == 'lstm':
                model_type = 'LSTM Classifier'
            elif model_name == 'gru':
                model_type = 'GRU Classifier'
            elif model_name == 'bilstm':
                model_type = 'Bidirectional LSTM'
            elif model_name == 'ts_lstm':
                model_type = 'Time Series LSTM'
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
        df.to_csv(os.path.join(output_dir, 'rnn_performance.csv'), index=False)
        
        # Create performance visualization
        plt.figure(figsize=(12, 8))
        
        # Time comparison
        plt.subplot(2, 1, 1)
        
        # Use seaborn for better styling
        sns.barplot(x='Model', y='Avg Time (s)', data=df)
        plt.title('Average Processing Time by RNN Model Type')
        plt.grid(True, alpha=0.3)
        
        # Memory comparison
        plt.subplot(2, 1, 2)
        sns.barplot(x='Model', y='Avg Memory (MB)', data=df)
        plt.title('Average Memory Usage by RNN Model Type')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rnn_performance_comparison.png'), dpi=150)
        plt.close()
        
        # Create sequence length impact visualization
        if self.results['sequence_length_impact']:
            self._visualize_sequence_length_impact()
        
        # Save as markdown table
        with open(os.path.join(output_dir, 'rnn_performance.md'), 'w') as f:
            f.write("# RNN Explainability Performance\n\n")
            
            # Print main metrics
            f.write("| Model | Type | Avg Time (s) | Avg Memory (MB) |\n")
            f.write("|-------|------|-------------|----------------|\n")
            
            for _, row in df.iterrows():
                f.write(f"| {row['Model']} | {row['Type']} | {row['Avg Time (s)']:.4f} | {row['Avg Memory (MB)']:.2f} |\n")
        
        print(f"Performance report saved to {output_dir}/rnn_performance.md")
    
    def _visualize_sequence_length_impact(self):
        """Visualize the impact of sequence length on performance."""
        if not self.results['sequence_length_impact']:
            return
        
        # Model to visualize
        model_name = list(self.results['sequence_length_impact'].keys())[0]
        data = self.results['sequence_length_impact'][model_name]
        
        # Extract data
        seq_lengths = sorted(list(data.keys()))
        times = [data[seq_len]['avg_time'] for seq_len in seq_lengths]
        memories = [data[seq_len]['avg_memory'] for seq_len in seq_lengths]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Time plot
        plt.subplot(2, 1, 1)
        plt.plot(seq_lengths, times, 'o-', linewidth=2)
        plt.title(f"Impact of Sequence Length on Processing Time ({model_name.upper()})")
        plt.xlabel("Sequence Length")
        plt.ylabel("Processing Time (s)")
        plt.grid(True, alpha=0.3)
        
        # Memory plot
        plt.subplot(2, 1, 2)
        plt.plot(seq_lengths, memories, 'o-', linewidth=2, color='orange')
        plt.title(f"Impact of Sequence Length on Memory Usage ({model_name.upper()})")
        plt.xlabel("Sequence Length")
        plt.ylabel("Memory Usage (MB)")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sequence_length_impact.png'), dpi=150)
        plt.close()
        
        print(f"Sequence length impact visualization saved to {output_dir}/sequence_length_impact.png")

def main():
    """Run RNN benchmarks."""
    # Initialize benchmark
    benchmark = RNNBenchmark()
    
    # Run sequence classification benchmark
    benchmark.benchmark_sequence_classification(num_samples=10)
    
    # Run time series forecasting benchmark
    benchmark.benchmark_time_series_forecasting(num_samples=10)
    
    # Test impact of sequence length
    benchmark.benchmark_sequence_length_impact()
    
    # Generate visualizations for each model
    for model_name in ['lstm', 'gru', 'bilstm']:  # Remove ts_lstm from visualization
        benchmark._visualize_sequence_explanation(model_name)
    
    # Generate performance report
    benchmark.generate_performance_report()
    
    print("RNN benchmarks completed. Results saved to", output_dir)

if __name__ == "__main__":
    main() 