import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import gc
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

def simplify_temporal_importance(importance, threshold_percent=10):
    """Simplify temporal importance map by keeping only the top % most important timesteps
    
    Args:
        importance: Numpy array of importance values per timestep
        threshold_percent: Percentage threshold (1-100)
        
    Returns:
        simplified: Thresholded importance map
    """
    threshold = np.percentile(importance, 100 - threshold_percent)
    simplified = np.zeros_like(importance)
    simplified[importance > threshold] = importance[importance > threshold]
    return simplified

def analyze_rnn_decision_boundary(model, sequence_tensor, num_samples=10, perturbation_std=0.05):
    """Analyze decision boundary stability by adding small perturbations
    
    Args:
        model: RNN model
        sequence_tensor: Input sequence
        num_samples: Number of perturbed samples to generate
        perturbation_std: Standard deviation of perturbation
        
    Returns:
        dict: Decision boundary analysis results
    """
    device = next(model.parameters()).device
    original_seq = sequence_tensor.clone()
    
    # Get original prediction
    with torch.no_grad():
        original_output = model(original_seq)
        if isinstance(original_output, dict) and "logits" in original_output:
            original_logits = original_output["logits"].cpu().numpy()
        else:
            original_logits = original_output.cpu().numpy()
        
        original_probs = softmax(original_logits, axis=-1)
        original_pred = np.argmax(original_probs, axis=-1)[0]
        original_conf = original_probs[0, original_pred]
    
    # Generate perturbed samples
    perturbed_preds = []
    perturbed_confs = []
    
    for i in range(num_samples):
        # Add random noise to sequence
        noise = torch.randn_like(original_seq) * perturbation_std
        perturbed_seq = original_seq + noise
        
        # Get prediction
        with torch.no_grad():
            perturbed_output = model(perturbed_seq)
            if isinstance(perturbed_output, dict) and "logits" in perturbed_output:
                perturbed_logits = perturbed_output["logits"].cpu().numpy()
            else:
                perturbed_logits = perturbed_output.cpu().numpy()
            
            perturbed_probs = softmax(perturbed_logits, axis=-1)
            perturbed_pred = np.argmax(perturbed_probs, axis=-1)[0]
            perturbed_conf = perturbed_probs[0, perturbed_pred]
            
            perturbed_preds.append(perturbed_pred)
            perturbed_confs.append(perturbed_conf)
    
    # Calculate stability metrics
    stability_rate = np.mean(np.array(perturbed_preds) == original_pred)
    avg_conf = np.mean(perturbed_confs)
    conf_std = np.std(perturbed_confs)
    
    return {
        "original_prediction": original_pred,
        "original_confidence": original_conf,
        "stability_rate": stability_rate,
        "average_perturbed_confidence": avg_conf,
        "confidence_std": conf_std,
        "perturbed_predictions": perturbed_preds
    }

def fairness_assessment(model, dataset_groups, group_labels, target_labels, threshold_percent=10):
    """Assess explanation fairness across different demographic groups
    
    Args:
        model: RNN model
        dataset_groups: Dictionary of sequence tensors for different groups
        group_labels: Labels for the groups (e.g., 'male', 'female')
        target_labels: Target class labels
        threshold_percent: Threshold for importance map simplification
        
    Returns:
        dict: Fairness metrics across groups
    """
    explainer = RNNExplainer(model)
    group_metrics = {}
    group_explanations = {}
    group_confusion = {}
    
    for group_name, group_data in dataset_groups.items():
        # Initialize metrics for this group
        group_metrics[group_name] = {
            "avg_importance_std": [],
            "avg_num_important_timesteps": [],
            "avg_processing_time": [],
            "accuracy": [],
            "precision": [],
            "recall": []
        }
        group_explanations[group_name] = []
        
        # Get predictions and explanations for each item in group
        all_preds = []
        all_targets = []
        
        for i, (seq, target) in enumerate(zip(group_data, target_labels.get(group_name, []))):
            # Get explanation
            explanation = explainer.explain(seq.unsqueeze(0), target_class=target, threshold_percent=threshold_percent)
            
            # Calculate metrics
            importance = explanation["importance"]
            simplified = explanation["simplified_importance"]
            
            # Number of important timesteps
            num_important = np.sum(simplified > 0)
            
            # Standard deviation of importance (measure of concentration)
            importance_std = np.std(importance)
            
            # Store metrics
            group_metrics[group_name]["avg_importance_std"].append(importance_std)
            group_metrics[group_name]["avg_num_important_timesteps"].append(num_important)
            group_metrics[group_name]["avg_processing_time"].append(explanation["processing_time"])
            
            # Store explanation
            group_explanations[group_name].append(explanation)
            
            # Store prediction for accuracy calculation
            if "target_class" in explanation:
                pred = explanation["target_class"]
                all_preds.append(pred)
                all_targets.append(target)
        
        # Compute confusion matrix if we have predictions
        if all_preds:
            cm = confusion_matrix(all_targets, all_preds)
            group_confusion[group_name] = cm
            
            # Calculate accuracy from confusion matrix
            accuracy = np.trace(cm) / np.sum(cm)
            group_metrics[group_name]["accuracy"] = accuracy
            
            # Calculate precision and recall for each class
            n_classes = cm.shape[0]
            precision = np.zeros(n_classes)
            recall = np.zeros(n_classes)
            
            for c in range(n_classes):
                # Precision = TP / (TP + FP)
                if np.sum(cm[:, c]) > 0:
                    precision[c] = cm[c, c] / np.sum(cm[:, c])
                    
                # Recall = TP / (TP + FN)
                if np.sum(cm[c, :]) > 0:
                    recall[c] = cm[c, c] / np.sum(cm[c, :])
            
            group_metrics[group_name]["precision"] = precision.mean()
            group_metrics[group_name]["recall"] = recall.mean()
    
    # Compute fairness metrics across groups
    fairness_metrics = {
        "importance_std_disparity": {},
        "important_timesteps_disparity": {},
        "processing_time_disparity": {},
        "accuracy_disparity": {},
        "precision_disparity": {},
        "recall_disparity": {},
    }
    
    # Compute average metrics for each group
    for group_name in group_metrics:
        for metric in ["avg_importance_std", "avg_num_important_timesteps", "avg_processing_time"]:
            group_metrics[group_name][metric] = np.mean(group_metrics[group_name][metric])
    
    # Calculate disparities between groups
    # We use group_labels[0] as the reference group
    reference_group = group_labels[0]
    
    for group_name in group_labels[1:]:
        if group_name in group_metrics:
            # For each metric, calculate disparity compared to reference group
            fairness_metrics["importance_std_disparity"][group_name] = group_metrics[group_name]["avg_importance_std"] / (group_metrics[reference_group]["avg_importance_std"] + 1e-10)
            fairness_metrics["important_timesteps_disparity"][group_name] = group_metrics[group_name]["avg_num_important_timesteps"] / (group_metrics[reference_group]["avg_num_important_timesteps"] + 1e-10)
            fairness_metrics["processing_time_disparity"][group_name] = group_metrics[group_name]["avg_processing_time"] / (group_metrics[reference_group]["avg_processing_time"] + 1e-10)
            
            # Performance disparities
            if "accuracy" in group_metrics[group_name] and "accuracy" in group_metrics[reference_group]:
                fairness_metrics["accuracy_disparity"][group_name] = group_metrics[group_name]["accuracy"] / (group_metrics[reference_group]["accuracy"] + 1e-10)
            
            if "precision" in group_metrics[group_name] and "precision" in group_metrics[reference_group]:
                fairness_metrics["precision_disparity"][group_name] = group_metrics[group_name]["precision"] / (group_metrics[reference_group]["precision"] + 1e-10)
            
            if "recall" in group_metrics[group_name] and "recall" in group_metrics[reference_group]:
                fairness_metrics["recall_disparity"][group_name] = group_metrics[group_name]["recall"] / (group_metrics[reference_group]["recall"] + 1e-10)
    
    return {
        "group_metrics": group_metrics,
        "fairness_metrics": fairness_metrics,
        "group_explanations": group_explanations,
        "group_confusion": group_confusion
    }

class RNNExplainer:
    """Class for explaining RNN/LSTM/GRU models"""
    
    def __init__(self, model):
        """Initialize the explainer
        
        Args:
            model: RNN-based model
        """
        self.model = model
        self.model.eval()
        self.hidden_states = []
        self.gradients = []
        self.hook_handles = []
        self.last_memory_usage = 0
        self.last_processing_time = 0
        
    def _register_hooks(self):
        """Register hooks to capture hidden states and gradients"""
        self.hidden_states = []
        self.gradients = []
        
        # Hook function to capture hidden states
        def capture_hidden_states(module, input, output):
            if isinstance(output, tuple):  # LSTM returns (hidden_state, cell_state)
                self.hidden_states.append(output[0].detach())
            else:
                self.hidden_states.append(output.detach())
        
        # Hook function to capture gradients
        def capture_gradients(module, grad_input, grad_output):
            if isinstance(grad_input, tuple) and len(grad_input) > 0:
                if grad_input[0] is not None:
                    self.gradients.append(grad_input[0].detach())
                else:
                    # Handle None gradients
                    self.gradients.append(torch.zeros_like(self.hidden_states[-1]))
            else:
                if grad_input is not None:
                    self.gradients.append(grad_input.detach())
                else:
                    # Handle None gradients
                    self.gradients.append(torch.zeros_like(self.hidden_states[-1]))
        
        # Register hooks for RNN layers
        for name, module in self.model.named_modules():
            if any(x in name.lower() for x in ["lstm", "gru", "rnn"]):
                h1 = module.register_forward_hook(capture_hidden_states)
                h2 = module.register_full_backward_hook(capture_gradients)
                self.hook_handles.extend([h1, h2])
                
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        
    def explain(self, sequence_tensor, target_class=None, threshold_percent=10, analyze_stability=False):
        """Generate explanation for RNN model showing temporal importance
        
        Args:
            sequence_tensor: Input sequence tensor [batch_size, seq_len, features]
            target_class: Target class (None for predicted class)
            threshold_percent: Threshold for simplification
            analyze_stability: Whether to analyze decision boundary stability
            
        Returns:
            dict: Explanation results
        """
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
            
        start_time = time.time()
        
        # Register hooks
        self._register_hooks()
        
        # Forward pass
        self.model.zero_grad()
        outputs = self.model(sequence_tensor)
        
        # Get predicted class if not specified
        if target_class is None:
            if isinstance(outputs, dict) and "logits" in outputs:
                predicted = outputs["logits"].argmax(-1).item()
            else:
                predicted = outputs.argmax(-1).item()
            target_class = predicted
            
        # Create one-hot for target class
        if isinstance(outputs, dict) and "logits" in outputs:
            output_tensor = outputs["logits"]
        else:
            output_tensor = outputs
            
        # Ensure target_class is an integer
        if target_class is not None:
            target_class = int(target_class)
            
        one_hot = torch.zeros_like(output_tensor)
        one_hot[0, target_class] = 1
        
        # Backward pass
        output_tensor.backward(gradient=one_hot, retain_graph=True)
        
        try:
            # Get hidden states and gradients
            final_hidden = self.hidden_states[-1].cpu().numpy()
            final_grad = self.gradients[-1].cpu().numpy()
            
            # Calculate importance per time step
            # Element-wise product of hidden state and its gradient
            importance = np.abs(final_hidden * final_grad)
            
            # Sum across feature dimension
            importance = np.sum(importance, axis=-1).squeeze()
            
            # If 3D tensor (batch, seq_len, hidden_dim), take first batch
            if len(importance.shape) > 1:
                importance = importance[0]
                
            # Normalize
            if importance.max() > 0:
                importance = importance / importance.max()
                
            # Apply threshold
            simplified_importance = simplify_temporal_importance(importance, threshold_percent)
                
        except Exception as e:
            print(f"Error generating RNN explanation: {e}")
            # Create dummy importance array with length of sequence
            importance = np.zeros(sequence_tensor.shape[1])
            simplified_importance = importance.copy()
            
        # Remove hooks
        self._remove_hooks()
        
        # Memory tracking end
        if torch.cuda.is_available():
            end_mem = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()
            self.last_memory_usage = peak_mem - start_mem
        else:
            self.last_memory_usage = 0
            
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        self.last_processing_time = processing_time
        
        explanation_result = {
            "importance": importance,
            "simplified_importance": simplified_importance,
            "target_class": target_class,
            "processing_time": processing_time,
            "memory_usage": self.last_memory_usage,
            "threshold_percent": threshold_percent
        }
        
        # Capture cell states for LSTM
        if any(isinstance(module, torch.nn.LSTM) for _, module in self.model.named_modules()):
            try:
                # For LSTM, also look at cell state, which is the second element in the tuple
                lstm_states = [output[1].detach().cpu().numpy() for output in self.hidden_states 
                              if isinstance(output, tuple) and len(output) > 1]
                
                if lstm_states:
                    # Use the last LSTM layer's cell state
                    cell_states = lstm_states[-1]
                    
                    # Calculate cell state activity (mean absolute value)
                    cell_activity = np.abs(cell_states).mean(axis=-1).squeeze()
                    
                    # Add to result
                    explanation_result["cell_activity"] = cell_activity
            except Exception as e:
                print(f"Error processing LSTM cell states: {e}")
        
        # Add stability analysis if requested
        if analyze_stability:
            stability_analysis = analyze_rnn_decision_boundary(self.model, sequence_tensor)
            explanation_result["stability_analysis"] = stability_analysis
        
        return explanation_result
        
    def visualize_explanation(self, sequence_data, explanation_result, feature_names=None, title=None, save_path=None, show_cell_states=True):
        """Visualize the temporal importance
        
        Args:
            sequence_data: Original input sequence
            explanation_result: Result from explain() method
            feature_names: Optional list of feature names
            title: Optional title for the plot
            save_path: Optional path to save the visualization
            show_cell_states: Whether to show LSTM cell state activity
            
        Returns:
            PIL Image of the visualization
        """
        # Import libraries
        import matplotlib.pyplot as plt
        
        # Get importance maps
        importance = explanation_result["importance"]
        simplified_importance = explanation_result["simplified_importance"]
        
        # Create temporal indices
        timesteps = np.arange(len(importance))
        
        # Determine subplot configuration based on available data
        num_plots = 2  # Base plots: full importance + simplified importance
        if "cell_activity" in explanation_result and show_cell_states:
            num_plots += 1
        if "stability_analysis" in explanation_result:
            num_plots += 1
            
        # Create figure and subplots
        plt.figure(figsize=(12, 3 * num_plots))
        
        # Full importance plot
        plt.subplot(num_plots, 1, 1)
        plt.bar(timesteps, importance, color='blue', alpha=0.6)
        plt.title("Full Temporal Importance")
        plt.ylabel("Importance")
        
        # Find top k important timesteps
        top_indices = np.argsort(importance)[::-1][:5]  # Top 5 timesteps
        for idx in top_indices:
            plt.text(idx, importance[idx], f"t={idx}", ha='center', va='bottom')
        
        # Simplified importance plot
        plt.subplot(num_plots, 1, 2)
        plt.bar(timesteps, simplified_importance, color='red', alpha=0.6)
        plt.title(f"Simplified Importance ({explanation_result.get('threshold_percent', 10)}%)")
        plt.ylabel("Importance")
        
        # Show labels for important timesteps
        important_steps = np.where(simplified_importance > 0)[0]
        for idx in important_steps:
            plt.text(idx, simplified_importance[idx], f"t={idx}", ha='center', va='bottom')
        
        # Add cell states if available
        current_plot = 3
        if "cell_activity" in explanation_result and show_cell_states:
            cell_activity = explanation_result["cell_activity"]
            
            plt.subplot(num_plots, 1, current_plot)
            plt.bar(np.arange(len(cell_activity)), cell_activity, color='green', alpha=0.6)
            plt.title("LSTM Cell State Activity")
            plt.ylabel("Activity")
            current_plot += 1
        
        # Add stability analysis if available
        if "stability_analysis" in explanation_result:
            stability = explanation_result["stability_analysis"]
            
            plt.subplot(num_plots, 1, current_plot)
            perturbed_preds = stability["perturbed_predictions"]
            unique_preds, counts = np.unique(perturbed_preds, return_counts=True)
            
            plt.bar(unique_preds, counts, color='purple', alpha=0.6)
            plt.axvline(x=stability["original_prediction"], color='red', linestyle='--', 
                      label=f"Original Prediction (Class {stability['original_prediction']})")
            plt.title(f"Decision Stability (Rate: {stability['stability_rate']:.2f})")
            plt.xlabel("Predicted Class")
            plt.ylabel("Count")
            plt.legend()
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        # Create a separate figure for return value
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(timesteps, importance, color='blue', alpha=0.3, label='Full Importance')
        ax.bar(timesteps, simplified_importance, color='red', alpha=0.6, label='Simplified Importance')
        ax.set_title(title if title else "RNN Temporal Explanation")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Importance")
        
        # Add stability info if available
        if "stability_analysis" in explanation_result:
            stability = explanation_result["stability_analysis"]
            stability_text = f"Stability: {stability['stability_rate']:.2f}, Conf: {stability['original_confidence']:.2f}"
            ax.text(0.5, 0.01, stability_text, transform=ax.transAxes, ha='center')
            
        ax.legend()
        fig.tight_layout()
        
        # Convert matplotlib figure to PIL image
        fig.canvas.draw()
        plot_img = Image.frombytes('RGB', 
                                 fig.canvas.get_width_height(),
                                 fig.canvas.tostring_rgb())
        plt.close(fig)
        plt.close()
        
        return plot_img
        
    def assess_fairness(self, dataset_groups, group_labels, target_labels=None, threshold_percent=10):
        """Wrapper method to assess explanation fairness across demographic groups
        
        Args:
            dataset_groups: Dictionary of sequence tensors for different groups
            group_labels: Labels for the groups (e.g., 'male', 'female')
            target_labels: Optional target class labels for each group
            threshold_percent: Threshold for importance map simplification
            
        Returns:
            dict: Fairness assessment results
        """
        # Default empty target labels if not provided
        if target_labels is None:
            target_labels = {group: [] for group in dataset_groups.keys()}
            
        # Perform fairness assessment
        return fairness_assessment(self.model, dataset_groups, group_labels, target_labels, threshold_percent)
    
    def visualize_fairness_assessment(self, fairness_results, title=None, save_path=None):
        """Visualize fairness assessment results
        
        Args:
            fairness_results: Results from assess_fairness method
            title: Optional title for the plot
            save_path: Optional path to save the visualization
            
        Returns:
            PIL Image of the visualization
        """
        import matplotlib.pyplot as plt
        
        # Extract metrics
        group_metrics = fairness_results["group_metrics"]
        fairness_metrics = fairness_results["fairness_metrics"]
        
        # Set up the figure
        plt.figure(figsize=(15, 10))
        
        # Group metrics subplot
        plt.subplot(2, 1, 1)
        x = np.arange(len(group_metrics))
        width = 0.2
        
        # Extract metric values for each group
        groups = list(group_metrics.keys())
        importance_std = [group_metrics[g]["avg_importance_std"] for g in groups]
        important_timesteps = [group_metrics[g]["avg_num_important_timesteps"] for g in groups]
        processing_time = [group_metrics[g]["avg_processing_time"] for g in groups]
        
        # Normalize values for better visualization
        importance_std = np.array(importance_std) / max(importance_std) if max(importance_std) > 0 else importance_std
        important_timesteps = np.array(important_timesteps) / max(important_timesteps) if max(important_timesteps) > 0 else important_timesteps
        processing_time = np.array(processing_time) / max(processing_time) if max(processing_time) > 0 else processing_time
        
        # Plot bars
        plt.bar(x - width, importance_std, width, label='Importance Std (Normalized)')
        plt.bar(x, important_timesteps, width, label='Important Timesteps (Normalized)')
        plt.bar(x + width, processing_time, width, label='Processing Time (Normalized)')
        
        plt.xlabel('Group')
        plt.ylabel('Normalized Value')
        plt.title('Explanation Metrics by Group')
        plt.xticks(x, groups)
        plt.legend()
        
        # Fairness disparity subplot
        plt.subplot(2, 1, 2)
        
        # Get reference group
        reference_group = groups[0]
        comparison_groups = groups[1:]
        
        # Create x positions for bars
        x = np.arange(len(comparison_groups))
        width = 0.15
        
        # Extract disparities
        metrics_to_plot = [
            "importance_std_disparity", 
            "important_timesteps_disparity", 
            "processing_time_disparity",
            "accuracy_disparity",
            "precision_disparity",
            "recall_disparity"
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in fairness_metrics and fairness_metrics[metric]:
                values = [fairness_metrics[metric].get(g, 1.0) for g in comparison_groups]
                plt.bar(x + (i - len(metrics_to_plot)/2 + 0.5) * width, values, width, label=metric, color=colors[i])
        
        # Add reference line at 1.0 (perfect parity)
        plt.axhline(y=1.0, color='r', linestyle='--', label='Parity')
        
        plt.xlabel('Group (compared to reference)')
        plt.ylabel('Disparity Ratio')
        plt.title(f'Fairness Disparities (Relative to {reference_group})')
        plt.xticks(x, comparison_groups)
        plt.legend()
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        # Convert to PIL image
        fig = plt.gcf()
        fig.canvas.draw()
        plot_img = Image.frombytes('RGB', 
                                 fig.canvas.get_width_height(),
                                 fig.canvas.tostring_rgb())
        plt.close(fig)
        
        return plot_img 