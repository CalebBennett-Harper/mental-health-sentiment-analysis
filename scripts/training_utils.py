"""
Utilities for monitoring model training and visualizing results.
Includes functions for metric tracking, confusion matrix visualization, and learning curve plotting.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import confusion_matrix, classification_report
import wandb
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Class for monitoring and visualizing model training progress."""
    
    def __init__(
        self, 
        output_dir: str,
        emotion_labels: List[str],
        use_wandb: bool = False,
        project_name: str = "mental-health-sentiment",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize training monitor.
        
        Args:
            output_dir: Directory to save visualizations and logs
            emotion_labels: List of emotion labels
            use_wandb: Whether to use Weights & Biases for tracking
            project_name: W&B project name
            experiment_name: W&B experiment name (default: timestamp)
        """
        self.output_dir = output_dir
        self.emotion_labels = emotion_labels
        self.use_wandb = use_wandb
        self.metrics_history = {
            "train_loss": [],
            "eval_loss": [],
            "eval_accuracy": [],
            "eval_f1": [],
            "learning_rate": [],
            "epoch": []
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize W&B if requested
        if use_wandb:
            if experiment_name is None:
                experiment_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            wandb.init(
                project=project_name,
                name=experiment_name,
                config={
                    "emotion_labels": emotion_labels
                }
            )
            logger.info(f"Initialized W&B tracking for experiment: {experiment_name}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int, epoch: Optional[int] = None):
        """
        Log metrics for the current training step.
        
        Args:
            metrics: Dictionary of metric name to value
            step: Current training step
            epoch: Current epoch (optional)
        """
        # Store metrics in history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        if epoch is not None:
            self.metrics_history["epoch"].append(epoch)
        
        # Log to W&B if enabled
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {step}" + (f", Epoch {epoch}" if epoch is not None else "") + f": {metrics_str}")
    
    def plot_learning_curve(self, save_path: Optional[str] = None):
        """
        Plot learning curves for loss and metrics.
        
        Args:
            save_path: Path to save the plot (if None, will be saved in output_dir)
        """
        if not self.metrics_history["train_loss"]:
            logger.warning("No metrics history available for plotting learning curve")
            return
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot loss
        ax1.plot(self.metrics_history["train_loss"], label="Training Loss")
        if self.metrics_history["eval_loss"]:
            ax1.plot(self.metrics_history["eval_loss"], label="Validation Loss")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)
        
        # Plot metrics
        if self.metrics_history["eval_accuracy"]:
            ax2.plot(self.metrics_history["eval_accuracy"], label="Accuracy")
        if self.metrics_history["eval_f1"]:
            ax2.plot(self.metrics_history["eval_f1"], label="F1 Score")
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Metric Value")
        ax2.set_title("Evaluation Metrics")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, "learning_curve.png")
        
        plt.savefig(save_path)
        logger.info(f"Learning curve saved to {save_path}")
        
        # Log to W&B if enabled
        if self.use_wandb:
            wandb.log({"learning_curve": wandb.Image(fig)})
        
        plt.close(fig)
    
    def plot_confusion_matrix(
        self, 
        y_true: List[int], 
        y_pred: List[int],
        save_path: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Plot confusion matrix for model predictions.
        
        Args:
            y_true: List of true labels (as indices)
            y_pred: List of predicted labels (as indices)
            save_path: Path to save the plot (if None, will be saved in output_dir)
            normalize: Whether to normalize confusion matrix values
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.emotion_labels,
            yticklabels=self.emotion_labels
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, "confusion_matrix.png")
        
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")
        
        # Log to W&B if enabled
        if self.use_wandb:
            wandb.log({"confusion_matrix": wandb.Image(plt)})
        
        plt.close()
    
    def log_classification_report(
        self, 
        y_true: List[int], 
        y_pred: List[int],
        save_path: Optional[str] = None
    ):
        """
        Generate and log classification report.
        
        Args:
            y_true: List of true labels (as indices)
            y_pred: List of predicted labels (as indices)
            save_path: Path to save the report (if None, will be saved in output_dir)
        """
        # Generate classification report
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.emotion_labels,
            output_dict=True
        )
        
        # Convert to DataFrame for easier handling
        report_df = pd.DataFrame(report).transpose()
        
        # Log to console
        logger.info(f"Classification Report:\n{report_df}")
        
        # Save report
        if save_path is None:
            save_path = os.path.join(self.output_dir, "classification_report.json")
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Classification report saved to {save_path}")
        
        # Log to W&B if enabled
        if self.use_wandb:
            wandb.log({"classification_report": wandb.Table(dataframe=report_df)})
        
        return report
    
    def plot_prediction_examples(
        self, 
        texts: List[str], 
        true_labels: List[int], 
        pred_labels: List[int],
        probabilities: List[List[float]],
        n_examples: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Plot examples of model predictions with text and probabilities.
        
        Args:
            texts: List of input texts
            true_labels: List of true labels (as indices)
            pred_labels: List of predicted labels (as indices)
            probabilities: List of probability distributions across all emotions
            n_examples: Number of examples to plot
            save_path: Path to save the plot (if None, will be saved in output_dir)
        """
        # Select a subset of examples
        n_examples = min(n_examples, len(texts))
        indices = np.random.choice(len(texts), n_examples, replace=False)
        
        fig, axes = plt.subplots(n_examples, 1, figsize=(15, n_examples * 3))
        if n_examples == 1:
            axes = [axes]
        
        for i, idx in enumerate(indices):
            ax = axes[i]
            text = texts[idx]
            true_label = self.emotion_labels[true_labels[idx]]
            pred_label = self.emotion_labels[pred_labels[idx]]
            probs = probabilities[idx]
            
            # Create bar chart of probabilities
            sns.barplot(x=probs, y=self.emotion_labels, ax=ax)
            
            # Add text annotation
            title = f"Text: {text[:100]}..."
            subtitle = f"True: {true_label}, Predicted: {pred_label}"
            ax.set_title(f"{title}\n{subtitle}")
            ax.set_xlim(0, 1)
            
            # Highlight true and predicted labels
            true_idx = true_labels[idx]
            pred_idx = pred_labels[idx]
            
            # Highlight bars for true and predicted labels
            bars = ax.patches
            bars[true_idx].set_facecolor('green')
            if true_idx != pred_idx:
                bars[pred_idx].set_facecolor('red')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = os.path.join(self.output_dir, "prediction_examples.png")
        
        plt.savefig(save_path)
        logger.info(f"Prediction examples saved to {save_path}")
        
        # Log to W&B if enabled
        if self.use_wandb:
            wandb.log({"prediction_examples": wandb.Image(fig)})
        
        plt.close(fig)
    
    def save_metrics_history(self, save_path: Optional[str] = None):
        """
        Save metrics history to JSON file.
        
        Args:
            save_path: Path to save the metrics history (if None, will be saved in output_dir)
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "metrics_history.json")
        
        with open(save_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logger.info(f"Metrics history saved to {save_path}")
    
    def close(self):
        """Clean up resources and finalize logging."""
        if self.use_wandb:
            wandb.finish()


class TrainerCallback:
    """
    Callback class for HuggingFace Trainer to log metrics and visualizations.
    Integrates with TrainingMonitor for consistent tracking.
    """
    
    def __init__(self, monitor: TrainingMonitor):
        """
        Initialize callback with a training monitor.
        
        Args:
            monitor: TrainingMonitor instance for tracking and visualization
        """
        self.monitor = monitor
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when the trainer logs metrics.
        
        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            logs: Dictionary of logs
        """
        if logs:
            # Extract relevant metrics
            metrics = {}
            for key, value in logs.items():
                if key in self.monitor.metrics_history:
                    metrics[key] = value
            
            # Get current step and epoch
            step = state.global_step
            epoch = state.epoch if hasattr(state, 'epoch') else None
            
            # Log metrics
            self.monitor.log_metrics(metrics, step, epoch)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after evaluation.
        
        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            metrics: Dictionary of evaluation metrics
        """
        if metrics:
            # Log evaluation metrics
            step = state.global_step
            epoch = state.epoch if hasattr(state, 'epoch') else None
            
            # Filter metrics
            eval_metrics = {}
            for key, value in metrics.items():
                if key.startswith('eval_'):
                    eval_metrics[key] = value
            
            # Log metrics
            self.monitor.log_metrics(eval_metrics, step, epoch)
    
    def on_train_end(self, args, state, control, **kwargs):
        """
        Called at the end of training.
        
        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
        """
        # Plot final learning curve
        self.monitor.plot_learning_curve()
        
        # Save metrics history
        self.monitor.save_metrics_history()
        
        # Clean up
        self.monitor.close()


if __name__ == "__main__":
    # Simple test
    monitor = TrainingMonitor(
        output_dir="./test_monitor",
        emotion_labels=["happy", "sad", "angry", "surprised", "neutral"]
    )
    
    # Simulate training
    for step in range(100):
        metrics = {
            "train_loss": np.random.rand() * 0.5 + 0.5 * np.exp(-step/30),
            "learning_rate": 0.001 * (1 - step/100)
        }
        
        if step % 10 == 0:
            metrics.update({
                "eval_loss": np.random.rand() * 0.3 + 0.7 * np.exp(-step/30),
                "eval_accuracy": min(0.99, 0.5 + 0.4 * (1 - np.exp(-step/30)) + np.random.rand() * 0.1),
                "eval_f1": min(0.99, 0.5 + 0.4 * (1 - np.exp(-step/30)) + np.random.rand() * 0.1)
            })
        
        monitor.log_metrics(metrics, step, epoch=step // 20)
    
    # Plot learning curve
    monitor.plot_learning_curve()
    
    # Plot confusion matrix
    y_true = np.random.randint(0, 5, size=100)
    y_pred = np.array([t if np.random.rand() > 0.3 else np.random.randint(0, 5) for t in y_true])
    monitor.plot_confusion_matrix(y_true, y_pred)
    
    # Log classification report
    monitor.log_classification_report(y_true, y_pred)
    
    # Clean up
    monitor.close()