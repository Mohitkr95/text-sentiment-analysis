import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from typing import Dict, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations."""
    
    def __init__(self, save_plots: bool = True):
        self.logger = setup_logger(self.__class__.__name__)
        self.save_plots = save_plots
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro')
        }
        
        # Add AUC if probabilities are available
        if y_proba is not None:
            try:
                if y_proba.shape[1] == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    metrics['avg_precision'] = average_precision_score(y_true, y_proba[:, 1])
                else:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC scores: {e}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: list = None, title: str = "Confusion Matrix") -> plt.Figure:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        if self.save_plots:
            plt.savefig(f'logs/{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      title: str = "ROC Curve") -> plt.Figure:
        """Plot ROC curve for binary classification."""
        if y_proba.shape[1] != 2:
            self.logger.warning("ROC curve plotting only supports binary classification")
            return None
        
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auc_score = roc_auc_score(y_true, y_proba[:, 1])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        if self.save_plots:
            plt.savefig(f'logs/{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   title: str = "Precision-Recall Curve") -> plt.Figure:
        """Plot precision-recall curve for binary classification."""
        if y_proba.shape[1] != 2:
            self.logger.warning("PR curve plotting only supports binary classification")
            return None
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_proba[:, 1])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2,
               label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        if self.save_plots:
            plt.savefig(f'logs/{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, model_results: Dict[str, Dict[str, float]], 
                            metric: str = 'f1_score', title: str = "Model Comparison") -> plt.Figure:
        """Plot comparison of multiple models."""
        models = list(model_results.keys())
        scores = [model_results[model].get(metric, 0) for model in models]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(models, scores, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.annotate(f'{score:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        ax.set_xlabel('Models')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if they're long
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(f'logs/{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, model_results: Dict[str, Dict[str, float]], 
                                   y_true: np.ndarray = None, y_pred: np.ndarray = None,
                                   y_proba: np.ndarray = None) -> go.Figure:
        """Create an interactive dashboard with plotly."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Comparison (F1 Score)', 'Model Comparison (Accuracy)', 
                          'Confusion Matrix', 'ROC Curve'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        models = list(model_results.keys())
        f1_scores = [model_results[model].get('f1_score', 0) for model in models]
        accuracies = [model_results[model].get('accuracy', 0) for model in models]
        
        # F1 Score comparison
        fig.add_trace(
            go.Bar(x=models, y=f1_scores, name="F1 Score", marker_color='lightblue'),
            row=1, col=1
        )
        
        # Accuracy comparison
        fig.add_trace(
            go.Bar(x=models, y=accuracies, name="Accuracy", marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Confusion matrix (if data provided)
        if y_true is not None and y_pred is not None:
            cm = confusion_matrix(y_true, y_pred)
            fig.add_trace(
                go.Heatmap(z=cm, colorscale='Blues', showscale=False),
                row=2, col=1
            )
        
        # ROC curve (if probabilities provided)
        if y_true is not None and y_proba is not None and y_proba.shape[1] == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
            auc_score = roc_auc_score(y_true, y_proba[:, 1])
            
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines', 
                          name=f'ROC (AUC={auc_score:.3f})', line=dict(color='orange')),
                row=2, col=2
            )
            
            # Add diagonal line
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                          name='Random', line=dict(dash='dash', color='gray')),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, title_text="Model Evaluation Dashboard")
        
        if self.save_plots:
            fig.write_html("logs/interactive_dashboard.html")
        
        return fig
    
    def generate_detailed_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_proba: np.ndarray = None, model_name: str = "Model") -> str:
        """Generate a detailed evaluation report."""
        metrics = self.calculate_metrics(y_true, y_pred, y_proba)
        
        report = f"""
=== {model_name} Evaluation Report ===

Classification Metrics:
- Accuracy: {metrics['accuracy']:.4f}
- Precision (Weighted): {metrics['precision']:.4f}
- Recall (Weighted): {metrics['recall']:.4f}
- F1 Score (Weighted): {metrics['f1_score']:.4f}
- F1 Score (Macro): {metrics['f1_macro']:.4f}
- F1 Score (Micro): {metrics['f1_micro']:.4f}
"""
        
        if 'roc_auc' in metrics:
            report += f"- ROC AUC: {metrics['roc_auc']:.4f}\n"
        if 'avg_precision' in metrics:
            report += f"- Average Precision: {metrics['avg_precision']:.4f}\n"
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        report += f"\nConfusion Matrix:\n{cm}\n"
        
        # Add detailed classification report
        report += f"\nDetailed Classification Report:\n"
        report += classification_report(y_true, y_pred)
        
        return report
    
    def evaluate_model_comprehensive(self, model, X_test: np.ndarray, y_test: np.ndarray,
                                   model_name: str = "Model") -> Dict[str, Any]:
        """Perform comprehensive evaluation of a model."""
        self.logger.info(f"Performing comprehensive evaluation of {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except:
                self.logger.warning("Could not get prediction probabilities")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_proba)
        
        # Create visualizations
        class_names = ['Negative', 'Positive']  # For sentiment analysis
        
        plots = {}
        plots['confusion_matrix'] = self.plot_confusion_matrix(
            y_test, y_pred, class_names, f"{model_name} - Confusion Matrix"
        )
        
        if y_proba is not None and y_proba.shape[1] == 2:
            plots['roc_curve'] = self.plot_roc_curve(
                y_test, y_proba, f"{model_name} - ROC Curve"
            )
            plots['pr_curve'] = self.plot_precision_recall_curve(
                y_test, y_proba, f"{model_name} - Precision-Recall Curve"
            )
        
        # Generate detailed report
        detailed_report = self.generate_detailed_report(y_test, y_pred, y_proba, model_name)
        
        # Log key metrics
        self.logger.info(f"{model_name} Evaluation Results:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_proba,
            'plots': plots,
            'detailed_report': detailed_report
        } 