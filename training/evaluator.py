import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate model performance and generate insights"""
    
    def __init__(self):
        self.metrics_history = []
        self.feature_importance_history = []
        
    def evaluate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: np.ndarray, model_type: str = 'result') -> Dict:
        """Comprehensive model evaluation"""
        try:
            metrics = {}
            
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Probabilistic metrics
            try:
                metrics['log_loss'] = log_loss(y_true, y_prob)
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except:
                metrics['log_loss'] = None
                metrics['roc_auc'] = None
            
            # Class-specific metrics
            unique_classes = np.unique(y_true)
            class_metrics = {}
            
            for cls in unique_classes:
                cls_true = (y_true == cls).astype(int)
                cls_pred = (y_pred == cls).astype(int)
                
                class_metrics[str(cls)] = {
                    'precision': precision_score(cls_true, cls_pred, zero_division=0),
                    'recall': recall_score(cls_true, cls_pred, zero_division=0),
                    'f1': f1_score(cls_true, cls_pred, zero_division=0),
                    'support': int(np.sum(cls_true))
                }
            
            metrics['class_metrics'] = class_metrics
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Classification report
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics['classification_report'] = report
            
            # Calculate profit/loss if odds are available
            if hasattr(self, 'odds_data'):
                metrics['profit_analysis'] = self._calculate_profit(y_true, y_pred, y_prob)
            
            # Store in history
            self.metrics_history.append({
                'timestamp': datetime.now().isoformat(),
                'model_type': model_type,
                'metrics': metrics
            })
            
            logger.info(f"Model evaluation completed for {model_type} model")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def _calculate_profit(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: np.ndarray) -> Dict:
        """Calculate potential profit from predictions"""
        profit_analysis = {
            'total_bets': 0,
            'correct_bets': 0,
            'total_stake': 0,
            'total_return': 0,
            'profit': 0,
            'roi': 0
        }
        
        try:
            # Simulate betting strategy
            for i in range(len(y_true)):
                true_class = y_true[i]
                pred_class = y_pred[i]
                pred_prob = np.max(y_prob[i])
                
                # Only bet if confidence is high
                if pred_prob >= 0.65:  # Configurable threshold
                    profit_analysis['total_bets'] += 1
                    profit_analysis['total_stake'] += 1  # Unit stake
                    
                    if pred_class == true_class:
                        profit_analysis['correct_bets'] += 1
                        # Calculate returns based on odds
                        odds = self.odds_data[i] if i < len(self.odds_data) else 2.0
                        profit_analysis['total_return'] += odds
                    else:
                        profit_analysis['total_return'] += 0
            
            # Calculate profit metrics
            profit_analysis['profit'] = profit_analysis['total_return'] - profit_analysis['total_stake']
            if profit_analysis['total_stake'] > 0:
                profit_analysis['roi'] = (profit_analysis['profit'] / profit_analysis['total_stake']) * 100
            
        except Exception as e:
            logger.error(f"Error calculating profit: {e}")
        
        return profit_analysis
    
    def evaluate_feature_importance(self, model, feature_names: List[str]) -> Dict:
        """Evaluate feature importance"""
        try:
            importance_dict = {}
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                for i in indices:
                    if i < len(feature_names):
                        importance_dict[feature_names[i]] = float(importances[i])
            
            # Store in history
            self.feature_importance_history.append({
                'timestamp': datetime.now().isoformat(),
                'feature_importance': importance_dict
            })
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error evaluating feature importance: {e}")
            return {}
    
    def generate_performance_report(self, metrics: Dict, model_type: str = 'result') -> str:
        """Generate detailed performance report"""
        try:
            report = f"""
            ========================================
            {model_type.upper()} MODEL PERFORMANCE REPORT
            ========================================
            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            OVERALL METRICS:
            ----------------
            Accuracy:   {metrics.get('accuracy', 0):.4f}
            Precision:  {metrics.get('precision', 0):.4f}
            Recall:     {metrics.get('recall', 0):.4f}
            F1-Score:   {metrics.get('f1_score', 0):.4f}
            Log Loss:   {metrics.get('log_loss', 'N/A')}
            ROC AUC:    {metrics.get('roc_auc', 'N/A')}
            
            """
            
            # Add class-specific metrics
            if 'class_metrics' in metrics:
                report += "CLASS-SPECIFIC METRICS:\n"
                report += "----------------------\n"
                for cls, cls_metrics in metrics['class_metrics'].items():
                    report += f"Class {cls}:\n"
                    report += f"  Precision: {cls_metrics['precision']:.4f}\n"
                    report += f"  Recall:    {cls_metrics['recall']:.4f}\n"
                    report += f"  F1-Score:  {cls_metrics['f1']:.4f}\n"
                    report += f"  Support:   {cls_metrics['support']}\n"
                    report += "\n"
            
            # Add profit analysis if available
            if 'profit_analysis' in metrics:
                profit = metrics['profit_analysis']
                report += "PROFIT ANALYSIS:\n"
                report += "----------------\n"
                report += f"Total Bets:      {profit['total_bets']}\n"
                report += f"Correct Bets:    {profit['correct_bets']} ({profit['correct_bets']/max(profit['total_bets'], 1)*100:.1f}%)\n"
                report += f"Total Stake:     ${profit['total_stake']:.2f}\n"
                report += f"Total Return:    ${profit['total_return']:.2f}\n"
                report += f"Profit:          ${profit['profit']:.2f}\n"
                report += f"ROI:             {profit['roi']:.2f}%\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {str(e)}"
    
    def create_visualizations(self, metrics: Dict, feature_importance: Dict, 
                            save_path: str = "reports/visualizations"):
        """Create visualization plots"""
        try:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            # 1. Confusion Matrix Heatmap
            if 'confusion_matrix' in metrics:
                self._plot_confusion_matrix(metrics['confusion_matrix'], save_path)
            
            # 2. Feature Importance Bar Chart
            if feature_importance:
                self._plot_feature_importance(feature_importance, save_path)
            
            # 3. ROC Curve (for binary classification)
            if 'roc_curve' in metrics:
                self._plot_roc_curve(metrics['roc_curve'], save_path)
            
            # 4. Precision-Recall Curve
            if 'precision_recall_curve' in metrics:
                self._plot_precision_recall_curve(metrics['precision_recall_curve'], save_path)
            
            # 5. Performance History
            if self.metrics_history:
                self._plot_performance_history(save_path)
            
            logger.info(f"Visualizations saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _plot_confusion_matrix(self, cm: List[List[int]], save_path: str):
        """Plot confusion matrix"""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
    
    def _plot_feature_importance(self, feature_importance: Dict, save_path: str):
        """Plot feature importance"""
        try:
            # Get top 20 features
            top_features = dict(sorted(feature_importance.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True)[:20])
            
            plt.figure(figsize=(12, 8))
            features = list(top_features.keys())
            importances = list(top_features.values())
            
            y_pos = np.arange(len(features))
            plt.barh(y_pos, importances, align='center')
            plt.yticks(y_pos, features)
            plt.xlabel('Importance')
            plt.title('Top 20 Feature Importance')
            plt.gca().invert_yaxis()
            plt.savefig(f"{save_path}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
    
    def _plot_performance_history(self, save_path: str):
        """Plot performance history over time"""
        try:
            if len(self.metrics_history) < 2:
                return
            
            timestamps = []
            accuracies = []
            f1_scores = []
            
            for entry in self.metrics_history:
                timestamps.append(datetime.fromisoformat(entry['timestamp']))
                accuracies.append(entry['metrics'].get('accuracy', 0))
                f1_scores.append(entry['metrics'].get('f1_score', 0))
            
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, accuracies, label='Accuracy', marker='o')
            plt.plot(timestamps, f1_scores, label='F1-Score', marker='s')
            plt.xlabel('Date')
            plt.ylabel('Score')
            plt.title('Model Performance History')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{save_path}/performance_history.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error plotting performance history: {e}")
    
    def create_interactive_dashboard(self, metrics: Dict, feature_importance: Dict) -> go.Figure:
        """Create interactive dashboard with Plotly"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Confusion Matrix', 'Feature Importance',
                              'Performance Metrics', 'Class Distribution'),
                specs=[[{'type': 'heatmap'}, {'type': 'bar'}],
                      [{'type': 'bar'}, {'type': 'pie'}]]
            )
            
            # 1. Confusion Matrix
            if 'confusion_matrix' in metrics:
                cm = np.array(metrics['confusion_matrix'])
                fig.add_trace(
                    go.Heatmap(z=cm, colorscale='Blues', showscale=True),
                    row=1, col=1
                )
            
            # 2. Feature Importance
            if feature_importance:
                top_features = dict(sorted(feature_importance.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True)[:10])
                fig.add_trace(
                    go.Bar(x=list(top_features.values()), 
                          y=list(top_features.keys()),
                          orientation='h'),
                    row=1, col=2
                )
            
            # 3. Performance Metrics
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metric_values = [
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1_score', 0)
            ]
            
            fig.add_trace(
                go.Bar(x=metric_names, y=metric_values,
                      marker_color=['blue', 'green', 'orange', 'red']),
                row=2, col=1
            )
            
            # 4. Class Distribution
            if 'class_metrics' in metrics:
                class_names = list(metrics['class_metrics'].keys())
                class_supports = [m['support'] for m in metrics['class_metrics'].values()]
                
                fig.add_trace(
                    go.Pie(labels=class_names, values=class_supports),
                    row=2, col=2
                )
            
            fig.update_layout(height=800, showlegend=False,
                            title_text="Model Performance Dashboard")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {e}")
            return None
    
    def save_evaluation_results(self, metrics: Dict, feature_importance: Dict, 
                              save_path: str = "reports"):
        """Save evaluation results to files"""
        try:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save metrics as JSON
            metrics_file = f"{save_path}/metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save feature importance
            if feature_importance:
                importance_file = f"{save_path}/feature_importance_{timestamp}.json"
                with open(importance_file, 'w') as f:
                    json.dump(feature_importance, f, indent=2)
            
            # Generate and save report
            report = self.generate_performance_report(metrics)
            report_file = f"{save_path}/report_{timestamp}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Evaluation results saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
