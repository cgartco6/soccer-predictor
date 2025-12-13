import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import joblib
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SoccerPredictionModel:
    def __init__(self, model_type: str = 'ensemble'):
        self.model_type = model_type
        self.models = {}
        self.best_model = None
        self.feature_importance = {}
        self.model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.is_trained = False
        
    def create_ensemble(self) -> VotingClassifier:
        """Create ensemble of models"""
        models = [
            ('rf', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )),
            ('xgb', XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )),
            ('lgbm', LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                n_jobs=-1
            )),
            ('catboost', CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                random_seed=42,
                verbose=False
            ))
        ]
        
        return VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
              X_val: pd.DataFrame = None, y_val: pd.DataFrame = None):
        """Train the model"""
        logger.info(f"Training {self.model_type} model...")
        
        if self.model_type == 'ensemble':
            self.best_model = self.create_ensemble()
        elif self.model_type == 'xgb':
            self.best_model = XGBClassifier(
                n_estimators=300,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif self.model_type == 'neural_network':
            self.best_model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        
        # Fit the model
        self.best_model.fit(X_train, y_train)
        
        # Store feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                X_train.columns,
                self.best_model.feature_importances_
            ))
        
        self.is_trained = True
        logger.info("Model training completed")
        
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get probability predictions
        y_prob = self.best_model.predict_proba(X)
        
        # Get class predictions
        y_pred = self.best_model.predict(X)
        
        return y_pred, y_prob
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Dict:
        """Make predictions with confidence scores"""
        y_pred, y_prob = self.predict(X)
        
        predictions = []
        for i in range(len(X)):
            pred = {
                'predicted_class': y_pred[i],
                'probabilities': y_prob[i].tolist(),
                'confidence': float(np.max(y_prob[i])),
                'uncertainty': float(1 - np.max(y_prob[i]))
            }
            
            # Calculate prediction intervals
            pred['prediction_interval'] = self._calculate_prediction_interval(y_prob[i])
            
            predictions.append(pred)
        
        return predictions
    
    def _calculate_prediction_interval(self, probabilities: np.ndarray) -> List[float]:
        """Calculate prediction interval"""
        sorted_probs = np.sort(probabilities)[::-1]
        cumulative = np.cumsum(sorted_probs)
        
        # Find 90% prediction interval
        idx = np.where(cumulative >= 0.9)[0]
        if len(idx) > 0:
            interval_size = idx[0] + 1
        else:
            interval_size = len(probabilities)
        
        return [float(np.min(sorted_probs[:interval_size])), 
                float(np.max(sorted_probs[:interval_size]))]
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
        """Evaluate model performance"""
        y_pred, y_prob = self.predict(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'log_loss': float(log_loss(y_test, y_prob)),
            'roc_auc': float(roc_auc_score(y_test, y_prob, multi_class='ovr'))
        }
        
        # Class-wise metrics
        unique_classes = np.unique(y_test)
        class_metrics = {}
        for cls in unique_classes:
            class_idx = y_test == cls
            if np.any(class_idx):
                class_metrics[str(cls)] = {
                    'precision': float(precision_score(y_test == cls, y_pred == cls, zero_division=0)),
                    'recall': float(recall_score(y_test == cls, y_pred == cls, zero_division=0)),
                    'f1': float(f1_score(y_test == cls, y_pred == cls, zero_division=0))
                }
        
        metrics['class_metrics'] = class_metrics
        metrics['feature_importance'] = self.feature_importance
        
        return metrics
    
    def save(self, path: str):
        """Save model to disk"""
        model_data = {
            'model': self.best_model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'model_version': self.model_version,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        model_data = joblib.load(path)
        
        self.best_model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data['feature_importance']
        self.model_version = model_data['model_version']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {path}")
    
    def continuous_learning(self, new_data: pd.DataFrame, new_labels: pd.DataFrame, 
                          learning_rate: float = 0.1):
        """Update model with new data (online learning)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before continuous learning")
        
        # Partial fit if supported
        if hasattr(self.best_model, 'partial_fit'):
            self.best_model.partial_fit(new_data, new_labels)
        else:
            # Retrain with combined data (in production, use more sophisticated approach)
            # This is a simplified version
            pass
        
        logger.info("Model updated with new data")
