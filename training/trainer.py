import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from models.ml_model import SoccerPredictionModel
from data.database.manager import DatabaseManager
from data.processing.data_processor import DataProcessor
from config.settings import config

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and optimize soccer prediction models"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.data_processor = DataProcessor()
        self.result_model = None
        self.btts_model = None
        self.best_params = {}
        
    def train_models(self, retrain: bool = False) -> Tuple[SoccerPredictionModel, SoccerPredictionModel]:
        """Train or retrain prediction models"""
        try:
            # Check if models exist and should be retrained
            models_dir = Path("models")
            result_model_path = models_dir / "result_model.pkl"
            btts_model_path = models_dir / "btts_model.pkl"
            
            if not retrain and result_model_path.exists() and btts_model_path.exists():
                logger.info("Loading existing models")
                self.result_model = SoccerPredictionModel(model_type='ensemble')
                self.btts_model = SoccerPredictionModel(model_type='ensemble')
                self.result_model.load(str(result_model_path))
                self.btts_model.load(str(btts_model_path))
                return self.result_model, self.btts_model
            
            # Get training data
            historical_data = self.db_manager.get_historical_matches(
                limit=config.model.min_training_samples * 2
            )
            
            if historical_data.empty:
                raise ValueError("No historical data available for training")
            
            logger.info(f"Training with {len(historical_data)} historical matches")
            
            # Process data
            X, y_result, y_btts = self.data_processor.process_training_data(historical_data)
            
            if len(X) < config.model.min_training_samples:
                raise ValueError(f"Insufficient training data: {len(X)} samples, minimum {config.model.min_training_samples} required")
            
            # Split data
            X_train, X_test, y_result_train, y_result_test, y_btts_train, y_btts_test = train_test_split(
                X, y_result, y_btts,
                test_size=config.model.test_size,
                random_state=config.model.random_state,
                stratify=y_result
            )
            
            # Fit scalers on training data
            self.data_processor.fit_scalers(X_train)
            
            # Transform features
            X_train_scaled = self.data_processor.transform_features(X_train)
            X_test_scaled = self.data_processor.transform_features(X_test)
            
            # Train result model
            logger.info("Training result prediction model...")
            self.result_model = self._train_result_model(X_train_scaled, y_result_train, X_test_scaled, y_result_test)
            
            # Train BTTS model
            logger.info("Training BTTS prediction model...")
            self.btts_model = self._train_btts_model(X_train_scaled, y_btts_train, X_test_scaled, y_btts_test)
            
            # Save models
            models_dir.mkdir(exist_ok=True)
            self.result_model.save(str(result_model_path))
            self.btts_model.save(str(btts_model_path))
            
            # Evaluate models
            result_metrics = self.result_model.evaluate(X_test_scaled, y_result_test)
            btts_metrics = self.btts_model.evaluate(X_test_scaled, y_btts_test)
            
            # Save performance to database
            self._save_performance_metrics(result_metrics, btts_metrics)
            
            logger.info("Model training completed successfully")
            return self.result_model, self.btts_model
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    def _train_result_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series) -> SoccerPredictionModel:
        """Train result prediction model"""
        model = SoccerPredictionModel(model_type=config.model.model_type)
        
        # Hyperparameter optimization if needed
        if config.model.model_type == 'ensemble':
            # Use ensemble with optimized base models
            model.train(X_train, y_train, X_val, y_val)
        else:
            # Single model training with optimization
            best_params = self._optimize_hyperparameters(X_train, y_train, model_type='result')
            model.model_type = config.model.model_type
            model.best_params = best_params
            model.train(X_train, y_train, X_val, y_val)
        
        return model
    
    def _train_btts_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series) -> SoccerPredictionModel:
        """Train BTTS prediction model"""
        model = SoccerPredictionModel(model_type=config.model.model_type)
        
        # Hyperparameter optimization if needed
        if config.model.model_type == 'ensemble':
            model.train(X_train, y_train, X_val, y_val)
        else:
            best_params = self._optimize_hyperparameters(X_train, y_train, model_type='btts')
            model.model_type = config.model.model_type
            model.best_params = best_params
            model.train(X_train, y_train, X_val, y_val)
        
        return model
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                 model_type: str = 'result') -> Dict:
        """Optimize hyperparameters using Optuna"""
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{model_type}_model_optimization',
            storage=f"sqlite:///optuna_studies/{model_type}.db",
            load_if_exists=True
        )
        
        def objective(trial):
            # Suggest hyperparameters
            if config.model.model_type == 'xgb':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0)
                }
                model = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False)
                
            elif config.model.model_type == 'lgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0)
                }
                model = lgb.LGBMClassifier(**params, random_state=42)
                
            elif config.model.model_type == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 500),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 1.0),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'random_strength': trial.suggest_loguniform('random_strength', 1e-8, 1.0)
                }
                model = CatBoostClassifier(**params, random_state=42, verbose=False)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            
            return scores.mean()
        
        # Run optimization
        n_trials = 50 if len(X) > 1000 else 30
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters for {model_type}: {study.best_params}")
        logger.info(f"Best CV score: {study.best_value}")
        
        return study.best_params
    
    def _save_performance_metrics(self, result_metrics: Dict, btts_metrics: Dict):
        """Save model performance metrics to database"""
        try:
            # Save result model performance
            self.db_manager.save_model_performance(
                model_version=self.result_model.model_version,
                accuracy=result_metrics['accuracy'],
                precision=result_metrics['precision'],
                recall=result_metrics['recall'],
                f1_score=result_metrics['f1_score'],
                roc_auc=result_metrics['roc_auc'],
                log_loss=result_metrics['log_loss'],
                model_type='result',
                features_list=config.model.features,
                test_size=int(len(result_metrics.get('test_indices', [])))
            )
            
            # Save BTTS model performance
            self.db_manager.save_model_performance(
                model_version=self.btts_model.model_version,
                accuracy=btts_metrics['accuracy'],
                precision=btts_metrics['precision'],
                recall=btts_metrics['recall'],
                f1_score=btts_metrics['f1_score'],
                roc_auc=btts_metrics['roc_auc'],
                log_loss=btts_metrics['log_loss'],
                model_type='btts',
                features_list=config.model.features,
                test_size=int(len(btts_metrics.get('test_indices', [])))
            )
            
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
    
    def incremental_training(self, new_data: pd.DataFrame):
        """Incremental training with new data"""
        try:
            if self.result_model is None or self.btts_model is None:
                self.train_models()
                return
            
            # Process new data
            X_new, y_result_new, y_btts_new = self.data_processor.process_training_data(new_data)
            
            if len(X_new) < 50:  # Minimum batch size for incremental training
                logger.info(f"Insufficient new data for incremental training: {len(X_new)} samples")
                return
            
            # Scale new data
            X_new_scaled = self.data_processor.transform_features(X_new)
            
            # Update models incrementally
            self.result_model.continuous_learning(X_new_scaled, y_result_new)
            self.btts_model.continuous_learning(X_new_scaled, y_btts_new)
            
            # Save updated models
            models_dir = Path("models")
            result_model_path = models_dir / "result_model.pkl"
            btts_model_path = models_dir / "btts_model.pkl"
            
            self.result_model.save(str(result_model_path))
            self.btts_model.save(str(btts_model_path))
            
            logger.info(f"Incremental training completed with {len(X_new)} new samples")
            
        except Exception as e:
            logger.error(f"Error in incremental training: {e}")
    
    def retrain_if_needed(self) -> bool:
        """Check if models need retraining and retrain if necessary"""
        try:
            models_dir = Path("models")
            result_model_path = models_dir / "result_model.pkl"
            
            if not result_model_path.exists():
                logger.info("Models don't exist, training new models")
                self.train_models()
                return True
            
            # Check last modification time
            model_age = datetime.now() - datetime.fromtimestamp(result_model_path.stat().st_mtime)
            
            if model_age.days >= config.model.retrain_frequency_days:
                logger.info(f"Models are {model_age.days} days old, retraining...")
                self.train_models(retrain=True)
                return True
            else:
                logger.info(f"Models are fresh ({model_age.days} days old), no retraining needed")
                return False
                
        except Exception as e:
            logger.error(f"Error checking retraining need: {e}")
            return False
