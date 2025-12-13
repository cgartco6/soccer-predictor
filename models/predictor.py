import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
from .feature_engineer import FeatureEngineer
from .ml_model import SoccerPredictionModel
from data.database.manager import DatabaseManager

logger = logging.getLogger(__name__)

class SoccerPredictor:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.feature_engineer = FeatureEngineer()
        self.result_model = SoccerPredictionModel(model_type='ensemble')
        self.btts_model = SoccerPredictionModel(model_type='ensemble')
        self.is_initialized = False
        
    def initialize_models(self):
        """Initialize models from database or train new ones"""
        try:
            # Try to load existing models
            self.result_model.load('models/result_model.pkl')
            self.btts_model.load('models/btts_model.pkl')
            logger.info("Loaded existing models")
        except:
            logger.info("Training new models...")
            self.train_models()
            
        self.is_initialized = True
    
    def train_models(self):
        """Train models using historical data"""
        # Get historical data
        historical_data = self.db_manager.get_historical_matches(limit=10000)
        
        if historical_data.empty:
            logger.error("No historical data available for training")
            return
        
        # Prepare features and labels
        X, y_result, y_btts = self._prepare_training_data(historical_data)
        
        # Split data
        X_train, X_test, y_result_train, y_result_test, y_btts_train, y_btts_test = train_test_split(
            X, y_result, y_btts, test_size=0.2, random_state=42
        )
        
        # Train result model
        self.result_model.train(X_train, y_result_train)
        result_metrics = self.result_model.evaluate(X_test, y_result_test)
        logger.info(f"Result model metrics: {result_metrics}")
        
        # Train BTTS model
        self.btts_model.train(X_train, y_btts_train)
        btts_metrics = self.btts_model.evaluate(X_test, y_btts_test)
        logger.info(f"BTTS model metrics: {btts_metrics}")
        
        # Save models
        self.result_model.save('models/result_model.pkl')
        self.btts_model.save('models/btts_model.pkl')
        
        # Save performance metrics to database
        self.db_manager.save_model_performance(
            model_version=self.result_model.model_version,
            accuracy=result_metrics['accuracy'],
            precision=result_metrics['precision'],
            recall=result_metrics['recall'],
            f1_score=result_metrics['f1_score'],
            roc_auc=result_metrics['roc_auc'],
            log_loss=result_metrics['log_loss'],
            model_type='result'
        )
    
    def _prepare_training_data(self, historical_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training data from historical matches"""
        features_list = []
        result_labels = []
        btts_labels = []
        
        for _, match in historical_data.iterrows():
            # Create features for each historical match
            match_features = self.feature_engineer.create_features(
                match.to_dict(),
                historical_data
            )
            
            if not match_features.empty:
                features_list.append(match_features.iloc[0])
                
                # Result labels (H/D/A)
                result_labels.append(match['result'])
                
                # BTTS labels
                btts_labels.append(1 if match['btts_result'] else 0)
        
        X = pd.DataFrame(features_list)
        y_result = pd.Series(result_labels)
        y_btts = pd.Series(btts_labels)
        
        return X, y_result, y_btts
    
    def predict_todays_matches(self) -> List[Dict]:
        """Predict today's matches"""
        if not self.is_initialized:
            self.initialize_models()
        
        # Get today's matches
        today_matches = self.db_manager.get_todays_matches()
        
        predictions = []
        
        for match in today_matches:
            try:
                # Create features for prediction
                match_features = self.feature_engineer.create_features(
                    match,
                    self.db_manager.get_historical_matches()
                )
                
                if match_features.empty:
                    continue
                
                # Transform features
                X = self.feature_engineer.transform(match_features)
                
                # Predict result
                result_pred, result_prob = self.result_model.predict(X)
                
                # Predict BTTS
                btts_pred, btts_prob = self.btts_model.predict(X)
                
                # Create prediction object
                prediction = {
                    'match_id': match.get('id'),
                    'home_team': match.get('home_team'),
                    'away_team': match.get('away_team'),
                    'league': match.get('league'),
                    'match_time': match.get('match_time'),
                    
                    # Result predictions
                    'predicted_result': result_pred[0],
                    'result_probabilities': {
                        'home_win': float(result_prob[0][0]),
                        'draw': float(result_prob[0][1]),
                        'away_win': float(result_prob[0][2])
                    },
                    
                    # BTTS predictions
                    'predicted_btts': bool(btts_pred[0]),
                    'btts_probabilities': {
                        'yes': float(btts_prob[0][1]),
                        'no': float(btts_prob[0][0])
                    },
                    
                    # Confidence scores
                    'result_confidence': float(np.max(result_prob[0])),
                    'btts_confidence': float(np.max(btts_prob[0])),
                    
                    # Bookmaker odds for comparison
                    'odds': match.get('odds', {}),
                    
                    # Recommended bets (if confidence is high)
                    'recommended_bets': self._generate_recommendations(
                        result_prob[0], 
                        btts_prob[0],
                        match.get('odds', {})
                    ),
                    
                    'prediction_timestamp': datetime.now().isoformat(),
                    'model_version': self.result_model.model_version
                }
                
                predictions.append(prediction)
                
                # Save prediction to database
                self.db_manager.save_prediction(prediction)
                
            except Exception as e:
                logger.error(f"Error predicting match {match.get('home_team')} vs {match.get('away_team')}: {e}")
                continue
        
        return predictions
    
    def _generate_recommendations(self, result_probs: np.ndarray, 
                                 btts_probs: np.ndarray, 
                                 odds: Dict) -> List[Dict]:
        """Generate betting recommendations"""
        recommendations = []
        
        # Check for value bets
        implied_prob_home = 1 / odds.get('home_win', 2.5) if odds.get('home_win') else None
        implied_prob_draw = 1 / odds.get('draw', 3.2) if odds.get('draw') else None
        implied_prob_away = 1 / odds.get('away_win', 2.8) if odds.get('away_win') else None
        
        # Result recommendations
        if implied_prob_home and result_probs[0] > implied_prob_home * 1.1:
            recommendations.append({
                'type': 'match_result',
                'bet': 'home_win',
                'confidence': result_probs[0],
                'value': result_probs[0] - implied_prob_home,
                'odds': odds.get('home_win'),
                'stake_recommendation': self._calculate_stake(result_probs[0], implied_prob_home)
            })
        
        if implied_prob_draw and result_probs[1] > implied_prob_draw * 1.1:
            recommendations.append({
                'type': 'match_result',
                'bet': 'draw',
                'confidence': result_probs[1],
                'value': result_probs[1] - implied_prob_draw,
                'odds': odds.get('draw'),
                'stake_recommendation': self._calculate_stake(result_probs[1], implied_prob_draw)
            })
        
        if implied_prob_away and result_probs[2] > implied_prob_away * 1.1:
            recommendations.append({
                'type': 'match_result',
                'bet': 'away_win',
                'confidence': result_probs[2],
                'value': result_probs[2] - implied_prob_away,
                'odds': odds.get('away_win'),
                'stake_recommendation': self._calculate_stake(result_probs[2], implied_prob_away)
            })
        
        # BTTS recommendations
        implied_prob_btts_yes = 1 / odds.get('btts_yes', 1.8) if odds.get('btts_yes') else None
        implied_prob_btts_no = 1 / odds.get('btts_no', 2.0) if odds.get('btts_no') else None
        
        if implied_prob_btts_yes and btts_probs[1] > implied_prob_btts_yes * 1.1:
            recommendations.append({
                'type': 'btts',
                'bet': 'yes',
                'confidence': btts_probs[1],
                'value': btts_probs[1] - implied_prob_btts_yes,
                'odds': odds.get('btts_yes'),
                'stake_recommendation': self._calculate_stake(btts_probs[1], implied_prob_btts_yes)
            })
        
        if implied_prob_btts_no and btts_probs[0] > implied_prob_btts_no * 1.1:
            recommendations.append({
                'type': 'btts',
                'bet': 'no',
                'confidence': btts_probs[0],
                'value': btts_probs[0] - implied_prob_btts_no,
                'odds': odds.get('btts_no'),
                'stake_recommendation': self._calculate_stake(btts_probs[0], implied_prob_btts_no)
            })
        
        return recommendations
    
    def _calculate_stake(self, predicted_prob: float, implied_prob: float) -> str:
        """Calculate recommended stake based on Kelly Criterion"""
        odds = 1 / implied_prob
        kelly = (predicted_prob * odds - 1) / (odds - 1)
        
        if kelly <= 0:
            return 'avoid'
        elif kelly <= 0.02:
            return 'small'
        elif kelly <= 0.05:
            return 'medium'
        else:
            return 'large'
    
    def update_with_results(self):
        """Update models with actual results (continuous learning)"""
        # Get recent predictions with actual results
        recent_predictions = self.db_manager.get_recent_predictions_with_results()
        
        if not recent_predictions.empty:
            # Prepare data for continuous learning
            new_data = []
            new_result_labels = []
            new_btts_labels = []
            
            for _, row in recent_predictions.iterrows():
                # Create features for the match
                features = self.feature_engineer.create_features(
                    row.to_dict(),
                    self.db_manager.get_historical_matches()
                )
                
                if not features.empty:
                    new_data.append(features.iloc[0])
                    new_result_labels.append(row['actual_result'])
                    new_btts_labels.append(1 if row['actual_btts'] else 0)
            
            if new_data:
                X_new = pd.DataFrame(new_data)
                X_transformed = self.feature_engineer.transform(X_new)
                
                # Update models
                self.result_model.continuous_learning(
                    X_transformed, 
                    pd.Series(new_result_labels)
                )
                self.btts_model.continuous_learning(
                    X_transformed, 
                    pd.Series(new_btts_labels)
                )
                
                # Save updated models
                self.result_model.save('models/result_model.pkl')
                self.btts_model.save('models/btts_model.pkl')
                
                logger.info("Models updated with new results")
