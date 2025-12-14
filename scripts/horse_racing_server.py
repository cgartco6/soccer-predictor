import os
import json
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Global variables for the AI system
model = None
scaler = None
label_encoders = {}
feature_columns = []
current_predictions = []

class HorseRacingAI:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.data = None
        self.initialized = False
        
    def generate_sample_data(self):
        """Generate realistic horse racing sample data"""
        np.random.seed(42)
        
        # Create sample races
        num_races = 50
        num_horses = 300
        
        data = []
        horse_id = 0
        
        for race_id in range(num_races):
            race_date = datetime.now() - timedelta(days=np.random.randint(0, 365))
            track = np.random.choice(['Churchill Downs', 'Santa Anita', 'Belmont Park', 'Keeneland', 'Gulfstream'])
            track_condition = np.random.choice(['Fast', 'Good', 'Sloppy', 'Wet Fast'])
            distance = np.random.choice([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
            race_class = np.random.choice(['Claiming', 'Allowance', 'Stakes', 'Graded Stakes'])
            
            horses_in_race = np.random.randint(6, 12)
            
            for position in range(horses_in_race):
                horse_id += 1
                
                # Generate horse features
                age = np.random.randint(2, 8)
                weight = np.random.randint(1000, 1200)
                odds = round(np.random.uniform(1.5, 20.0), 1)
                
                # Past performance metrics
                win_percent = np.random.uniform(0.05, 0.35)
                top3_percent = np.random.uniform(0.15, 0.60)
                avg_speed_rating = np.random.randint(70, 100)
                last_race_days = np.random.randint(7, 90)
                
                # Jockey and trainer stats
                jockey_win_percent = np.random.uniform(0.08, 0.25)
                trainer_win_percent = np.random.uniform(0.10, 0.30)
                
                # Calculate some derived features
                form_rating = (win_percent * 100 + top3_percent * 50 + avg_speed_rating) / 3
                experience_factor = np.log(age * 10 + 1)
                
                # Simulate finish position (1 = win, 2 = place, 3 = show, 4+ = loss)
                # Weight the probabilities based on features
                win_prob = (win_percent * 0.3 + jockey_win_percent * 0.3 + 
                           trainer_win_percent * 0.2 + (1/odds) * 0.2)
                
                # Add some randomness
                win_prob += np.random.normal(0, 0.1)
                win_prob = max(0.01, min(0.95, win_prob))
                
                # Determine actual finish (simplified)
                finish_position = position + 1 if np.random.random() > win_prob else 1
                
                horse_data = {
                    'race_id': race_id,
                    'horse_id': horse_id,
                    'horse_name': f'Horse_{horse_id}',
                    'age': age,
                    'weight': weight,
                    'odds': odds,
                    'win_percent': round(win_percent, 3),
                    'top3_percent': round(top3_percent, 3),
                    'avg_speed_rating': avg_speed_rating,
                    'last_race_days': last_race_days,
                    'jockey_win_percent': round(jockey_win_percent, 3),
                    'trainer_win_percent': round(trainer_win_percent, 3),
                    'form_rating': round(form_rating, 2),
                    'experience_factor': round(experience_factor, 2),
                    'track': track,
                    'track_condition': track_condition,
                    'distance': distance,
                    'race_class': race_class,
                    'race_date': race_date.strftime('%Y-%m-%d'),
                    'finish_position': finish_position,
                    'predicted_win_prob': round(win_prob, 3)
                }
                
                data.append(horse_data)
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def train_model(self):
        """Train the AI prediction model"""
        if self.data is None:
            self.generate_sample_data()
        
        # Prepare features and target
        features = ['age', 'weight', 'odds', 'win_percent', 'top3_percent', 
                   'avg_speed_rating', 'last_race_days', 'jockey_win_percent',
                   'trainer_win_percent', 'form_rating', 'experience_factor',
                   'distance']
        
        # Create binary target: 1 for win (position 1), 0 otherwise
        self.data['target'] = (self.data['finish_position'] == 1).astype(int)
        
        X = self.data[features].copy()
        y = self.data['target']
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train ensemble model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        rf_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        
        # Ensemble predictions
        rf_pred = rf_model.predict_proba(X_test)[:, 1]
        gb_pred = gb_model.predict_proba(X_test)[:, 1]
        ensemble_pred = (rf_pred * 0.5 + gb_pred * 0.5)
        
        # Calculate accuracy
        threshold = 0.5
        predictions = (ensemble_pred > threshold).astype(int)
        accuracy = accuracy_score(y_test, predictions)
        
        # Store the trained model
        self.model = {
            'rf': rf_model,
            'gb': gb_model,
            'scaler': self.scaler,
            'features': features,
            'accuracy': round(accuracy, 4)
        }
        
        # Update predictions in data
        X_all = self.scaler.transform(X)
        rf_all = rf_model.predict_proba(X_all)[:, 1]
        gb_all = gb_model.predict_proba(X_all)[:, 1]
        self.data['ai_win_probability'] = (rf_all * 0.5 + gb_all * 0.5)
        
        self.initialized = True
        print(f"Model trained with accuracy: {accuracy:.2%}")
        
        return self.model
    
    def predict_race(self, horse_data_list):
        """Predict outcomes for a list of horses in a race"""
        if not self.initialized:
            self.train_model()
        
        predictions = []
        
        for horse in horse_data_list:
            # Prepare features for prediction
            features = self.model['features']
            horse_df = pd.DataFrame([horse])
            
            # Ensure all required features are present
            for feature in features:
                if feature not in horse_df.columns:
                    horse_df[feature] = self.data[feature].mean()
            
            # Scale features
            X = horse_df[features].fillna(self.data[features].mean())
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            rf_prob = self.model['rf'].predict_proba(X_scaled)[0, 1]
            gb_prob = self.model['gb'].predict_proba(X_scaled)[0, 1]
            win_prob = (rf_prob * 0.5 + gb_prob * 0.5)
            
            # Calculate value bet indicator
            implied_prob = 1 / horse['odds'] if horse['odds'] > 0 else 0.01
            value_rating = win_prob / implied_prob if implied_prob > 0 else 0
            
            prediction = {
                'horse_name': horse.get('horse_name', 'Unknown'),
                'odds': horse['odds'],
                'predicted_win_probability': round(win_prob, 4),
                'implied_probability': round(implied_prob, 4),
                'value_rating': round(value_rating, 3),
                'is_value_bet': value_rating > 1.2,
                'recommended_bet': 'WIN' if win_prob > 0.3 else ('PLACE' if win_prob > 0.15 else 'SHOW'),
                'confidence_score': 'HIGH' if win_prob > 0.4 else ('MEDIUM' if win_prob > 0.25 else 'LOW')
            }
            
            predictions.append(prediction)
        
        # Sort by predicted win probability
        predictions.sort(key=lambda x: x['predicted_win_probability'], reverse=True)
        
        # Assign rank
        for i, pred in enumerate(predictions):
            pred['rank'] = i + 1
        
        return predictions
    
    def generate_upcoming_race(self):
        """Generate a simulated upcoming race"""
        tracks = ['Churchill Downs', 'Santa Anita', 'Belmont Park', 'Keeneland']
        conditions = ['Fast', 'Good', 'Sloppy']
        
        race_info = {
            'track': np.random.choice(tracks),
            'race_time': (datetime.now() + timedelta(hours=np.random.randint(1, 24))).strftime('%Y-%m-%d %H:%M'),
            'distance': round(np.random.uniform(5.0, 10.0), 1),
            'track_condition': np.random.choice(conditions),
            'race_class': np.random.choice(['Claiming', 'Allowance', 'Stakes']),
            'purse': f"${np.random.randint(20000, 100000):,}"
        }
        
        # Generate horses for this race
        horses = []
        num_horses = np.random.randint(6, 10)
        
        for i in range(num_horses):
            horse = {
                'horse_id': 1000 + i,
                'horse_name': f'Contender_{chr(65 + i)}',
                'age': np.random.randint(2, 7),
                'weight': np.random.randint(1050, 1175),
                'odds': round(np.random.uniform(1.5, 15.0), 1),
                'win_percent': round(np.random.uniform(0.08, 0.35), 3),
                'top3_percent': round(np.random.uniform(0.20, 0.65), 3),
                'avg_speed_rating': np.random.randint(75, 98),
                'last_race_days': np.random.randint(14, 60),
                'jockey_win_percent': round(np.random.uniform(0.10, 0.30), 3),
                'trainer_win_percent': round(np.random.uniform(0.12, 0.35), 3),
                'form_rating': round(np.random.uniform(65, 95), 1),
                'experience_factor': round(np.random.uniform(1.0, 2.5), 2)
            }
            horses.append(horse)
        
        return race_info, horses

# Initialize the AI system
ai_system = HorseRacingAI()

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/api/init', methods=['GET'])
def initialize_system():
    """Initialize and train the AI model"""
    try:
        model = ai_system.train_model()
        return jsonify({
            'status': 'success',
            'message': f'AI system initialized with accuracy: {model["accuracy"]:.2%}',
            'data_samples': len(ai_system.data),
            'features_used': len(model['features'])
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/generate_race', methods=['GET'])
def generate_race():
    """Generate a new upcoming race"""
    try:
        race_info, horses = ai_system.generate_upcoming_race()
        
        # Predict outcomes
        predictions = ai_system.predict_race(horses)
        
        # Combine horse data with predictions
        for horse, prediction in zip(horses, predictions):
            horse.update(prediction)
        
        return jsonify({
            'status': 'success',
            'race_info': race_info,
            'horses': horses,
            'predictions': predictions,
            'top_pick': predictions[0] if predictions else None,
            'value_bets': [p for p in predictions if p['is_value_bet']]
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions for custom horse data"""
    try:
        data = request.json
        predictions = ai_system.predict_race(data['horses'])
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'race_analysis': {
                'favorite': predictions[0] if predictions else None,
                'longshot_value': next((p for p in predictions if p['odds'] > 10 and p['is_value_bet']), None),
                'expected_value': sum(p['predicted_win_probability'] * p['odds'] for p in predictions) / len(predictions) if predictions else 0
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    if ai_system.data is not None:
        stats = {
            'total_races': ai_system.data['race_id'].nunique(),
            'total_horses': len(ai_system.data),
            'win_rate': (ai_system.data['finish_position'] == 1).mean(),
            'avg_odds': ai_system.data['odds'].mean(),
            'model_accuracy': ai_system.model['accuracy'] if ai_system.model else 0
        }
        return jsonify({'status': 'success', 'stats': stats})
    else:
        return jsonify({'status': 'error', 'message': 'No data available'})

@app.route('/api/simulate_bet', methods=['POST'])
def simulate_bet():
    """Simulate a betting strategy"""
    try:
        data = request.json
        bankroll = float(data.get('bankroll', 1000))
        bet_amount = float(data.get('bet_amount', 100))
        strategy = data.get('strategy', 'top_pick')
        
        # Get current predictions
        if not current_predictions:
            return jsonify({'status': 'error', 'message': 'No predictions available'})
        
        # Simulate different betting strategies
        if strategy == 'top_pick':
            pick = current_predictions[0]
            odds = pick['odds']
            prob = pick['predicted_win_probability']
        elif strategy == 'value_bets':
            value_bets = [p for p in current_predictions if p['is_value_bet']]
            pick = value_bets[0] if value_bets else current_predictions[0]
            odds = pick['odds']
            prob = pick['predicted_win_probability']
        else:
            pick = current_predictions[np.random.randint(0, len(current_predictions))]
            odds = pick['odds']
            prob = pick['predicted_win_probability']
        
        # Calculate expected value
        expected_value = (prob * (bet_amount * (odds - 1))) - ((1 - prob) * bet_amount)
        
        return jsonify({
            'status': 'success',
            'bet_simulation': {
                'strategy': strategy,
                'selected_horse': pick['horse_name'],
                'odds': odds,
                'bet_amount': bet_amount,
                'potential_payout': bet_amount * odds,
                'win_probability': prob,
                'expected_value': round(expected_value, 2),
                'bankroll_after_win': bankroll + (bet_amount * (odds - 1)),
                'bankroll_after_loss': bankroll - bet_amount
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("AI Horse Racing Predictor System")
    print("=" * 50)
    print("Starting server on http://localhost:5000")
    print("Initializing AI system...")
    
    # Initialize the AI system
    ai_system.generate_sample_data()
    print(f"Generated {len(ai_system.data)} data samples")
    
    app.run(debug=True, port=5000)
