import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import re

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process and clean soccer data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.feature_selector = None
        self.feature_columns = []
        self.categorical_columns = ['league', 'venue', 'pitch_condition', 'referee']
        self.numerical_columns = []
        
    def process_raw_match_data(self, raw_data: List[Dict]) -> pd.DataFrame:
        """Process raw match data into clean DataFrame"""
        processed_matches = []
        
        for match in raw_data:
            try:
                processed_match = self._process_single_match(match)
                if processed_match:
                    processed_matches.append(processed_match)
            except Exception as e:
                logger.error(f"Error processing match {match.get('home_team', 'Unknown')}: {e}")
                continue
        
        if processed_matches:
            df = pd.DataFrame(processed_matches)
            logger.info(f"Processed {len(df)} matches")
            return df
        else:
            return pd.DataFrame()
    
    def _process_single_match(self, match: Dict) -> Optional[Dict]:
        """Process single match data"""
        try:
            processed = {}
            
            # Basic match info
            processed['home_team'] = match.get('home_team', '').strip()
            processed['away_team'] = match.get('away_team', '').strip()
            processed['league'] = match.get('league', 'Unknown').strip()
            processed['match_date'] = self._parse_date(match.get('match_time'))
            
            # Odds processing
            odds = match.get('odds', {})
            processed['home_odds'] = self._safe_float(odds.get('home_win'))
            processed['draw_odds'] = self._safe_float(odds.get('draw'))
            processed['away_odds'] = self._safe_float(odds.get('away_win'))
            processed['btts_yes_odds'] = self._safe_float(odds.get('btts_yes'))
            processed['btts_no_odds'] = self._safe_float(odds.get('btts_no'))
            
            # Calculate implied probabilities
            if processed['home_odds']:
                processed['implied_prob_home'] = 1 / processed['home_odds']
            if processed['draw_odds']:
                processed['implied_prob_draw'] = 1 / processed['draw_odds']
            if processed['away_odds']:
                processed['implied_prob_away'] = 1 / processed['away_odds']
            
            # Source info
            processed['source'] = match.get('source', 'unknown')
            processed['scraped_at'] = match.get('scraped_at', datetime.now().isoformat())
            
            # Additional data if available
            if 'weather' in match:
                processed.update(self._process_weather_data(match['weather']))
            
            if 'pitch_condition' in match:
                processed['pitch_condition'] = match['pitch_condition']
            
            if 'referee' in match:
                processed['referee'] = match['referee']
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in _process_single_match: {e}")
            return None
    
    def process_training_data(self, historical_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Process historical data for training"""
        try:
            # Clean the data
            df_clean = self._clean_data(historical_data)
            
            # Feature engineering
            df_features = self._engineer_features(df_clean)
            
            # Handle missing values
            df_filled = self._handle_missing_values(df_features)
            
            # Encode categorical variables
            df_encoded = self._encode_categorical(df_filled)
            
            # Prepare labels
            y_result = self._prepare_result_labels(df_clean)
            y_btts = self._prepare_btts_labels(df_clean)
            
            # Feature selection
            X_selected = self._select_features(df_encoded, y_result)
            
            self.numerical_columns = X_selected.columns.tolist()
            
            logger.info(f"Processed training data: {X_selected.shape} features, {len(y_result)} samples")
            return X_selected, y_result, y_btts
            
        except Exception as e:
            logger.error(f"Error processing training data: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data"""
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Handle date columns
        if 'match_date' in df_clean.columns:
            df_clean['match_date'] = pd.to_datetime(df_clean['match_date'], errors='coerce')
        
        # Clean team names
        if 'home_team' in df_clean.columns:
            df_clean['home_team'] = df_clean['home_team'].str.strip()
        if 'away_team' in df_clean.columns:
            df_clean['away_team'] = df_clean['away_team'].str.strip()
        
        # Clean odds columns
        odds_columns = ['home_odds', 'draw_odds', 'away_odds', 'btts_yes_odds', 'btts_no_odds']
        for col in odds_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                # Remove extreme outliers
                if df_clean[col].notna().any():
                    q1 = df_clean[col].quantile(0.01)
                    q3 = df_clean[col].quantile(0.99)
                    df_clean[col] = df_clean[col].clip(lower=q1, upper=q3)
        
        # Clean score columns
        if 'home_score' in df_clean.columns:
            df_clean['home_score'] = pd.to_numeric(df_clean['home_score'], errors='coerce').fillna(0)
        if 'away_score' in df_clean.columns:
            df_clean['away_score'] = pd.to_numeric(df_clean['away_score'], errors='coerce').fillna(0)
        
        # Remove rows with missing critical data
        critical_columns = ['home_team', 'away_team', 'result']
        df_clean = df_clean.dropna(subset=[col for col in critical_columns if col in df_clean.columns])
        
        return df_clean
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from cleaned data"""
        features = pd.DataFrame()
        
        # Basic features
        features['is_home'] = 1  # Always 1 for home team perspective
        
        # Team performance features (would be populated from historical data)
        # For now, create placeholder columns
        performance_features = [
            'home_form_last_5', 'away_form_last_5',
            'home_avg_goals_scored', 'away_avg_goals_scored',
            'home_avg_goals_conceded', 'away_avg_goals_conceded',
            'home_home_strength', 'away_away_strength'
        ]
        
        for feat in performance_features:
            features[feat] = 0.5  # Default value
        
        # Odds-based features
        if 'home_odds' in df.columns and 'draw_odds' in df.columns and 'away_odds' in df.columns:
            features['implied_prob_home'] = 1 / df['home_odds']
            features['implied_prob_draw'] = 1 / df['draw_odds']
            features['implied_prob_away'] = 1 / df['away_odds']
            
            # Value metrics
            features['odds_variance'] = df[['home_odds', 'draw_odds', 'away_odds']].var(axis=1)
            features['max_odds'] = df[['home_odds', 'draw_odds', 'away_odds']].max(axis=1)
            features['min_odds'] = df[['home_odds', 'draw_odds', 'away_odds']].min(axis=1)
        
        # BTTS odds features
        if 'btts_yes_odds' in df.columns and 'btts_no_odds' in df.columns:
            features['btts_implied_prob_yes'] = 1 / df['btts_yes_odds']
            features['btts_implied_prob_no'] = 1 / df['btts_no_odds']
            features['btts_odds_ratio'] = df['btts_yes_odds'] / df['btts_no_odds']
        
        # Time-based features
        if 'match_date' in df.columns:
            features['day_of_week'] = df['match_date'].dt.dayofweek
            features['month'] = df['match_date'].dt.month
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
            features['is_night_match'] = ((df['match_date'].dt.hour >= 18) | (df['match_date'].dt.hour <= 6)).astype(int)
        
        # League features
        if 'league' in df.columns:
            # One-hot encode league (will be handled separately)
            pass
        
        # Derived features
        features['goal_expectancy_home'] = features['home_avg_goals_scored'] * features['away_avg_goals_conceded']
        features['goal_expectancy_away'] = features['away_avg_goals_scored'] * features['home_avg_goals_conceded']
        features['total_goal_expectancy'] = features['goal_expectancy_home'] + features['goal_expectancy_away']
        
        # BTTS probability estimate
        features['btts_probability'] = (1 - np.exp(-features['goal_expectancy_home'])) * (1 - np.exp(-features['goal_expectancy_away']))
        
        # Strength difference
        features['strength_difference'] = features['home_form_last_5'] - features['away_form_last_5']
        
        return features
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        df_filled = df.copy()
        
        # Fill numerical columns
        numerical_cols = df_filled.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_filled[col].isna().any():
                df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        
        # Fill categorical columns
        categorical_cols = df_filled.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_filled[col].isna().any():
                df_filled[col] = df_filled[col].fillna('Unknown')
        
        return df_filled
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                # Fit on all possible values
                unique_values = df_encoded[col].unique()
                self.label_encoders[col].fit(unique_values)
            
            df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        return df_encoded
    
    def _prepare_result_labels(self, df: pd.DataFrame) -> pd.Series:
        """Prepare result labels (H/D/A)"""
        if 'result' in df.columns:
            # Map results to numerical values
            result_mapping = {'H': 0, 'D': 1, 'A': 2}
            y_result = df['result'].map(result_mapping)
            return y_result
        else:
            raise ValueError("Result column not found in data")
    
    def _prepare_btts_labels(self, df: pd.DataFrame) -> pd.Series:
        """Prepare BTTS labels"""
        if 'btts_result' in df.columns:
            return df['btts_result'].astype(int)
        elif 'home_score' in df.columns and 'away_score' in df.columns:
            # Calculate BTTS from scores
            return ((df['home_score'] > 0) & (df['away_score'] > 0)).astype(int)
        else:
            raise ValueError("Cannot prepare BTTS labels")
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select best features"""
        if len(X.columns) > 50:
            self.feature_selector = SelectKBest(score_func=f_classif, k=30)
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_indices = self.feature_selector.get_support(indices=True)
            selected_columns = X.columns[selected_indices]
            X_selected = pd.DataFrame(X_selected, columns=selected_columns)
        else:
            X_selected = X.copy()
        
        return X_selected
    
    def fit_scalers(self, X: pd.DataFrame):
        """Fit scalers on training data"""
        self.scaler.fit(X)
        self.min_max_scaler.fit(X)
    
    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scalers"""
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def inverse_transform(self, X_scaled: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled features"""
        X_original = self.scaler.inverse_transform(X_scaled)
        return pd.DataFrame(X_original, columns=X_scaled.columns)
    
    # Helper methods
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string"""
        if not date_str:
            return datetime.now()
        
        try:
            # Try common formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d/%m/%Y %H:%M', '%H:%M']:
                try:
                    if fmt == '%H:%M':
                        # Combine with today's date
                        time_only = datetime.strptime(date_str, fmt).time()
                        return datetime.combine(datetime.now().date(), time_only)
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # If all formats fail, return current date
            return datetime.now()
            
        except Exception:
            return datetime.now()
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert to float"""
        if value is None:
            return None
        
        try:
            if isinstance(value, str):
                # Remove non-numeric characters
                value = re.sub(r'[^\d.]', '', value)
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _process_weather_data(self, weather_data: Dict) -> Dict:
        """Process weather data"""
        processed = {}
        
        if isinstance(weather_data, dict):
            processed['temperature'] = weather_data.get('temperature')
            processed['humidity'] = weather_data.get('humidity')
            processed['wind_speed'] = weather_data.get('wind_speed')
            processed['precipitation'] = weather_data.get('precipitation')
            
            # Calculate weather impact score
            impact_score = 0.5  # Default
            
            if 'temperature' in weather_data:
                temp = weather_data['temperature']
                if temp < 0 or temp > 30:
                    impact_score += 0.2
                elif 10 <= temp <= 25:
                    impact_score -= 0.1
            
            if 'precipitation' in weather_data and weather_data['precipitation'] > 0:
                impact_score += 0.1
            
            if 'wind_speed' in weather_data and weather_data['wind_speed'] > 20:
                impact_score += 0.1
            
            processed['weather_impact'] = min(max(impact_score, 0), 1)
        
        return processed
