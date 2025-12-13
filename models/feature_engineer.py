import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.feature_columns = []
        
    def create_features(self, match_data: Dict, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features for prediction
        """
        features = {}
        
        # Basic match features
        features['is_home'] = 1
        features['match_importance'] = self._calculate_match_importance(match_data)
        
        # Team performance features
        home_stats = self._get_team_stats(match_data['home_team'], historical_data)
        away_stats = self._get_team_stats(match_data['away_team'], historical_data)
        
        # Form features
        features['home_form_last_5'] = self._calculate_form(home_stats, 5)
        features['away_form_last_5'] = self._calculate_form(away_stats, 5)
        features['home_form_last_10'] = self._calculate_form(home_stats, 10)
        features['away_form_last_10'] = self._calculate_form(away_stats, 10)
        
        # Goal-related features
        features['home_avg_goals_scored'] = home_stats.get('avg_goals_scored', 1.5)
        features['home_avg_goals_conceded'] = home_stats.get('avg_goals_conceded', 1.2)
        features['away_avg_goals_scored'] = away_stats.get('avg_goals_scored', 1.3)
        features['away_avg_goals_conceded'] = away_stats.get('avg_goals_conceded', 1.4)
        
        # Home/Away specific stats
        features['home_home_strength'] = self._calculate_home_strength(home_stats)
        features['away_away_strength'] = self._calculate_away_strength(away_stats)
        
        # Head-to-head features
        h2h_stats = self._get_head_to_head_stats(
            match_data['home_team'], 
            match_data['away_team'], 
            historical_data
        )
        features['h2h_home_wins'] = h2h_stats.get('home_wins', 0)
        features['h2h_draws'] = h2h_stats.get('draws', 0)
        features['h2h_away_wins'] = h2h_stats.get('away_wins', 0)
        features['h2h_avg_goals'] = h2h_stats.get('avg_goals', 2.5)
        
        # Player features
        features['home_key_players_available'] = self._get_key_players_available(match_data, 'home')
        features['away_key_players_available'] = self._get_key_players_available(match_data, 'away')
        features['home_injury_impact'] = self._calculate_injury_impact(match_data, 'home')
        features['away_injury_impact'] = self._calculate_injury_impact(match_data, 'away')
        
        # Coach features
        features['home_coach_win_rate'] = match_data.get('home_coach_win_rate', 0.5)
        features['away_coach_win_rate'] = match_data.get('away_coach_win_rate', 0.5)
        features['coach_h2h_record'] = self._get_coach_h2h_record(match_data)
        
        # External factors
        features['weather_impact'] = self._calculate_weather_impact(match_data.get('weather', {}))
        features['pitch_condition_score'] = self._get_pitch_condition_score(match_data.get('pitch_condition', 'good'))
        features['crowd_factor'] = self._calculate_crowd_factor(match_data)
        features['travel_distance_km'] = self._calculate_travel_distance(match_data)
        
        # Referee features
        features['referee_home_bias'] = self._get_referee_bias(match_data.get('referee', ''), 'home')
        features['referee_avg_cards'] = match_data.get('referee_avg_cards', 3.5)
        features['referee_avg_penalties'] = match_data.get('referee_avg_penalties', 0.2)
        
        # Odds-based features
        odds = match_data.get('odds', {})
        features['implied_prob_home'] = 1 / odds.get('home_win', 2.5) if odds.get('home_win') else 0.4
        features['implied_prob_draw'] = 1 / odds.get('draw', 3.2) if odds.get('draw') else 0.3
        features['implied_prob_away'] = 1 / odds.get('away_win', 2.8) if odds.get('away_win') else 0.3
        features['odds_volatility'] = self._calculate_odds_volatility(odds)
        
        # Time-based features
        features['days_since_last_match_home'] = self._get_days_since_last_match(match_data, 'home')
        features['days_since_last_match_away'] = self._get_days_since_last_match(match_data, 'away')
        features['is_weekend'] = 1 if match_data.get('match_date', datetime.now()).weekday() >= 5 else 0
        
        # League features
        features['league_avg_goals'] = self._get_league_avg_goals(match_data.get('league', ''), historical_data)
        features['league_home_advantage'] = self._get_league_home_advantage(match_data.get('league', ''), historical_data)
        
        # Create derived features
        features['goal_expectancy_home'] = (
            features['home_avg_goals_scored'] * 
            features['away_avg_goals_conceded'] * 
            features['home_home_strength']
        )
        
        features['goal_expectancy_away'] = (
            features['away_avg_goals_scored'] * 
            features['home_avg_goals_conceded'] * 
            features['away_away_strength']
        )
        
        features['strength_difference'] = (
            features['home_form_last_5'] - features['away_form_last_5']
        )
        
        # BTTS specific features
        features['btts_probability'] = self._calculate_btts_probability(
            features['home_avg_goals_scored'],
            features['away_avg_goals_scored'],
            features['home_avg_goals_conceded'],
            features['away_avg_goals_conceded']
        )
        
        return pd.DataFrame([features])
    
    def _calculate_form(self, team_stats: Dict, matches: int) -> float:
        """Calculate team form over last N matches"""
        if 'recent_results' in team_stats and len(team_stats['recent_results']) >= matches:
            results = team_stats['recent_results'][:matches]
            points = sum([3 if r == 'W' else 1 if r == 'D' else 0 for r in results])
            return points / (matches * 3)
        return 0.5
    
    def _calculate_home_strength(self, team_stats: Dict) -> float:
        """Calculate home strength"""
        home_record = team_stats.get('home_record', {})
        total = home_record.get('wins', 0) + home_record.get('draws', 0) + home_record.get('losses', 0)
        if total > 0:
            return (home_record.get('wins', 0) * 3 + home_record.get('draws', 0)) / (total * 3)
        return 0.5
    
    def _calculate_away_strength(self, team_stats: Dict) -> float:
        """Calculate away strength"""
        away_record = team_stats.get('away_record', {})
        total = away_record.get('wins', 0) + away_record.get('draws', 0) + away_record.get('losses', 0)
        if total > 0:
            return (away_record.get('wins', 0) * 3 + away_record.get('draws', 0)) / (total * 3)
        return 0.5
    
    def _get_head_to_head_stats(self, home_team: str, away_team: str, historical_data: pd.DataFrame) -> Dict:
        """Get head-to-head statistics"""
        h2h_matches = historical_data[
            ((historical_data['home_team'] == home_team) & (historical_data['away_team'] == away_team)) |
            ((historical_data['home_team'] == away_team) & (historical_data['away_team'] == home_team))
        ]
        
        stats = {
            'home_wins': 0,
            'draws': 0,
            'away_wins': 0,
            'avg_goals': 2.5
        }
        
        if not h2h_matches.empty:
            for _, match in h2h_matches.iterrows():
                if match['home_team'] == home_team:
                    if match['result'] == 'H':
                        stats['home_wins'] += 1
                    elif match['result'] == 'A':
                        stats['away_wins'] += 1
                    else:
                        stats['draws'] += 1
                else:
                    if match['result'] == 'A':
                        stats['home_wins'] += 1
                    elif match['result'] == 'H':
                        stats['away_wins'] += 1
                    else:
                        stats['draws'] += 1
            
            stats['avg_goals'] = h2h_matches['home_score'].mean() + h2h_matches['away_score'].mean()
            
        return stats
    
    def _calculate_injury_impact(self, match_data: Dict, team_type: str) -> float:
        """Calculate impact of injuries (0-1 scale, 1 = full strength)"""
        injuries = match_data.get(f'{team_type}_injuries', [])
        key_players = match_data.get(f'{team_type}_key_players', [])
        
        if not key_players:
            return 1.0
            
        injured_key_players = [p for p in injuries if p in key_players]
        impact = 1.0 - (len(injured_key_players) / len(key_players)) * 0.5
        return max(0.5, impact)
    
    def _calculate_btts_probability(self, home_scored: float, away_scored: float, 
                                  home_conceded: float, away_conceded: float) -> float:
        """Calculate probability of Both Teams to Score"""
        home_goal_expectancy = (home_scored + away_conceded) / 2
        away_goal_expectancy = (away_scored + home_conceded) / 2
        
        # Using Poisson distribution approximation
        prob_home_scores = 1 - np.exp(-home_goal_expectancy)
        prob_away_scores = 1 - np.exp(-away_goal_expectancy)
        
        return prob_home_scores * prob_away_scores
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform features"""
        # Handle missing values
        X_filled = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_filled)
        
        self.feature_columns = X.columns.tolist()
        
        return pd.DataFrame(X_scaled, columns=self.feature_columns)
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler"""
        X_filled = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_filled)
        return pd.DataFrame(X_scaled, columns=self.feature_columns)
