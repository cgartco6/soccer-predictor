import re
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd

def validate_match_data(match_data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate match data
    
    Args:
        match_data: Dictionary containing match data
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields
    required_fields = ['home_team', 'away_team', 'match_date']
    
    for field in required_fields:
        if field not in match_data or not match_data[field]:
            errors.append(f"Missing required field: {field}")
    
    # Validate team names
    if 'home_team' in match_data:
        home_team = match_data['home_team']
        if not isinstance(home_team, str) or len(home_team.strip()) < 2:
            errors.append("Invalid home team name")
    
    if 'away_team' in match_data:
        away_team = match_data['away_team']
        if not isinstance(away_team, str) or len(away_team.strip()) < 2:
            errors.append("Invalid away team name")
    
    # Validate teams are different
    if ('home_team' in match_data and 'away_team' in match_data and
        match_data['home_team'] == match_data['away_team']):
        errors.append("Home and away teams cannot be the same")
    
    # Validate match date
    if 'match_date' in match_data:
        try:
            if isinstance(match_data['match_date'], str):
                datetime.fromisoformat(match_data['match_date'].replace('Z', '+00:00'))
            elif not isinstance(match_data['match_date'], datetime):
                errors.append("Invalid match date format")
        except (ValueError, TypeError):
            errors.append("Invalid match date format")
    
    # Validate odds
    odds_fields = ['home_odds', 'draw_odds', 'away_odds', 'btts_yes_odds', 'btts_no_odds']
    
    for field in odds_fields:
        if field in match_data and match_data[field] is not None:
            try:
                odds = float(match_data[field])
                if odds < 1.0 or odds > 1000:
                    errors.append(f"Invalid {field}: {odds} (must be between 1.0 and 1000)")
            except (ValueError, TypeError):
                errors.append(f"Invalid {field} format")
    
    # Validate scores (if present)
    score_fields = ['home_score', 'away_score']
    
    for field in score_fields:
        if field in match_data and match_data[field] is not None:
            try:
                score = int(match_data[field])
                if score < 0 or score > 50:  # Reasonable score limit
                    errors.append(f"Invalid {field}: {score}")
            except (ValueError, TypeError):
                errors.append(f"Invalid {field} format")
    
    # Validate result (if present)
    if 'result' in match_data and match_data['result'] is not None:
        result = str(match_data['result']).upper()
        if result not in ['H', 'D', 'A', '1', 'X', '2', 'HOME', 'DRAW', 'AWAY']:
            errors.append(f"Invalid result: {match_data['result']}")
    
    # Validate BTTS result (if present)
    if 'btts_result' in match_data and match_data['btts_result'] is not None:
        btts = match_data['btts_result']
        if not isinstance(btts, (bool, int)):
            errors.append("Invalid btts_result format (must be boolean or 0/1)")
    
    # Validate league (if present)
    if 'league' in match_data and match_data['league'] is not None:
        league = str(match_data['league'])
        if len(league.strip()) < 2:
            errors.append("Invalid league name")
    
    return len(errors) == 0, errors

def validate_team_data(team_data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate team data
    
    Args:
        team_data: Dictionary containing team data
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required field
    if 'name' not in team_data or not team_data['name']:
        errors.append("Missing team name")
    else:
        name = team_data['name']
        if not isinstance(name, str) or len(name.strip()) < 2:
            errors.append("Invalid team name")
    
    # Validate optional fields
    if 'country' in team_data and team_data['country'] is not None:
        country = str(team_data['country'])
        if len(country.strip()) < 2:
            errors.append("Invalid country name")
    
    if 'stadium' in team_data and team_data['stadium'] is not None:
        stadium = str(team_data['stadium'])
        if len(stadium.strip()) < 2:
            errors.append("Invalid stadium name")
    
    if 'capacity' in team_data and team_data['capacity'] is not None:
        try:
            capacity = int(team_data['capacity'])
            if capacity < 0 or capacity > 200000:  # Reasonable capacity limit
                errors.append(f"Invalid capacity: {capacity}")
        except (ValueError, TypeError):
            errors.append("Invalid capacity format")
    
    return len(errors) == 0, errors

def validate_prediction_data(prediction_data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate prediction data
    
    Args:
        prediction_data: Dictionary containing prediction data
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields
    required_fields = ['match_id', 'predicted_result', 'predicted_btts']
    
    for field in required_fields:
        if field not in prediction_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate match_id
    if 'match_id' in prediction_data:
        try:
            match_id = int(prediction_data['match_id'])
            if match_id <= 0:
                errors.append("Invalid match_id (must be positive)")
        except (ValueError, TypeError):
            errors.append("Invalid match_id format")
    
    # Validate predicted_result
    if 'predicted_result' in prediction_data:
        result = str(prediction_data['predicted_result']).upper()
        if result not in ['H', 'D', 'A', '1', 'X', '2']:
            errors.append(f"Invalid predicted_result: {prediction_data['predicted_result']}")
    
    # Validate predicted_btts
    if 'predicted_btts' in prediction_data:
        btts = prediction_data['predicted_btts']
        if not isinstance(btts, bool):
            errors.append("predicted_btts must be boolean")
    
    # Validate probabilities (if present)
    probability_fields = ['home_win_prob', 'draw_prob', 'away_win_prob', 
                         'btts_yes_prob', 'btts_no_prob']
    
    for field in probability_fields:
        if field in prediction_data and prediction_data[field] is not None:
            try:
                prob = float(prediction_data[field])
                if prob < 0 or prob > 1:
                    errors.append(f"Invalid {field}: {prob} (must be between 0 and 1)")
            except (ValueError, TypeError):
                errors.append(f"Invalid {field} format")
    
    # Validate confidence_score (if present)
    if 'confidence_score' in prediction_data and prediction_data['confidence_score'] is not None:
        try:
            confidence = float(prediction_data['confidence_score'])
            if confidence < 0 or confidence > 1:
                errors.append(f"Invalid confidence_score: {confidence} (must be between 0 and 1)")
        except (ValueError, TypeError):
            errors.append("Invalid confidence_score format")
    
    return len(errors) == 0, errors

def validate_odds_data(odds_data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate odds data
    
    Args:
        odds_data: Dictionary containing odds data
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for at least one odds field
    odds_fields = ['home_win', 'draw', 'away_win', 'btts_yes', 'btts_no']
    has_odds = any(field in odds_data and odds_data[field] is not None 
                   for field in odds_fields)
    
    if not has_odds:
        errors.append("No odds data provided")
    
    # Validate individual odds
    for field in odds_fields:
        if field in odds_data and odds_data[field] is not None:
            try:
                odds = float(odds_data[field])
                if odds < 1.0 or odds > 1000:
                    errors.append(f"Invalid {field} odds: {odds}")
            except (ValueError, TypeError):
                errors.append(f"Invalid {field} odds format")
    
    # Validate odds consistency (if all three result odds are present)
    if all(field in odds_data and odds_data[field] is not None 
           for field in ['home_win', 'draw', 'away_win']):
        total_margin = sum(1/odds_data[field] for field in ['home_win', 'draw', 'away_win'])
        
        if total_margin > 1.15:  # Allow up to 15% margin
            errors.append(f"Excessive bookmaker margin: {total_margin:.2%}")
    
    return len(errors) == 0, errors

def validate_weather_data(weather_data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate weather data
    
    Args:
        weather_data: Dictionary containing weather data
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Validate temperature
    if 'temperature' in weather_data and weather_data['temperature'] is not None:
        try:
            temp = float(weather_data['temperature'])
            if temp < -50 or temp > 60:  # Reasonable temperature range
                errors.append(f"Invalid temperature: {temp}°C")
        except (ValueError, TypeError):
            errors.append("Invalid temperature format")
    
    # Validate humidity
    if 'humidity' in weather_data and weather_data['humidity'] is not None:
        try:
            humidity = float(weather_data['humidity'])
            if humidity < 0 or humidity > 100:
                errors.append(f"Invalid humidity: {humidity}%")
        except (ValueError, TypeError):
            errors.append("Invalid humidity format")
    
    # Validate wind speed
    if 'wind_speed' in weather_data and weather_data['wind_speed'] is not None:
        try:
            wind = float(weather_data['wind_speed'])
            if wind < 0 or wind > 200:  # Reasonable wind speed limit
                errors.append(f"Invalid wind speed: {wind} km/h")
        except (ValueError, TypeError):
            errors.append("Invalid wind speed format")
    
    # Validate precipitation
    if 'precipitation' in weather_data and weather_data['precipitation'] is not None:
        try:
            precip = float(weather_data['precipitation'])
            if precip < 0 or precip > 500:  # Reasonable precipitation limit
                errors.append(f"Invalid precipitation: {precipitation} mm")
        except (ValueError, TypeError):
            errors.append("Invalid precipitation format")
    
    return len(errors) == 0, errors

def validate_features_data(features_data: Union[Dict, pd.DataFrame]) -> Tuple[bool, List[str]]:
    """
    Validate features data
    
    Args:
        features_data: Dictionary or DataFrame containing features
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if isinstance(features_data, pd.DataFrame):
        # Check for required columns
        required_columns = ['home_form_last_5', 'away_form_last_5']
        
        for col in required_columns:
            if col not in features_data.columns:
                errors.append(f"Missing required feature column: {col}")
        
        # Check for NaN values
        if features_data.isna().any().any():
            nan_columns = features_data.columns[features_data.isna().any()].tolist()
            errors.append(f"NaN values found in columns: {', '.join(nan_columns)}")
        
        # Check for infinite values
        numeric_df = features_data.select_dtypes(include=[np.number])
        if np.any(np.isinf(numeric_df.values)):
            errors.append("Infinite values found in features")
    
    elif isinstance(features_data, dict):
        # Check for required features
        required_features = ['home_form_last_5', 'away_form_last_5']
        
        for feat in required_features:
            if feat not in features_data:
                errors.append(f"Missing required feature: {feat}")
        
        # Validate feature values
        for key, value in features_data.items():
            if value is None:
                errors.append(f"Feature {key} is None")
            elif isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    errors.append(f"Invalid value for feature {key}: {value}")
    
    return len(errors) == 0, errors

def sanitize_string(input_string: str) -> str:
    """
    Sanitize string input
    
    Args:
        input_string: Raw input string
    
    Returns:
        Sanitized string
    """
    if not isinstance(input_string, str):
        return ""
    
    # Remove excessive whitespace
    sanitized = ' '.join(input_string.strip().split())
    
    # Remove potentially dangerous characters (for SQL injection prevention)
    sanitized = re.sub(r'[;\'"\\]', '', sanitized)
    
    return sanitized

def validate_email(email: str) -> bool:
    """
    Validate email address format
    
    Args:
        email: Email address
    
    Returns:
        True if valid, False otherwise
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format
    
    Args:
        api_key: API key
    
    Returns:
        True if valid, False otherwise
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # Check length and format
    if len(api_key) < 20 or len(api_key) > 100:
        return False
    
    # Should contain alphanumeric characters and dashes
    if not re.match(r'^[a-zA-Z0-9\-_]+$', api_key):
        return False
    
    return True
