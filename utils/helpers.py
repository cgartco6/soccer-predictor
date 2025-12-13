import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import math
import re

def format_odds(odds_value: Union[float, str, None]) -> Optional[float]:
    """
    Format odds to standard decimal format
    
    Args:
        odds_value: Raw odds value (could be string, float, or fraction)
    
    Returns:
        Formatted decimal odds or None if invalid
    """
    if odds_value is None:
        return None
    
    try:
        # Handle string input
        if isinstance(odds_value, str):
            odds_str = odds_value.strip()
            
            # Check for fractional odds (e.g., "5/2")
            if '/' in odds_str:
                numerator, denominator = odds_str.split('/')
                return float(numerator) / float(denominator) + 1
            
            # Remove any non-numeric characters except decimal point
            odds_str = re.sub(r'[^\d.]', '', odds_str)
            
            if not odds_str:
                return None
            
            odds_float = float(odds_str)
            
            # Ensure reasonable odds range
            if 1.0 < odds_float < 1000:
                return odds_float
            else:
                return None
        
        # Handle float input
        elif isinstance(odds_value, (int, float)):
            if 1.0 < odds_value < 1000:
                return float(odds_value)
            else:
                return None
        
        return None
        
    except (ValueError, TypeError, ZeroDivisionError):
        return None

def calculate_implied_probability(odds: float) -> float:
    """
    Calculate implied probability from decimal odds
    
    Args:
        odds: Decimal odds
    
    Returns:
        Implied probability (0-1)
    """
    if odds is None or odds <= 1:
        return 0.0
    
    probability = 1 / odds
    
    # Add overround/margin adjustment
    # Typically bookmaker margin is 2-10%
    adjusted_probability = probability * 0.95  # Assume 5% margin
    
    return min(max(adjusted_probability, 0.0), 1.0)

def calculate_kelly_criterion(probability: float, odds: float, 
                             bankroll_fraction: float = 0.25) -> float:
    """
    Calculate Kelly Criterion stake
    
    Args:
        probability: True probability of outcome (0-1)
        odds: Decimal odds
        bankroll_fraction: Fraction of bankroll to risk (default 0.25 = quarter Kelly)
    
    Returns:
        Fraction of bankroll to stake
    """
    if odds <= 1 or probability <= 0 or probability >= 1:
        return 0.0
    
    # Full Kelly formula: f* = (p * b - q) / b
    # where b = odds - 1, p = probability, q = 1 - p
    b = odds - 1
    p = probability
    q = 1 - p
    
    full_kelly = (p * b - q) / b
    
    # Apply fraction and ensure non-negative
    kelly_fraction = max(full_kelly * bankroll_fraction, 0.0)
    
    return min(kelly_fraction, 1.0)  # Cap at 100% of bankroll

def calculate_expected_value(probability: float, odds: float, stake: float = 1.0) -> float:
    """
    Calculate expected value of a bet
    
    Args:
        probability: True probability of outcome
        odds: Decimal odds
        stake: Bet stake
    
    Returns:
        Expected value
    """
    win_return = (odds - 1) * stake
    ev = (probability * win_return) - ((1 - probability) * stake)
    return ev

def calculate_value(probability: float, odds: float) -> float:
    """
    Calculate value (probability edge)
    
    Args:
        probability: True probability
        odds: Decimal odds
    
    Returns:
        Value ratio (>1 indicates positive value)
    """
    implied_prob = calculate_implied_probability(odds)
    
    if implied_prob > 0:
        return probability / implied_prob
    else:
        return 0.0

def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format float as percentage string
    
    Args:
        value: Float value (0-1)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    """
    percentage = value * 100
    return f"{percentage:.{decimals}f}%"

def calculate_poisson_probability(lambda_val: float, k: int) -> float:
    """
    Calculate Poisson probability
    
    Args:
        lambda_val: Expected goals
        k: Number of goals
    
    Returns:
        Probability
    """
    return (math.exp(-lambda_val) * (lambda_val ** k)) / math.factorial(k)

def calculate_btts_probability(home_goals_exp: float, away_goals_exp: float) -> float:
    """
    Calculate Both Teams to Score probability
    
    Args:
        home_goals_exp: Expected home goals
        away_goals_exp: Expected away goals
    
    Returns:
        BTTS probability (0-1)
    """
    # Probability that home team scores
    prob_home_scores = 1 - math.exp(-home_goals_exp)
    
    # Probability that away team scores
    prob_away_scores = 1 - math.exp(-away_goals_exp)
    
    # Probability both score
    prob_btts = prob_home_scores * prob_away_scores
    
    return prob_btts

def calculate_correct_score_probability(home_goals_exp: float, away_goals_exp: float, 
                                       home_goals: int, away_goals: int) -> float:
    """
    Calculate correct score probability
    
    Args:
        home_goals_exp: Expected home goals
        away_goals_exp: Expected away goals
        home_goals: Actual home goals
        away_goals: Actual away goals
    
    Returns:
        Correct score probability
    """
    prob_home = calculate_poisson_probability(home_goals_exp, home_goals)
    prob_away = calculate_poisson_probability(away_goals_exp, away_goals)
    
    return prob_home * prob_away

def normalize_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features to 0-1 range
    
    Args:
        features: DataFrame of features
    
    Returns:
        Normalized features
    """
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(features)
    
    return pd.DataFrame(normalized, columns=features.columns)

def calculate_confidence_interval(predictions: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for predictions
    
    Args:
        predictions: Array of prediction probabilities
        confidence_level: Confidence level (0-1)
    
    Returns:
        Lower and upper bounds of confidence interval
    """
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    
    # Z-score for confidence level
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }
    
    z = z_scores.get(confidence_level, 1.96)
    
    margin_error = z * (std_pred / np.sqrt(len(predictions)))
    
    lower_bound = max(0, mean_pred - margin_error)
    upper_bound = min(1, mean_pred + margin_error)
    
    return lower_bound, upper_bound

def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        RMSE
    """
    return np.sqrt(np.mean((actual - predicted) ** 2))

def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        MAE
    """
    return np.mean(np.abs(actual - predicted))

def calculate_r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate R-squared
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        R-squared
    """
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for zero denominator
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
    
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator

def format_timestamp(timestamp: Union[str, datetime]) -> str:
    """
    Format timestamp to readable string
    
    Args:
        timestamp: Timestamp string or datetime object
    
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            return timestamp
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        return str(timestamp)
    
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def calculate_time_until_match(match_time: Union[str, datetime]) -> str:
    """
    Calculate time until match starts
    
    Args:
        match_time: Match time
    
    Returns:
        Formatted time until match
    """
    if isinstance(match_time, str):
        try:
            match_dt = datetime.fromisoformat(match_time.replace('Z', '+00:00'))
        except ValueError:
            return "Unknown"
    elif isinstance(match_time, datetime):
        match_dt = match_time
    else:
        return "Unknown"
    
    now = datetime.now()
    
    if match_dt < now:
        return "Started"
    
    time_diff = match_dt - now
    
    if time_diff.days > 0:
        return f"{time_diff.days}d {time_diff.seconds // 3600}h"
    elif time_diff.seconds >= 3600:
        hours = time_diff.seconds // 3600
        minutes = (time_diff.seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    else:
        minutes = time_diff.seconds // 60
        return f"{minutes}m"
