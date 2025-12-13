"""
Database module for soccer prediction system
"""
from .models import (
    Base, Team, Player, Coach, Match, Injury,
    TeamStatistics, Prediction, ModelPerformance
)
from .manager import DatabaseManager

__all__ = [
    'Base',
    'Team',
    'Player',
    'Coach',
    'Match',
    'Injury',
    'TeamStatistics',
    'Prediction',
    'ModelPerformance',
    'DatabaseManager'
]
