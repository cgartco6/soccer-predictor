"""
Data module for soccer prediction system
"""
from .database.manager import DatabaseManager
from .scrapers.hollywoodbets import HollywoodbetsScraper
from .scrapers.betway import BetwayScraper
from .scrapers.football_data import FootballDataScraper
from .processing.data_processor import DataProcessor

__all__ = [
    'DatabaseManager',
    'HollywoodbetsScraper',
    'BetwayScraper',
    'FootballDataScraper',
    'DataProcessor'
]
