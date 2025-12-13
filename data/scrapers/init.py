"""
Scrapers module for collecting soccer data
"""
from .base_scraper import BaseScraper
from .hollywoodbets import HollywoodbetsScraper
from .betway import BetwayScraper
from .football_data import FootballDataScraper

__all__ = [
    'BaseScraper',
    'HollywoodbetsScraper',
    'BetwayScraper',
    'FootballDataScraper'
]
