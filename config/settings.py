import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ScraperConfig:
    hollywoodbets_url: str = "https://www.hollywoodbets.net"
    betway_url: str = "https://www.betway.co.za"
    football_data_urls: Dict = field(default_factory=lambda: {
        'espn': 'https://www.espn.com/soccer',
        'whoscored': 'https://www.whoscored.com',
        'transfermarkt': 'https://www.transfermarkt.com',
        'sofascore': 'https://www.sofascore.com',
        'premierleague': 'https://www.premierleague.com'
    })
    request_timeout: int = 30
    retry_attempts: int = 3
    delay_between_requests: float = 2.0
    max_concurrent_requests: int = 5
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "soccer_predictor"
    user: str = "postgres"
    password: str = "password"
    pool_size: int = 20
    max_overflow: int = 30
    echo: bool = False
    pool_recycle: int = 3600
    
@dataclass
class ModelConfig:
    features: List[str] = field(default_factory=lambda: [
        'home_form_last_5', 'away_form_last_5', 'home_avg_goals_scored',
        'away_avg_goals_scored', 'home_avg_goals_conceded', 'away_avg_goals_conceded',
        'home_home_strength', 'away_away_strength', 'h2h_home_wins', 'h2h_draws',
        'h2h_away_wins', 'home_key_players_available', 'away_key_players_available',
        'home_injury_impact', 'away_injury_impact', 'home_coach_win_rate',
        'away_coach_win_rate', 'weather_impact', 'pitch_condition_score',
        'crowd_factor', 'referee_home_bias', 'referee_avg_cards',
        'implied_prob_home', 'implied_prob_draw', 'implied_prob_away',
        'days_since_last_match_home', 'days_since_last_match_away',
        'league_avg_goals', 'league_home_advantage', 'goal_expectancy_home',
        'goal_expectancy_away', 'strength_difference', 'btts_probability'
    ])
    target_columns: List[str] = field(default_factory=lambda: ['result', 'btts_result'])
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    model_type: str = "ensemble"
    ensemble_models: List[str] = field(default_factory=lambda: ['xgb', 'lgbm', 'rf', 'catboost'])
    retrain_frequency_days: int = 7
    min_training_samples: int = 100
    
@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    workers: int = 4
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key_header: str = "X-API-Key"
    rate_limit_per_minute: int = 60
    
@dataclass
class PredictionConfig:
    min_confidence: float = 0.65
    min_confidence_btts: float = 0.60
    value_threshold: float = 1.1
    max_recommendations_per_match: int = 3
    kelly_fraction: float = 0.25
    min_odds_value: float = 1.5
    
@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/soccer_predictor.log"
    max_size_mb: int = 100
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.scraper = ScraperConfig()
            self.database = DatabaseConfig()
            self.model = ModelConfig()
            self.api = APIConfig()
            self.prediction = PredictionConfig()
            self.logging = LoggingConfig()
            self._config_path = Path("config.yaml")
            self._load_config()
            self._initialized = True
    
    def _load_config(self):
        """Load configuration from YAML file"""
        if self._config_path.exists():
            try:
                with open(self._config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                    self._update_from_dict(config_data)
                    logger.info("Configuration loaded from config.yaml")
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        else:
            logger.info("No config.yaml found, using default configuration")
            self._create_default_config()
    
    def _update_from_dict(self, config_dict: Dict):
        """Update configuration from dictionary"""
        try:
            # Update scraper config
            if 'scraper' in config_dict:
                for key, value in config_dict['scraper'].items():
                    if hasattr(self.scraper, key):
                        setattr(self.scraper, key, value)
            
            # Update database config
            if 'database' in config_dict:
                for key, value in config_dict['database'].items():
                    if hasattr(self.database, key):
                        setattr(self.database, key, value)
            
            # Update model config
            if 'model' in config_dict:
                for key, value in config_dict['model'].items():
                    if hasattr(self.model, key):
                        setattr(self.model, key, value)
            
            # Update api config
            if 'api' in config_dict:
                for key, value in config_dict['api'].items():
                    if hasattr(self.api, key):
                        setattr(self.api, key, value)
            
            # Update prediction config
            if 'prediction' in config_dict:
                for key, value in config_dict['prediction'].items():
                    if hasattr(self.prediction, key):
                        setattr(self.prediction, key, value)
            
            # Update logging config
            if 'logging' in config_dict:
                for key, value in config_dict['logging'].items():
                    if hasattr(self.logging, key):
                        setattr(self.logging, key, value)
                        
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
    
    def _create_default_config(self):
        """Create default configuration file"""
        default_config = {
            'scraper': {
                'hollywoodbets_url': 'https://www.hollywoodbets.net',
                'betway_url': 'https://www.betway.co.za',
                'request_timeout': 30,
                'retry_attempts': 3,
                'delay_between_requests': 2.0
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'soccer_predictor',
                'user': 'postgres',
                'password': 'password'
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': True
            }
        }
        
        try:
            with open(self._config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logger.info(f"Created default configuration at {self._config_path}")
        except Exception as e:
            logger.error(f"Error creating default config: {e}")
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        return f"postgresql://{self.database.user}:{self.database.password}@{self.database.host}:{self.database.port}/{self.database.database}"
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        return f"redis://localhost:6379/0"

config = Config()
