import os
from dataclasses import dataclass
from typing import Dict, List
import yaml

@dataclass
class ScraperConfig:
    hollywoodbets_url: str = "https://www.hollywoodbets.net"
    betway_url: str = "https://www.betway.co.za"
    request_timeout: int = 30
    retry_attempts: int = 3
    delay_between_requests: float = 2.0
    
@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "soccer_predictor"
    user: str = "postgres"
    password: str = "password"
    pool_size: int = 20
    
@dataclass
class ModelConfig:
    features: List[str] = None
    target_columns: List[str] = None
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    model_type: str = "ensemble"
    
@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    workers: int = 4
    
class Config:
    def __init__(self):
        self.scraper = ScraperConfig()
        self.database = DatabaseConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self._load_config()
        
    def _load_config(self):
        if os.path.exists("config.yaml"):
            with open("config.yaml", "r") as f:
                config_data = yaml.safe_load(f)
                self._update_from_dict(config_data)
                
    def _update_from_dict(self, config_dict: Dict):
        # Update configuration from dictionary
        pass
