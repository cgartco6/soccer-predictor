import asyncio
import logging
from datetime import datetime
import schedule
import time
from typing import List, Dict

from data.database.manager import DatabaseManager
from data.scrapers.hollywoodbets import HollywoodbetsScraper
from data.scrapers.betway import BetwayScraper
from data.scrapers.football_data import FootballDataScraper
from models.predictor import SoccerPredictor
from utils.logger import setup_logging

logger = setup_logging()

class SoccerPredictionSystem:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.predictor = SoccerPredictor(self.db_manager)
        self.scrapers = {
            'hollywoodbets': HollywoodbetsScraper(),
            'betway': BetwayScraper(),
            'football_data': FootballDataScraper()
        }
        
    async def initialize(self):
        """Initialize the system"""
        logger.info("Initializing Soccer Prediction System...")
        
        # Initialize database
        await self.db_manager.initialize()
        
        # Initialize scrapers
        for name, scraper in self.scrapers.items():
            await scraper.initialize()
            logger.info(f"Initialized {name} scraper")
        
        # Initialize predictor
        self.predictor.initialize_models()
        
        logger.info("System initialized successfully")
    
    async def scrape_all_data(self):
        """Scrape data from all sources"""
        logger.info("Starting data scraping...")
        
        all_matches = []
        
        # Scrape from Hollywoodbets
        try:
            hollywoodbets_matches = await self.scrapers['hollywoodbets'].scrape_matches()
            all_matches.extend(hollywoodbets_matches)
            logger.info(f"Scraped {len(hollywoodbets_matches)} matches from Hollywoodbets")
        except Exception as e:
            logger.error(f"Error scraping Hollywoodbets: {e}")
        
        # Scrape from Betway
        try:
            betway_matches = await self.scrapers['betway'].scrape_matches()
            all_matches.extend(betway_matches)
            logger.info(f"Scraped {len(betway_matches)} matches from Betway")
        except Exception as e:
            logger.error(f"Error scraping Betway: {e}")
        
        # Scrape additional football data
        try:
            football_data = await self.scrapers['football_data'].scrape_football_data(all_matches)
            # Merge additional data with matches
            logger.info(f"Scraped additional data for {len(football_data)} matches")
        except Exception as e:
            logger.error(f"Error scraping football data: {e}")
        
        # Save matches to database
        for match in all_matches:
            await self.db_manager.save_match(match)
        
        logger.info(f"Total matches scraped and saved: {len(all_matches)}")
    
    async def run_predictions(self):
        """Run predictions for today's matches"""
        logger.info("Running predictions...")
        
        try:
            predictions = self.predictor.predict_todays_matches()
            
            # Log predictions
            for pred in predictions:
                logger.info(
                    f"Prediction: {pred['home_team']} vs {pred['away_team']} - "
                    f"Result: {pred['predicted_result']} (conf: {pred['result_confidence']:.2f}), "
                    f"BTTS: {pred['predicted_btts']} (conf: {pred['btts_confidence']:.2f})"
                )
                
                # Log recommendations
                if pred['recommended_bets']:
                    for rec in pred['recommended_bets']:
                        logger.info(f"  Recommended bet: {rec['type']} - {rec['bet']} "
                                  f"(odds: {rec['odds']}, value: {rec['value']:.3f})")
            
            logger.info(f"Generated {len(predictions)} predictions")
            
        except Exception as e:
            logger.error(f"Error running predictions: {e}")
    
    async def update_with_results(self):
        """Update models with actual results"""
        logger.info("Updating models with actual results...")
        
        try:
            self.predictor.update_with_results()
            logger.info("Models updated successfully")
        except Exception as e:
            logger.error(f"Error updating models: {e}")
    
    async def run_daily_pipeline(self):
        """Run the complete daily pipeline"""
        logger.info(f"Starting daily pipeline at {datetime.now()}")
        
        # Step 1: Scrape data
        await self.scrape_all_data()
        
        # Step 2: Run predictions
        await self.run_predictions()
        
        # Step 3: For completed matches, update models
        await self.update_with_results()
        
        logger.info("Daily pipeline completed")
    
    def schedule_tasks(self):
        """Schedule periodic tasks"""
        # Run daily at 8 AM
        schedule.every().day.at("08:00").do(
            lambda: asyncio.create_task(self.run_daily_pipeline())
        )
        
        # Run predictions every 3 hours during match days
        schedule.every(3).hours.do(
            lambda: asyncio.create_task(self.run_predictions())
        )
        
        # Update models daily at midnight
        schedule.every().day.at("00:00").do(
            lambda: asyncio.create_task(self.update_with_results())
        )
        
        logger.info("Tasks scheduled")
    
    async def run(self):
        """Main run loop"""
        await self.initialize()
        self.schedule_tasks()
        
        logger.info("Soccer Prediction System is running. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                await asyncio.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        for scraper in self.scrapers.values():
            await scraper.close()
        
        await self.db_manager.close()
        logger.info("Cleanup completed")

async def main():
    """Main entry point"""
    system = SoccerPredictionSystem()
    await system.run()

if __name__ == "__main__":
    asyncio.run(main())
