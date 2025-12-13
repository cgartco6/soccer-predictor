#!/usr/bin/env python3
"""
Main entry point for Soccer Prediction System
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime
import schedule
import time

from config.settings import config
from data.database.manager import DatabaseManager
from data.scrapers.hollywoodbets import HollywoodbetsScraper
from data.scrapers.betway import BetwayScraper
from data.scrapers.football_data import FootballDataScraper
from models.predictor import SoccerPredictor
from training.trainer import ModelTrainer
from utils.logger import setup_logging

# Set up logging
logger = setup_logging()

class SoccerPredictionSystem:
    """Main system class"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.predictor = None
        self.trainer = None
        self.scrapers = {}
        self.running = False
        
    async def initialize(self):
        """Initialize the system"""
        logger.info("Initializing Soccer Prediction System...")
        
        try:
            # Initialize database
            await self.db_manager.initialize()
            logger.info("Database initialized")
            
            # Initialize scrapers
            self.scrapers = {
                'hollywoodbets': HollywoodbetsScraper(),
                'betway': BetwayScraper(),
                'football_data': FootballDataScraper()
            }
            
            for name, scraper in self.scrapers.items():
                await scraper.initialize()
                logger.info(f"Initialized {name} scraper")
            
            # Initialize predictor
            self.predictor = SoccerPredictor(self.db_manager)
            self.predictor.initialize_models()
            
            # Initialize trainer
            self.trainer = ModelTrainer(self.db_manager)
            
            # Check if models need retraining
            self.trainer.retrain_if_needed()
            
            logger.info("System initialized successfully")
            self.running = True
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            raise
    
    async def scrape_all_data(self):
        """Scrape data from all sources"""
        logger.info("Starting data scraping...")
        
        all_matches = []
        
        # Scrape from Hollywoodbets
        try:
            logger.info("Scraping Hollywoodbets...")
            hollywoodbets_matches = await self.scrapers['hollywoodbets'].scrape_matches()
            all_matches.extend(hollywoodbets_matches)
            logger.info(f"Scraped {len(hollywoodbets_matches)} matches from Hollywoodbets")
        except Exception as e:
            logger.error(f"Error scraping Hollywoodbets: {e}")
        
        # Scrape from Betway
        try:
            logger.info("Scraping Betway...")
            betway_matches = await self.scrapers['betway'].scrape_matches()
            all_matches.extend(betway_matches)
            logger.info(f"Scraped {len(betway_matches)} matches from Betway")
        except Exception as e:
            logger.error(f"Error scraping Betway: {e}")
        
        # Scrape additional football data
        try:
            logger.info("Scraping additional football data...")
            team_names = list(set([m.get('home_team') for m in all_matches] + 
                                 [m.get('away_team') for m in all_matches]))
            football_data = await self.scrapers['football_data'].scrape_football_data(team_names)
            logger.info(f"Scraped additional data for {len(team_names)} teams")
        except Exception as e:
            logger.error(f"Error scraping football data: {e}")
            football_data = {}
        
        # Process and save matches
        saved_count = 0
        for match_data in all_matches:
            try:
                # Merge with football data if available
                home_team = match_data.get('home_team')
                away_team = match_data.get('away_team')
                
                if home_team in football_data.get('injuries', {}):
                    match_data['home_injuries'] = football_data['injuries'][home_team]
                
                if away_team in football_data.get('injuries', {}):
                    match_data['away_injuries'] = football_data['injuries'][away_team]
                
                # Save match
                saved = await self.db_manager.save_match(match_data)
                if saved:
                    saved_count += 1
                    
            except Exception as e:
                logger.error(f"Error saving match {match_data.get('home_team')}: {e}")
                continue
        
        logger.info(f"Saved {saved_count} matches to database")
        return saved_count
    
    async def run_predictions(self):
        """Run predictions for today's matches"""
        logger.info("Running predictions...")
        
        try:
            predictions = self.predictor.predict_todays_matches()
            
            # Log predictions
            for pred in predictions:
                logger.info(
                    f"Prediction: {pred['home_team']} vs {pred['away_team']} - "
                    f"Result: {pred['predicted_result']} ({pred['result_confidence']:.2f}), "
                    f"BTTS: {pred['predicted_btts']} ({pred['btts_confidence']:.2f})"
                )
                
                # Log recommendations if any
                if pred.get('recommended_bets'):
                    for rec in pred['recommended_bets']:
                        logger.info(f"  Recommended: {rec['type']} - {rec['bet']} "
                                  f"(odds: {rec['odds']}, value: {rec['value']:.3f})")
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error running predictions: {e}")
            return []
    
    async def update_with_results(self):
        """Update models with actual results"""
        logger.info("Updating models with actual results...")
        
        try:
            self.predictor.update_with_results()
            logger.info("Models updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating models: {e}")
            return False
    
    async def run_daily_pipeline(self):
        """Run the complete daily pipeline"""
        logger.info(f"Starting daily pipeline at {datetime.now()}")
        
        try:
            # Step 1: Scrape data
            await self.scrape_all_data()
            
            # Step 2: Run predictions
            predictions = await self.run_predictions()
            
            # Step 3: For completed matches, update models
            await self.update_with_results()
            
            # Step 4: Generate report
            self._generate_daily_report(predictions)
            
            logger.info("Daily pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in daily pipeline: {e}")
    
    def _generate_daily_report(self, predictions):
        """Generate daily report"""
        try:
            report_dir = Path("reports/daily")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"report_{datetime.now().strftime('%Y%m%d')}.txt"
            
            with open(report_file, 'w') as f:
                f.write(f"Daily Prediction Report\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n")
                f.write(f"Total Predictions: {len(predictions)}\n\n")
                
                for pred in predictions:
                    f.write(f"Match: {pred['home_team']} vs {pred['away_team']}\n")
                    f.write(f"  Predicted Result: {pred['predicted_result']} "
                           f"(Confidence: {pred['result_confidence']:.2f})\n")
                    f.write(f"  Predicted BTTS: {pred['predicted_btts']} "
                           f"(Confidence: {pred['btts_confidence']:.2f})\n")
                    
                    if pred.get('recommended_bets'):
                        f.write(f"  Recommendations:\n")
                        for rec in pred['recommended_bets']:
                            f.write(f"    - {rec['type']}: {rec['bet']} "
                                   f"(Odds: {rec['odds']}, Value: {rec['value']:.3f})\n")
                    f.write("\n")
            
            logger.info(f"Daily report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
    
    def schedule_tasks(self):
        """Schedule periodic tasks"""
        # Run daily pipeline at 8 AM
        schedule.every().day.at("08:00").do(
            lambda: asyncio.create_task(self.run_daily_pipeline())
        )
        
        # Run predictions every 2 hours during match days
        schedule.every(2).hours.do(
            lambda: asyncio.create_task(self.run_predictions())
        )
        
        # Update models daily at midnight
        schedule.every().day.at("00:00").do(
            lambda: asyncio.create_task(self.update_with_results())
        )
        
        # Check for retraining weekly
        schedule.every(7).days.do(
            lambda: self.trainer.retrain_if_needed()
        )
        
        logger.info("Tasks scheduled")
    
    async def run(self):
        """Main run loop"""
        # Handle shutdown signals
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize system
        await self.initialize()
        
        # Schedule tasks
        self.schedule_tasks()
        
        logger.info("Soccer Prediction System is running. Press Ctrl+C to stop.")
        
        # Main loop
        try:
            while self.running:
                schedule.run_pending()
                await asyncio.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        
        try:
            # Close scrapers
            for scraper in self.scrapers.values():
                await scraper.close()
            
            # Close database
            await self.db_manager.close()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main entry point"""
    system = SoccerPredictionSystem()
    
    try:
        # Run the system
        asyncio.run(system.run())
    except KeyboardInterrupt:
        logger.info("System shutdown by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
