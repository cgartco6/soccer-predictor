import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from sqlalchemy import create_engine, text, func, and_, or_
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
import pandas as pd
import json

from config.settings import config
from .models import Base, Team, Player, Coach, Match, Injury, TeamStatistics, Prediction, ModelPerformance

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for soccer prediction system"""
    
    def __init__(self):
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize database connection"""
        try:
            # Create synchronous engine
            database_url = config.get_database_url()
            self.engine = create_engine(
                database_url,
                pool_size=config.database.pool_size,
                max_overflow=config.database.max_overflow,
                pool_recycle=config.database.pool_recycle,
                echo=config.database.echo
            )
            
            # Create async engine
            async_database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')
            self.async_engine = create_async_engine(
                async_database_url,
                echo=config.database.echo,
                pool_size=config.database.pool_size
            )
            
            # Create session factories
            self.SessionLocal = scoped_session(
                sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            )
            
            self.AsyncSessionLocal = sessionmaker(
                self.async_engine, class_=AsyncSession, expire_on_commit=False
            )
            
            # Create tables
            await self.create_tables()
            
            self.is_initialized = True
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    async def create_tables(self):
        """Create database tables"""
        try:
            # Create tables synchronously for simplicity
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    async def close(self):
        """Close database connections"""
        try:
            if self.engine:
                self.engine.dispose()
            if self.async_engine:
                await self.async_engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database: {e}")
    
    # Team operations
    async def save_team(self, team_data: Dict) -> Optional[Team]:
        """Save or update team"""
        async with self.AsyncSessionLocal() as session:
            try:
                # Check if team exists
                result = await session.execute(
                    select(Team).where(Team.name == team_data['name'])
                )
                existing_team = result.scalars().first()
                
                if existing_team:
                    # Update existing team
                    for key, value in team_data.items():
                        if hasattr(existing_team, key):
                            setattr(existing_team, key, value)
                    existing_team.updated_at = datetime.utcnow()
                else:
                    # Create new team
                    team_data['created_at'] = datetime.utcnow()
                    team_data['updated_at'] = datetime.utcnow()
                    existing_team = Team(**team_data)
                    session.add(existing_team)
                
                await session.commit()
                logger.info(f"Team saved: {team_data['name']}")
                return existing_team
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error saving team {team_data.get('name')}: {e}")
                return None
    
    async def get_team(self, team_id: int = None, team_name: str = None) -> Optional[Team]:
        """Get team by ID or name"""
        async with self.AsyncSessionLocal() as session:
            try:
                query = select(Team)
                
                if team_id:
                    query = query.where(Team.id == team_id)
                elif team_name:
                    query = query.where(Team.name == team_name)
                else:
                    return None
                
                result = await session.execute(query)
                team = result.scalars().first()
                return team
                
            except Exception as e:
                logger.error(f"Error getting team: {e}")
                return None
    
    async def get_all_teams(self) -> List[Team]:
        """Get all teams"""
        async with self.AsyncSessionLocal() as session:
            try:
                result = await session.execute(select(Team))
                teams = result.scalars().all()
                return teams
            except Exception as e:
                logger.error(f"Error getting all teams: {e}")
                return []
    
    # Match operations
    async def save_match(self, match_data: Dict) -> Optional[Match]:
        """Save or update match"""
        async with self.AsyncSessionLocal() as session:
            try:
                # Get or create teams
                home_team = await self._get_or_create_team(session, match_data['home_team'])
                away_team = await self._get_or_create_team(session, match_data['away_team'])
                
                if not home_team or not away_team:
                    logger.error(f"Could not find/create teams for match")
                    return None
                
                # Check if match exists
                result = await session.execute(
                    select(Match).where(
                        and_(
                            Match.home_team_id == home_team.id,
                            Match.away_team_id == away_team.id,
                            Match.match_date == match_data.get('match_date')
                        )
                    )
                )
                existing_match = result.scalars().first()
                
                # Prepare match data
                match_data_prepared = match_data.copy()
                match_data_prepared['home_team_id'] = home_team.id
                match_data_prepared['away_team_id'] = away_team.id
                
                # Convert odds to JSON if they're dict
                if 'odds' in match_data_prepared and isinstance(match_data_prepared['odds'], dict):
                    match_data_prepared['odds'] = json.dumps(match_data_prepared['odds'])
                
                if existing_match:
                    # Update existing match
                    for key, value in match_data_prepared.items():
                        if hasattr(existing_match, key):
                            setattr(existing_match, key, value)
                    existing_match.updated_at = datetime.utcnow()
                else:
                    # Create new match
                    match_data_prepared['created_at'] = datetime.utcnow()
                    match_data_prepared['updated_at'] = datetime.utcnow()
                    existing_match = Match(**match_data_prepared)
                    session.add(existing_match)
                
                await session.commit()
                logger.info(f"Match saved: {home_team.name} vs {away_team.name}")
                return existing_match
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error saving match: {e}")
                return None
    
    async def get_todays_matches(self) -> List[Dict]:
        """Get today's matches"""
        async with self.AsyncSessionLocal() as session:
            try:
                today = datetime.now().date()
                tomorrow = today + timedelta(days=1)
                
                result = await session.execute(
                    select(Match).where(
                        and_(
                            Match.match_date >= today,
                            Match.match_date < tomorrow
                        )
                    ).join(Team, Match.home_team_id == Team.id)
                )
                matches = result.scalars().all()
                
                # Convert to dictionary format
                matches_data = []
                for match in matches:
                    match_dict = {
                        'id': match.id,
                        'home_team': match.home_team.name,
                        'away_team': match.away_team.name,
                        'league': match.league,
                        'match_date': match.match_date,
                        'venue': match.venue,
                        'referee': match.referee,
                        'home_odds': match.home_odds,
                        'draw_odds': match.draw_odds,
                        'away_odds': match.away_odds,
                        'btts_yes_odds': match.btts_yes_odds,
                        'btts_no_odds': match.btts_no_odds,
                        'weather_conditions': match.weather_conditions,
                        'pitch_condition': match.pitch_condition
                    }
                    matches_data.append(match_dict)
                
                return matches_data
                
            except Exception as e:
                logger.error(f"Error getting today's matches: {e}")
                return []
    
    async def get_historical_matches(self, limit: int = 10000, days_back: int = 365) -> pd.DataFrame:
        """Get historical matches for training"""
        try:
            # Use synchronous session for pandas
            session = self.SessionLocal()
            
            try:
                cutoff_date = datetime.now() - timedelta(days=days_back)
                
                query = session.query(
                    Match.id,
                    Team.name.label('home_team'),
                    Team2.name.label('away_team'),
                    Match.league,
                    Match.match_date,
                    Match.home_score,
                    Match.away_score,
                    Match.result,
                    Match.btts_result,
                    Match.home_odds,
                    Match.draw_odds,
                    Match.away_odds,
                    Match.btts_yes_odds,
                    Match.btts_no_odds,
                    Match.weather_conditions,
                    Match.pitch_condition,
                    Match.referee,
                    Match.possession_home,
                    Match.possession_away,
                    Match.shots_home,
                    Match.shots_away
                ).join(Team, Match.home_team_id == Team.id
                ).join(Team2, Match.away_team_id == Team2.id
                ).filter(
                    Match.match_date >= cutoff_date,
                    Match.result.isnot(None)
                ).order_by(Match.match_date.desc()).limit(limit)
                
                df = pd.read_sql(query.statement, session.bind)
                return df
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error getting historical matches: {e}")
            return pd.DataFrame()
    
    # Prediction operations
    async def save_prediction(self, prediction_data: Dict) -> Optional[Prediction]:
        """Save prediction"""
        async with self.AsyncSessionLocal() as session:
            try:
                # Create prediction
                prediction = Prediction(
                    match_id=prediction_data['match_id'],
                    predicted_result=prediction_data['predicted_result'],
                    predicted_btts=prediction_data['predicted_btts'],
                    home_win_prob=prediction_data['result_probabilities']['home_win'],
                    draw_prob=prediction_data['result_probabilities']['draw'],
                    away_win_prob=prediction_data['result_probabilities']['away_win'],
                    btts_yes_prob=prediction_data['btts_probabilities']['yes'],
                    btts_no_prob=prediction_data['btts_probabilities']['no'],
                    confidence_score=prediction_data['result_confidence'],
                    features_used=prediction_data.get('features_used', {}),
                    model_version=prediction_data.get('model_version', 'unknown'),
                    created_at=datetime.utcnow()
                )
                
                session.add(prediction)
                await session.commit()
                
                logger.info(f"Prediction saved for match {prediction_data['match_id']}")
                return prediction
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error saving prediction: {e}")
                return None
    
    async def get_recent_predictions_with_results(self, days: int = 7) -> pd.DataFrame:
        """Get recent predictions with actual results"""
        try:
            session = self.SessionLocal()
            
            try:
                cutoff_date = datetime.now() - timedelta(days=days)
                
                query = session.query(
                    Prediction.id,
                    Prediction.match_id,
                    Prediction.predicted_result,
                    Prediction.predicted_btts,
                    Prediction.home_win_prob,
                    Prediction.draw_prob,
                    Prediction.away_win_prob,
                    Prediction.btts_yes_prob,
                    Prediction.btts_no_prob,
                    Prediction.confidence_score,
                    Match.result.label('actual_result'),
                    Match.btts_result.label('actual_btts')
                ).join(Match, Prediction.match_id == Match.id
                ).filter(
                    Prediction.created_at >= cutoff_date,
                    Match.result.isnot(None)
                )
                
                df = pd.read_sql(query.statement, session.bind)
                return df
                
            finally:
                session.close()
                
        except Exception as e:
            logger.error(f"Error getting recent predictions: {e}")
            return pd.DataFrame()
    
    # Model performance operations
    async def save_model_performance(self, **kwargs) -> Optional[ModelPerformance]:
        """Save model performance metrics"""
        async with self.AsyncSessionLocal() as session:
            try:
                performance = ModelPerformance(**kwargs)
                session.add(performance)
                await session.commit()
                
                logger.info(f"Model performance saved for version {kwargs.get('model_version')}")
                return performance
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error saving model performance: {e}")
                return None
    
    # Helper methods
    async def _get_or_create_team(self, session, team_name: str) -> Optional[Team]:
        """Get existing team or create new one"""
        try:
            result = await session.execute(
                select(Team).where(Team.name == team_name)
            )
            team = result.scalars().first()
            
            if not team:
                team = Team(name=team_name, created_at=datetime.utcnow())
                session.add(team)
                await session.flush()
                logger.info(f"Created new team: {team_name}")
            
            return team
            
        except Exception as e:
            logger.error(f"Error getting/creating team {team_name}: {e}")
            return None
    
    async def bulk_insert_matches(self, matches_data: List[Dict]) -> int:
        """Bulk insert matches"""
        count = 0
        for match_data in matches_data:
            match = await self.save_match(match_data)
            if match:
                count += 1
        return count
    
    async def get_team_statistics(self, team_id: int, season: str = None) -> Optional[TeamStatistics]:
        """Get team statistics"""
        async with self.AsyncSessionLocal() as session:
            try:
                query = select(TeamStatistics).where(TeamStatistics.team_id == team_id)
                
                if season:
                    query = query.where(TeamStatistics.season == season)
                else:
                    # Get latest season
                    query = query.order_by(TeamStatistics.created_at.desc())
                
                result = await session.execute(query)
                stats = result.scalars().first()
                return stats
                
            except Exception as e:
                logger.error(f"Error getting team statistics: {e}")
                return None
    
    async def update_match_result(self, match_id: int, result_data: Dict) -> bool:
        """Update match result"""
        async with self.AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(Match).where(Match.id == match_id)
                )
                match = result.scalars().first()
                
                if match:
                    for key, value in result_data.items():
                        if hasattr(match, key):
                            setattr(match, key, value)
                    
                    match.updated_at = datetime.utcnow()
                    await session.commit()
                    logger.info(f"Match result updated for match {match_id}")
                    return True
                    
                return False
                
            except Exception as e:
                await session.rollback()
                logger.error(f"Error updating match result: {e}")
                return False
