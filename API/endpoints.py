from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import logging

from data.database.manager import DatabaseManager
from models.predictor import SoccerPredictor
from training.trainer import ModelTrainer
from utils.validators import validate_match_data, validate_team_data, validate_prediction_data
from utils.helpers import format_odds, calculate_kelly_criterion
from config.settings import config

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Soccer Prediction API",
    description="AI-powered soccer match prediction system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key security
api_key_header = APIKeyHeader(name=config.api.api_key_header, auto_error=False)

# Router
router = APIRouter()

# Pydantic models
class TeamCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    country: Optional[str] = None
    league: Optional[str] = None
    stadium: Optional[str] = None
    capacity: Optional[int] = None

class TeamResponse(BaseModel):
    id: int
    name: str
    country: Optional[str]
    league: Optional[str]
    stadium: Optional[str]
    capacity: Optional[int]
    created_at: datetime
    updated_at: datetime

class MatchCreate(BaseModel):
    home_team: str = Field(..., min_length=2)
    away_team: str = Field(..., min_length=2)
    league: str = Field(..., min_length=2)
    match_date: datetime
    venue: Optional[str] = None
    referee: Optional[str] = None
    home_odds: Optional[float] = Field(None, gt=1.0, lt=1000)
    draw_odds: Optional[float] = Field(None, gt=1.0, lt=1000)
    away_odds: Optional[float] = Field(None, gt=1.0, lt=1000)
    btts_yes_odds: Optional[float] = Field(None, gt=1.0, lt=1000)
    btts_no_odds: Optional[float] = Field(None, gt=1.0, lt=1000)
    weather_conditions: Optional[Dict[str, Any]] = None
    pitch_condition: Optional[str] = None

class MatchResponse(BaseModel):
    id: int
    home_team: str
    away_team: str
    league: str
    match_date: datetime
    venue: Optional[str]
    referee: Optional[str]
    home_odds: Optional[float]
    draw_odds: Optional[float]
    away_odds: Optional[float]
    btts_yes_odds: Optional[float]
    btts_no_odds: Optional[float]
    weather_conditions: Optional[Dict[str, Any]]
    pitch_condition: Optional[str]
    created_at: datetime
    updated_at: datetime

class PredictionResponse(BaseModel):
    match_id: int
    home_team: str
    away_team: str
    league: str
    match_date: datetime
    predicted_result: str
    predicted_btts: bool
    result_probabilities: Dict[str, float]
    btts_probabilities: Dict[str, float]
    result_confidence: float
    btts_confidence: float
    odds: Dict[str, Optional[float]]
    recommended_bets: List[Dict[str, Any]]
    prediction_timestamp: datetime
    model_version: str

class BetRecommendation(BaseModel):
    type: str
    bet: str
    confidence: float
    value: float
    odds: float
    stake_recommendation: str
    kelly_fraction: Optional[float] = None

class PredictionRequest(BaseModel):
    match_id: Optional[int] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    league: Optional[str] = None
    match_date: Optional[datetime] = None
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None
    btts_yes_odds: Optional[float] = None
    btts_no_odds: Optional[float] = None

class SystemStatus(BaseModel):
    status: str
    version: str
    uptime: float
    models_loaded: bool
    total_matches: int
    total_predictions: int
    last_training: Optional[datetime]
    model_accuracy: Optional[float]

# Dependency injection
async def get_db_manager():
    """Get database manager instance"""
    db_manager = DatabaseManager()
    await db_manager.initialize()
    try:
        yield db_manager
    finally:
        await db_manager.close()

async def get_predictor():
    """Get predictor instance"""
    db_manager = DatabaseManager()
    await db_manager.initialize()
    predictor = SoccerPredictor(db_manager)
    predictor.initialize_models()
    try:
        yield predictor
    finally:
        await db_manager.close()

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key"""
    # In production, you would check against a database of valid API keys
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Simple check for demonstration
    if len(api_key) < 20:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return api_key

# Health check endpoint
@router.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# System status
@router.get("/status", response_model=SystemStatus)
async def system_status(
    db_manager: DatabaseManager = Depends(get_db_manager),
    predictor: SoccerPredictor = Depends(get_predictor)
):
    """Get system status"""
    try:
        # Get total matches
        async with db_manager.AsyncSessionLocal() as session:
            from sqlalchemy import func
            from data.database.models import Match, Prediction
            
            match_count = await session.scalar(func.count(Match.id))
            prediction_count = await session.scalar(func.count(Prediction.id))
        
        # Calculate uptime (simplified)
        import time
        start_time = getattr(app.state, 'start_time', time.time())
        uptime = time.time() - start_time
        
        status = SystemStatus(
            status="running",
            version="1.0.0",
            uptime=uptime,
            models_loaded=predictor.is_initialized,
            total_matches=match_count,
            total_predictions=prediction_count,
            last_training=None,  # Would be fetched from database
            model_accuracy=0.75  # Would be calculated from recent predictions
        )
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Team endpoints
@router.post("/teams", response_model=TeamResponse)
async def create_team(
    team: TeamCreate,
    db_manager: DatabaseManager = Depends(get_db_manager),
    api_key: str = Depends(verify_api_key)
):
    """Create a new team"""
    try:
        # Validate team data
        is_valid, errors = validate_team_data(team.dict())
        if not is_valid:
            raise HTTPException(status_code=400, detail={"errors": errors})
        
        # Save team
        team_data = team.dict()
        saved_team = await db_manager.save_team(team_data)
        
        if not saved_team:
            raise HTTPException(status_code=500, detail="Failed to save team")
        
        # Convert to response model
        response = TeamResponse(
            id=saved_team.id,
            name=saved_team.name,
            country=saved_team.country,
            league=saved_team.league,
            stadium=saved_team.stadium,
            capacity=saved_team.capacity,
            created_at=saved_team.created_at,
            updated_at=saved_team.updated_at
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating team: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/teams", response_model=List[TeamResponse])
async def get_teams(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Get all teams"""
    try:
        teams = await db_manager.get_all_teams()
        
        # Apply pagination
        paginated_teams = teams[skip:skip + limit]
        
        # Convert to response models
        response_teams = []
        for team in paginated_teams:
            response_teams.append(TeamResponse(
                id=team.id,
                name=team.name,
                country=team.country,
                league=team.league,
                stadium=team.stadium,
                capacity=team.capacity,
                created_at=team.created_at,
                updated_at=team.updated_at
            ))
        
        return response_teams
        
    except Exception as e:
        logger.error(f"Error getting teams: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Match endpoints
@router.post("/matches", response_model=MatchResponse)
async def create_match(
    match: MatchCreate,
    db_manager: DatabaseManager = Depends(get_db_manager),
    api_key: str = Depends(verify_api_key)
):
    """Create a new match"""
    try:
        # Validate match data
        is_valid, errors = validate_match_data(match.dict())
        if not is_valid:
            raise HTTPException(status_code=400, detail={"errors": errors})
        
        # Save match
        match_data = match.dict()
        saved_match = await db_manager.save_match(match_data)
        
        if not saved_match:
            raise HTTPException(status_code=500, detail="Failed to save match")
        
        # Get team names
        home_team_name = match.home_team
        away_team_name = match.away_team
        
        # Convert to response model
        response = MatchResponse(
            id=saved_match.id,
            home_team=home_team_name,
            away_team=away_team_name,
            league=saved_match.league,
            match_date=saved_match.match_date,
            venue=saved_match.venue,
            referee=saved_match.referee,
            home_odds=saved_match.home_odds,
            draw_odds=saved_match.draw_odds,
            away_odds=saved_match.away_odds,
            btts_yes_odds=saved_match.btts_yes_odds,
            btts_no_odds=saved_match.btts_no_odds,
            weather_conditions=saved_match.weather_conditions,
            pitch_condition=saved_match.pitch_condition,
            created_at=saved_match.created_at,
            updated_at=saved_match.updated_at
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating match: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/matches/today", response_model=List[MatchResponse])
async def get_todays_matches(
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Get today's matches"""
    try:
        matches_data = await db_manager.get_todays_matches()
        
        response_matches = []
        for match_data in matches_data:
            response_matches.append(MatchResponse(
                id=match_data['id'],
                home_team=match_data['home_team'],
                away_team=match_data['away_team'],
                league=match_data['league'],
                match_date=match_data['match_date'],
                venue=match_data.get('venue'),
                referee=match_data.get('referee'),
                home_odds=match_data.get('home_odds'),
                draw_odds=match_data.get('draw_odds'),
                away_odds=match_data.get('away_odds'),
                btts_yes_odds=match_data.get('btts_yes_odds'),
                btts_no_odds=match_data.get('btts_no_odds'),
                weather_conditions=match_data.get('weather_conditions'),
                pitch_condition=match_data.get('pitch_condition'),
                created_at=datetime.now(),  # Would be fetched from database
                updated_at=datetime.now()   # Would be fetched from database
            ))
        
        return response_matches
        
    except Exception as e:
        logger.error(f"Error getting today's matches: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Prediction endpoints
@router.get("/predictions/today", response_model=List[PredictionResponse])
async def get_todays_predictions(
    predictor: SoccerPredictor = Depends(get_predictor)
):
    """Get predictions for today's matches"""
    try:
        predictions = predictor.predict_todays_matches()
        
        response_predictions = []
        for pred in predictions:
            response_predictions.append(PredictionResponse(
                match_id=pred['match_id'],
                home_team=pred['home_team'],
                away_team=pred['away_team'],
                league=pred['league'],
                match_date=pred.get('match_time', datetime.now()),
                predicted_result=pred['predicted_result'],
                predicted_btts=pred['predicted_btts'],
                result_probabilities=pred['result_probabilities'],
                btts_probabilities=pred['btts_probabilities'],
                result_confidence=pred['result_confidence'],
                btts_confidence=pred['btts_confidence'],
                odds=pred.get('odds', {}),
                recommended_bets=pred.get('recommended_bets', []),
                prediction_timestamp=datetime.fromisoformat(pred['prediction_timestamp']),
                model_version=pred['model_version']
            ))
        
        return response_predictions
        
    except Exception as e:
        logger.error(f"Error getting today's predictions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/predict", response_model=PredictionResponse)
async def predict_match(
    prediction_request: PredictionRequest,
    predictor: SoccerPredictor = Depends(get_predictor),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Predict a single match"""
    try:
        # Create match data from request
        match_data = {
            'home_team': prediction_request.home_team,
            'away_team': prediction_request.away_team,
            'league': prediction_request.league or 'Unknown',
            'match_date': prediction_request.match_date or datetime.now(),
            'odds': {
                'home_win': prediction_request.home_odds,
                'draw': prediction_request.draw_odds,
                'away_win': prediction_request.away_odds,
                'btts_yes': prediction_request.btts_yes_odds,
                'btts_no': prediction_request.btts_no_odds
            }
        }
        
        # Save match if not already exists
        saved_match = await db_manager.save_match(match_data)
        match_id = saved_match.id if saved_match else None
        
        # Get historical data for feature engineering
        historical_data = await db_manager.get_historical_matches(limit=1000)
        
        # Create features and predict
        features = predictor.feature_engineer.create_features(match_data, historical_data)
        
        if features.empty:
            raise HTTPException(status_code=400, detail="Could not create features for prediction")
        
        # Transform features
        X = predictor.feature_engineer.transform(features)
        
        # Predict result
        result_pred, result_prob = predictor.result_model.predict(X)
        
        # Predict BTTS
        btts_pred, btts_prob = predictor.btts_model.predict(X)
        
        # Create prediction response
        prediction = {
            'match_id': match_id,
            'home_team': match_data['home_team'],
            'away_team': match_data['away_team'],
            'league': match_data['league'],
            'match_time': match_data['match_date'],
            
            'predicted_result': result_pred[0],
            'result_probabilities': {
                'home_win': float(result_prob[0][0]),
                'draw': float(result_prob[0][1]),
                'away_win': float(result_prob[0][2])
            },
            
            'predicted_btts': bool(btts_pred[0]),
            'btts_probabilities': {
                'yes': float(btts_prob[0][1]),
                'no': float(btts_prob[0][0])
            },
            
            'result_confidence': float(np.max(result_prob[0])),
            'btts_confidence': float(np.max(btts_prob[0])),
            
            'odds': match_data['odds'],
            
            'recommended_bets': predictor._generate_recommendations(
                result_prob[0], 
                btts_prob[0],
                match_data['odds']
            ),
            
            'prediction_timestamp': datetime.now().isoformat(),
            'model_version': predictor.result_model.model_version
        }
        
        # Save prediction to database
        if match_id:
            await db_manager.save_prediction(prediction)
        
        # Convert to response model
        response = PredictionResponse(
            match_id=prediction['match_id'],
            home_team=prediction['home_team'],
            away_team=prediction['away_team'],
            league=prediction['league'],
            match_date=prediction['match_time'],
            predicted_result=prediction['predicted_result'],
            predicted_btts=prediction['predicted_btts'],
            result_probabilities=prediction['result_probabilities'],
            btts_probabilities=prediction['btts_probabilities'],
            result_confidence=prediction['result_confidence'],
            btts_confidence=prediction['btts_confidence'],
            odds=prediction['odds'],
            recommended_bets=prediction['recommended_bets'],
            prediction_timestamp=datetime.fromisoformat(prediction['prediction_timestamp']),
            model_version=prediction['model_version']
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error predicting match: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Training endpoints
@router.post("/train", status_code=202)
async def train_models(
    background_tasks: BackgroundTasks,
    retrain: bool = Query(False, description="Force retrain even if models exist"),
    api_key: str = Depends(verify_api_key)
):
    """Train or retrain models (background task)"""
    try:
        async def train_models_async():
            try:
                db_manager = DatabaseManager()
                await db_manager.initialize()
                
                trainer = ModelTrainer(db_manager)
                trainer.train_models(retrain=retrain)
                
                await db_manager.close()
                logger.info("Background training completed successfully")
            except Exception as e:
                logger.error(f"Error in background training: {e}")
        
        # Start training in background
        background_tasks.add_task(train_models_async)
        
        return {
            "message": "Training started in background",
            "retrain": retrain,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Statistics endpoints
@router.get("/statistics")
async def get_statistics(
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """Get system statistics"""
    try:
        # Get counts from database
        async with db_manager.AsyncSessionLocal() as session:
            from sqlalchemy import func
            from data.database.models import Match, Prediction, Team
            
            total_teams = await session.scalar(func.count(Team.id))
            total_matches = await session.scalar(func.count(Match.id))
            total_predictions = await session.scalar(func.count(Prediction.id))
            
            # Get accuracy of recent predictions
            recent_predictions = await db_manager.get_recent_predictions_with_results(days=30)
            
            if not recent_predictions.empty:
                correct_predictions = (
                    recent_predictions['predicted_result'] == 
                    recent_predictions['actual_result']
                ).sum()
                total_recent = len(recent_predictions)
                recent_accuracy = correct_predictions / total_recent if total_recent > 0 else 0
            else:
                recent_accuracy = 0
        
        statistics = {
            "total_teams": total_teams,
            "total_matches": total_matches,
            "total_predictions": total_predictions,
            "recent_accuracy": recent_accuracy,
            "updated_at": datetime.now().isoformat()
        }
        
        return statistics
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Include router
app.include_router(router, prefix="/api")

# Add startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    import time
    app.state.start_time = time.time()
    
    logger.info("Soccer Prediction API starting up...")
    
    # Initialize database connection pool
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        await db_manager.close()
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
    
    logger.info("Soccer Prediction API started successfully")

# Add shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Soccer Prediction API shutting down...")

# Main entry point for running the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.endpoints:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        workers=config.api.workers
    )
