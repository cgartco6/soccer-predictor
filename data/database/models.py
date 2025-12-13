from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class Team(Base):
    __tablename__ = 'teams'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True)
    country = Column(String(100))
    league = Column(String(100))
    stadium = Column(String(200))
    capacity = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    players = relationship("Player", back_populates="team")
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    statistics = relationship("TeamStatistics", back_populates="team")

class Player(Base):
    __tablename__ = 'players'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    team_id = Column(Integer, ForeignKey('teams.id'))
    position = Column(String(50))
    age = Column(Integer)
    nationality = Column(String(100))
    market_value = Column(Float)
    is_key_player = Column(Boolean, default=False)
    fitness_level = Column(Float)  # 0-1 scale
    injury_status = Column(String(50))
    expected_return = Column(DateTime)
    performance_data = Column(JSON)  # Stores recent performance metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    team = relationship("Team", back_populates="players")
    injuries = relationship("Injury", back_populates="player")

class Coach(Base):
    __tablename__ = 'coaches'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    team_id = Column(Integer, ForeignKey('teams.id'))
    nationality = Column(String(100))
    age = Column(Integer)
    win_rate = Column(Float)
    preferred_formation = Column(String(50))
    performance_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Match(Base):
    __tablename__ = 'matches'
    
    id = Column(Integer, primary_key=True)
    home_team_id = Column(Integer, ForeignKey('teams.id'))
    away_team_id = Column(Integer, ForeignKey('teams.id'))
    league = Column(String(100))
    match_date = Column(DateTime, nullable=False)
    venue = Column(String(200))
    referee = Column(String(200))
    
    # Pre-match data
    weather_conditions = Column(JSON)
    pitch_condition = Column(String(50))
    expected_attendance = Column(Integer)
    
    # Odds data
    home_odds = Column(Float)
    draw_odds = Column(Float)
    away_odds = Column(Float)
    btts_yes_odds = Column(Float)
    btts_no_odds = Column(Float)
    
    # Result data
    home_score = Column(Integer)
    away_score = Column(Integer)
    result = Column(String(1))  # H, D, A
    btts_result = Column(Boolean)  # True if both teams scored
    
    # Statistics
    possession_home = Column(Float)
    possession_away = Column(Float)
    shots_home = Column(Integer)
    shots_away = Column(Integer)
    shots_on_target_home = Column(Integer)
    shots_on_target_away = Column(Integer)
    corners_home = Column(Integer)
    corners_away = Column(Integer)
    fouls_home = Column(Integer)
    fouls_away = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    predictions = relationship("Prediction", back_populates="match")

class Injury(Base):
    __tablename__ = 'injuries'
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey('players.id'))
    match_id = Column(Integer, ForeignKey('matches.id'))
    injury_type = Column(String(100))
    severity = Column(String(50))  # Minor, Moderate, Severe
    estimated_recovery_time = Column(Integer)  # in days
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    player = relationship("Player", back_populates="injuries")

class TeamStatistics(Base):
    __tablename__ = 'team_statistics'
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.id'))
    season = Column(String(50))
    matches_played = Column(Integer)
    wins = Column(Integer)
    draws = Column(Integer)
    losses = Column(Integer)
    goals_for = Column(Integer)
    goals_against = Column(Integer)
    home_record = Column(JSON)  # {wins: X, draws: Y, losses: Z}
    away_record = Column(JSON)
    form_last_5 = Column(String(5))  # e.g., "WWDLW"
    avg_possession = Column(Float)
    avg_shots = Column(Float)
    avg_shots_on_target = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    team = relationship("Team", back_populates="statistics")

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey('matches.id'))
    predicted_result = Column(String(1))  # H, D, A
    predicted_btts = Column(Boolean)
    home_win_prob = Column(Float)
    draw_prob = Column(Float)
    away_win_prob = Column(Float)
    btts_yes_prob = Column(Float)
    btts_no_prob = Column(Float)
    confidence_score = Column(Float)
    features_used = Column(JSON)
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    match = relationship("Match", back_populates="predictions")

class ModelPerformance(Base):
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    model_version = Column(String(50))
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    log_loss = Column(Float)
    test_size = Column(Integer)
    training_date = Column(DateTime, default=datetime.utcnow)
    features_list = Column(JSON)
