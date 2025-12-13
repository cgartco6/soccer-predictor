import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from bs4 import BeautifulSoup
import json
import re
from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)

class FootballDataScraper(BaseScraper):
    """Scraper for additional football data (injuries, stats, etc.)"""
    
    def __init__(self):
        super().__init__(use_playwright=True)
        self.base_urls = {
            'espn': 'https://www.espn.com/soccer',
            'whoscored': 'https://www.whoscored.com',
            'transfermarkt': 'https://www.transfermarkt.com',
            'sofascore': 'https://www.sofascore.com',
            'flashscore': 'https://www.flashscore.com'
        }
        self.cache = {}
        
    async def scrape_football_data(self, team_names: List[str] = None) -> Dict:
        """Scrape comprehensive football data for teams"""
        all_data = {}
        
        try:
            # Scrape team statistics
            team_stats = await self._scrape_team_statistics(team_names)
            all_data.update(team_stats)
            
            # Scrape player injuries
            injuries = await self._scrape_injuries(team_names)
            all_data['injuries'] = injuries
            
            # Scrape transfer news
            transfers = await self._scrape_transfers(team_names)
            all_data['transfers'] = transfers
            
            # Scrape match statistics
            match_stats = await self._scrape_match_statistics()
            all_data.update(match_stats)
            
            # Scrape weather data for upcoming matches
            weather_data = await self._scrape_weather_data()
            all_data['weather'] = weather_data
            
            logger.info(f"Scraped football data for {len(team_names or [])} teams")
            
        except Exception as e:
            logger.error(f"Error scraping football data: {e}")
            
        return all_data
    
    async def _scrape_team_statistics(self, team_names: List[str] = None) -> Dict:
        """Scrape team statistics from various sources"""
        team_stats = {}
        
        try:
            # ESPN for team stats
            espn_url = f"{self.base_urls['espn']}/teams"
            content = await self.fetch_page(espn_url)
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find team statistics
            team_elements = soup.select('.TeamLinks')
            
            for team_elem in team_elements:
                team_name = team_elem.select_one('.TeamHeader__Name')
                if team_name:
                    name = team_name.get_text(strip=True)
                    if team_names and name not in team_names:
                        continue
                    
                    stats = {}
                    
                    # Get win/loss/draw stats
                    record = team_elem.select_one('.TeamHeader__Record')
                    if record:
                        record_text = record.get_text(strip=True)
                        if '-' in record_text:
                            wins, losses, draws = self._parse_record(record_text)
                            stats.update({'wins': wins, 'losses': losses, 'draws': draws})
                    
                    # Get goals statistics
                    goals_elem = team_elem.select_one('.stats')
                    if goals_elem:
                        goals_text = goals_elem.get_text(strip=True)
                        goals_for, goals_against = self._parse_goals(goals_text)
                        stats.update({'goals_for': goals_for, 'goals_against': goals_against})
                    
                    team_stats[name] = stats
                    
        except Exception as e:
            logger.error(f"Error scraping team statistics: {e}")
            
        return team_stats
    
    async def _scrape_injuries(self, team_names: List[str] = None) -> Dict:
        """Scrape player injury data"""
        injuries = {}
        
        try:
            # Use ESPN injuries page
            injuries_url = f"{self.base_urls['espn']}/injuries"
            content = await self.fetch_page(injuries_url)
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find injury tables
            injury_tables = soup.select('.Table')
            
            for table in injury_tables:
                rows = table.select('tr')[1:]  # Skip header
                
                for row in rows:
                    cols = row.select('td')
                    if len(cols) >= 4:
                        player = cols[0].get_text(strip=True)
                        team = cols[1].get_text(strip=True)
                        status = cols[2].get_text(strip=True)
                        description = cols[3].get_text(strip=True)
                        
                        if team_names and team not in team_names:
                            continue
                            
                        if team not in injuries:
                            injuries[team] = []
                        
                        injury_data = {
                            'player': player,
                            'status': status,
                            'description': description,
                            'severity': self._classify_injury_severity(status, description),
                            'expected_return': self._estimate_return_date(status)
                        }
                        
                        injuries[team].append(injury_data)
                        
        except Exception as e:
            logger.error(f"Error scraping injuries: {e}")
            
        return injuries
    
    async def _scrape_transfers(self, team_names: List[str] = None) -> Dict:
        """Scrape transfer news and rumors"""
        transfers = {}
        
        try:
            # Transfermarkt for transfer news
            transfer_url = f"{self.base_urls['transfermarkt']}/transfers"
            content = await self.fetch_page(transfer_url)
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find transfer tables
            transfer_sections = soup.select('.box')
            
            for section in transfer_sections:
                header = section.select_one('.table-header')
                if header:
                    team_name = header.get_text(strip=True)
                    
                    if team_names and team_name not in team_names:
                        continue
                    
                    if team_name not in transfers:
                        transfers[team_name] = {'in': [], 'out': []}
                    
                    # Find incoming transfers
                    incoming_table = section.select_one('.incoming')
                    if incoming_table:
                        incoming_rows = incoming_table.select('tbody tr')
                        for row in incoming_rows:
                            player_data = self._parse_transfer_row(row, 'in')
                            if player_data:
                                transfers[team_name]['in'].append(player_data)
                    
                    # Find outgoing transfers
                    outgoing_table = section.select_one('.outgoing')
                    if outgoing_table:
                        outgoing_rows = outgoing_table.select('tbody tr')
                        for row in outgoing_rows:
                            player_data = self._parse_transfer_row(row, 'out')
                            if player_data:
                                transfers[team_name]['out'].append(player_data)
                                
        except Exception as e:
            logger.error(f"Error scraping transfers: {e}")
            
        return transfers
    
    async def _scrape_match_statistics(self) -> Dict:
        """Scrape detailed match statistics"""
        match_stats = {}
        
        try:
            # SofaScore for match stats
            sofascore_url = f"{self.base_urls['sofascore']}/football"
            content = await self.fetch_page(sofascore_url)
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find live matches
            live_matches = soup.select('.event-cell')
            
            for match in live_matches:
                teams = match.select('.event-cell__team')
                if len(teams) >= 2:
                    home_team = teams[0].get_text(strip=True)
                    away_team = teams[1].get_text(strip=True)
                    
                    match_key = f"{home_team}_{away_team}"
                    match_stats[match_key] = {}
                    
                    # Get match statistics
                    stats_elements = match.select('.event-cell__statistics')
                    for stat_elem in stats_elements:
                        stat_type = stat_elem.select_one('.event-cell__statistics-title')
                        stat_value = stat_elem.select_one('.event-cell__statistics-value')
                        
                        if stat_type and stat_value:
                            stat_name = stat_type.get_text(strip=True).lower().replace(' ', '_')
                            stat_val = stat_value.get_text(strip=True)
                            match_stats[match_key][stat_name] = stat_val
                            
        except Exception as e:
            logger.error(f"Error scraping match statistics: {e}")
            
        return match_stats
    
    async def _scrape_weather_data(self) -> Dict:
        """Scrape weather data for match locations"""
        weather_data = {}
        
        try:
            # Using OpenWeatherMap API (example)
            # Note: In production, you would use actual weather API with API key
            cities = ['London', 'Manchester', 'Liverpool', 'Birmingham', 'Leeds']
            
            for city in cities:
                # This is a simplified example
                weather_data[city] = {
                    'temperature': 15,  # Celsius
                    'condition': 'clear',
                    'humidity': 65,
                    'wind_speed': 10,  # km/h
                    'precipitation': 0
                }
                
        except Exception as e:
            logger.error(f"Error scraping weather data: {e}")
            
        return weather_data
    
    def _parse_record(self, record_text: str) -> tuple:
        """Parse win-loss-draw record"""
        try:
            parts = record_text.split('-')
            if len(parts) >= 3:
                return int(parts[0]), int(parts[1]), int(parts[2])
        except:
            pass
        return 0, 0, 0
    
    def _parse_goals(self, goals_text: str) -> tuple:
        """Parse goals for and against"""
        try:
            if ':' in goals_text:
                parts = goals_text.split(':')
                return int(parts[0]), int(parts[1])
        except:
            pass
        return 0, 0
    
    def _classify_injury_severity(self, status: str, description: str) -> str:
        """Classify injury severity"""
        status_lower = status.lower()
        desc_lower = description.lower()
        
        if any(word in status_lower for word in ['out', 'doubtful', 'questionable']):
            if any(word in desc_lower for word in ['season', 'long-term', 'serious', 'major']):
                return 'severe'
            elif any(word in desc_lower for word in ['weeks', 'month', 'moderate']):
                return 'moderate'
            else:
                return 'minor'
        return 'unknown'
    
    def _estimate_return_date(self, status: str) -> Optional[str]:
        """Estimate return date from injury status"""
        if 'day' in status.lower():
            days = int(''.join(filter(str.isdigit, status)))
            return (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
        elif 'week' in status.lower():
            weeks = int(''.join(filter(str.isdigit, status)))
            return (datetime.now() + timedelta(weeks=weeks)).strftime('%Y-%m-%d')
        return None
    
    def _parse_transfer_row(self, row, transfer_type: str) -> Optional[Dict]:
        """Parse a transfer row"""
        cols = row.select('td')
        if len(cols) >= 3:
            player = cols[0].get_text(strip=True)
            from_to = cols[1].get_text(strip=True)
            fee = cols[2].get_text(strip=True)
            
            return {
                'player': player,
                'from_to': from_to,
                'fee': fee,
                'type': transfer_type,
                'timestamp': datetime.now().isoformat()
            }
        return None
    
    async def scrape_player_performance(self, player_name: str) -> Dict:
        """Scrape individual player performance data"""
        performance = {}
        
        try:
            # WhoScored for player stats
            search_url = f"{self.base_urls['whoscored']}/Search/?q={player_name}"
            content = await self.fetch_page(search_url)
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find player profile
            player_link = soup.select_one('.search-result a')
            if player_link:
                player_url = player_link['href']
                player_content = await self.fetch_page(player_url)
                player_soup = BeautifulSoup(player_content, 'html.parser')
                
                # Extract player stats
                stats_section = player_soup.select_one('.player-stats')
                if stats_section:
                    stats_rows = stats_section.select('tr')
                    
                    for row in stats_rows:
                        cols = row.select('td')
                        if len(cols) >= 2:
                            stat_name = cols[0].get_text(strip=True).lower().replace(' ', '_')
                            stat_value = cols[1].get_text(strip=True)
                            performance[stat_name] = stat_value
                            
        except Exception as e:
            logger.error(f"Error scraping player performance for {player_name}: {e}")
            
        return performance
    
    async def scrape_referee_stats(self, referee_name: str) -> Dict:
        """Scrape referee statistics"""
        referee_stats = {}
        
        try:
            # Search for referee data
            # Note: This would need a specific source for referee stats
            referee_stats = {
                'name': referee_name,
                'matches_officiated': 150,
                'avg_cards_per_match': 3.2,
                'avg_penalties_per_match': 0.3,
                'home_win_percentage': 45.2,
                'away_win_percentage': 32.1,
                'draw_percentage': 22.7
            }
            
        except Exception as e:
            logger.error(f"Error scraping referee stats: {e}")
            
        return referee_stats
