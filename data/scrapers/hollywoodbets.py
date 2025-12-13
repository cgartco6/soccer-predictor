import asyncio
from typing import Dict, List
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
from .base_scraper import BaseScraper
import logging

logger = logging.getLogger(__name__)

class HollywoodbetsScraper(BaseScraper):
    def __init__(self):
        super().__init__(use_playwright=True)
        self.base_url = "https://www.hollywoodbets.net"
        
    async def scrape_matches(self) -> List[Dict]:
        """Scrape today's matches from Hollywoodbets"""
        matches = []
        url = f"{self.base_url}/sport/Soccer"
        
        try:
            content = await self.fetch_page(url)
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find match containers - adjust selectors based on actual site structure
            match_containers = soup.select('.event-row, .match-container, [data-testid="match-row"]')
            
            for container in match_containers:
                try:
                    match_data = await self._parse_match(container)
                    if match_data:
                        matches.append(match_data)
                except Exception as e:
                    logger.error(f"Error parsing match: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping Hollywoodbets: {e}")
            
        return matches
    
    async def _parse_match(self, container) -> Optional[Dict]:
        """Parse individual match data"""
        try:
            # Extract team names
            teams_elem = container.select_one('.teams, .participants, [data-testid="team-names"]')
            if not teams_elem:
                return None
                
            teams_text = teams_elem.get_text(strip=True)
            teams = teams_text.split('vs')
            if len(teams) != 2:
                return None
                
            home_team = teams[0].strip()
            away_team = teams[1].strip()
            
            # Extract time
            time_elem = container.select_one('.time, .start-time, [data-testid="match-time"]')
            match_time = time_elem.get_text(strip=True) if time_elem else "TBD"
            
            # Extract league
            league_elem = container.select_one('.league, .competition, [data-testid="league-name"]')
            league = league_elem.get_text(strip=True) if league_elem else "Unknown"
            
            # Extract odds
            odds_data = await self._extract_odds(container)
            
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'match_time': match_time,
                'league': league,
                'source': 'hollywoodbets',
                'odds': odds_data,
                'scraped_at': datetime.now().isoformat()
            }
            
            return match_data
            
        except Exception as e:
            logger.error(f"Error in _parse_match: {e}")
            return None
    
    async def _extract_odds(self, container) -> Dict:
        """Extract odds from match container"""
        odds = {
            'home_win': None,
            'draw': None,
            'away_win': None,
            'btts_yes': None,
            'btts_no': None
        }
        
        try:
            # Find odds elements - adjust selectors based on actual structure
            odds_elems = container.select('.odds-value, .price, [data-testid="odds"]')
            
            if len(odds_elems) >= 3:
                odds['home_win'] = self._parse_odds_value(odds_elems[0])
                odds['draw'] = self._parse_odds_value(odds_elems[1])
                odds['away_win'] = self._parse_odds_value(odds_elems[2])
            
            # Look for BTTS odds
            btts_container = container.find(string=re.compile('Both Teams to Score', re.IGNORECASE))
            if btts_container:
                parent = btts_container.find_parent()
                if parent:
                    btts_odds = parent.select('.odds-value, .price')
                    if len(btts_odds) >= 2:
                        odds['btts_yes'] = self._parse_odds_value(btts_odds[0])
                        odds['btts_no'] = self._parse_odds_value(btts_odds[1])
                        
        except Exception as e:
            logger.error(f"Error extracting odds: {e}")
            
        return odds
    
    def _parse_odds_value(self, elem) -> Optional[float]:
        """Parse odds value from element"""
        try:
            text = elem.get_text(strip=True)
            # Remove any non-numeric characters except decimal point
            text = re.sub(r'[^\d.]', '', text)
            if text:
                return float(text)
        except:
            pass
        return None
    
    async def scrape_odds(self, match_url: str) -> Dict:
        """Scrape detailed odds for a specific match"""
        # Implementation for detailed odds page
        pass
