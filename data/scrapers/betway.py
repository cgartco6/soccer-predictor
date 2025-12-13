import asyncio
from typing import Dict, List
from bs4 import BeautifulSoup
import re
from datetime import datetime
from .base_scraper import BaseScraper
import logging

logger = logging.getLogger(__name__)

class BetwayScraper(BaseScraper):
    def __init__(self):
        super().__init__(use_playwright=True)
        self.base_url = "https://www.betway.co.za"
        
    async def scrape_matches(self) -> List[Dict]:
        """Scrape today's matches from Betway"""
        matches = []
        url = f"{self.base_url}/sports/soccer"
        
        try:
            content = await self.fetch_page(url)
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find soccer events - adjust selectors
            events = soup.select('.eventItem, .fixture, [data-testid="soccer-event"]')
            
            for event in events:
                try:
                    match_data = await self._parse_betway_event(event)
                    if match_data:
                        matches.append(match_data)
                except Exception as e:
                    logger.error(f"Error parsing Betway event: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error scraping Betway: {e}")
            
        return matches
    
    async def _parse_betway_event(self, event) -> Optional[Dict]:
        """Parse Betway event data"""
        try:
            # Extract teams
            teams_elem = event.select_one('.teams, .competitors, .participants')
            if not teams_elem:
                return None
                
            home_elem = teams_elem.select_one('.home, .competitor1')
            away_elem = teams_elem.select_one('.away, .competitor2')
            
            if not home_elem or not away_elem:
                return None
                
            home_team = home_elem.get_text(strip=True)
            away_team = away_elem.get_text(strip=True)
            
            # Extract time
            time_elem = event.select_one('.time, .startTime, .eventTime')
            match_time = time_elem.get_text(strip=True) if time_elem else "TBD"
            
            # Extract league
            league_elem = event.select_one('.league, .competition, .category')
            league = league_elem.get_text(strip=True) if league_elem else "Unknown"
            
            # Extract odds
            odds_data = await self._extract_betway_odds(event)
            
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'match_time': match_time,
                'league': league,
                'source': 'betway',
                'odds': odds_data,
                'scraped_at': datetime.now().isoformat()
            }
            
            return match_data
            
        except Exception as e:
            logger.error(f"Error in _parse_betway_event: {e}")
            return None
    
    async def _extract_betway_odds(self, event) -> Dict:
        """Extract odds from Betway event"""
        odds = {
            'home_win': None,
            'draw': None,
            'away_win': None,
            'btts_yes': None,
            'btts_no': None
        }
        
        try:
            # Find main market odds (1X2)
            market_odds = event.select('.odds, .price, [data-testid="odds"]')
            
            if len(market_odds) >= 3:
                odds['home_win'] = self._parse_odds_value(market_odds[0])
                odds['draw'] = self._parse_odds_value(market_odds[1])
                odds['away_win'] = self._parse_odds_value(market_odds[2])
            
            # Look for BTTS market
            btts_market = event.find(string=re.compile(r'Both Teams To Score|BTTS', re.IGNORECASE))
            if btts_market:
                btts_parent = btts_market.find_parent()
                if btts_parent:
                    btts_odds = btts_parent.select('.odds, .price')
                    if len(btts_odds) >= 2:
                        odds['btts_yes'] = self._parse_odds_value(btts_odds[0])
                        odds['btts_no'] = self._parse_odds_value(btts_odds[1])
                        
        except Exception as e:
            logger.error(f"Error extracting Betway odds: {e}")
            
        return odds
    
    def _parse_odds_value(self, elem) -> Optional[float]:
        """Parse odds value from element"""
        try:
            text = elem.get_text(strip=True)
            # Handle fractional odds if present
            if '/' in text:
                num, den = text.split('/')
                return float(num) / float(den) + 1
            else:
                # Remove non-numeric
                text = re.sub(r'[^\d.]', '', text)
                if text:
                    return float(text)
        except:
            pass
        return None
