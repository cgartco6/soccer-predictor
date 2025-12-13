import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from data.scrapers.base_scraper import BaseScraper
from data.scrapers.hollywoodbets import HollywoodbetsScraper
from data.scrapers.betway import BetwayScraper
from data.scrapers.football_data import FootballDataScraper

class TestBaseScraper:
    """Test BaseScraper class"""
    
    @pytest.fixture
    def base_scraper(self):
        return BaseScraper()
    
    @pytest.mark.asyncio
    async def test_initialization(self, base_scraper):
        """Test scraper initialization"""
        await base_scraper.initialize()
        assert base_scraper.session is not None or base_scraper.browser is not None
        await base_scraper.close()
    
    @pytest.mark.asyncio
    async def test_fetch_page_with_playwright(self, base_scraper):
        """Test page fetching with Playwright"""
        base_scraper.use_playwright = True
        
        # Mock Playwright page
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.content = AsyncMock(return_value="<html>Test content</html>")
        mock_page.close = AsyncMock()
        
        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        
        base_scraper.context = mock_context
        
        content = await base_scraper.fetch_page("http://test.com")
        assert content == "<html>Test content</html>"
        mock_page.goto.assert_called_once_with("http://test.com", wait_until="networkidle")
    
    def test_parse_json_ld(self, base_scraper):
        """Test JSON-LD parsing"""
        # Create mock HTML with JSON-LD
        html = """
        <html>
            <script type="application/ld+json">
                {"@type": "SportsEvent", "name": "Test Match"}
            </script>
        </html>
        """
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        result = base_scraper.parse_json_ld(soup)
        assert result == {"@type": "SportsEvent", "name": "Test Match"}
    
    def test_parse_json_ld_invalid(self, base_scraper):
        """Test JSON-LD parsing with invalid JSON"""
        html = """
        <html>
            <script type="application/ld+json">
                {"invalid: json"}
            </script>
        </html>
        """
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        result = base_scraper.parse_json_ld(soup)
        assert result is None

class TestHollywoodbetsScraper:
    """Test HollywoodbetsScraper class"""
    
    @pytest.fixture
    def hollywoodbets_scraper(self):
        return HollywoodbetsScraper()
    
    @pytest.fixture
    def mock_html_content(self):
        """Mock HTML content for Hollywoodbets"""
        return """
        <html>
            <div class="event-row">
                <div class="teams">Manchester United vs Liverpool</div>
                <div class="time">15:00</div>
                <div class="league">Premier League</div>
                <div class="odds">
                    <span class="odds-value">2.50</span>
                    <span class="odds-value">3.20</span>
                    <span class="odds-value">2.80</span>
                </div>
            </div>
        </html>
        """
    
    @pytest.mark.asyncio
    async def test_scrape_matches(self, hollywoodbets_scraper, mock_html_content):
        """Test match scraping"""
        with patch.object(hollywoodbets_scraper, 'fetch_page', 
                         AsyncMock(return_value=mock_html_content)):
            matches = await hollywoodbets_scraper.scrape_matches()
            
            assert len(matches) == 1
            match = matches[0]
            
            assert match['home_team'] == 'Manchester United'
            assert match['away_team'] == 'Liverpool'
            assert match['league'] == 'Premier League'
            assert match['source'] == 'hollywoodbets'
            assert 'odds' in match
    
    @pytest.mark.asyncio
    async def test_parse_match(self, hollywoodbets_scraper):
        """Test individual match parsing"""
        from bs4 import BeautifulSoup
        
        html = """
        <div class="event-row">
            <div class="teams">Chelsea vs Arsenal</div>
            <div class="time">18:30</div>
            <div class="league">Premier League</div>
            <div class="odds">
                <span>2.10</span>
                <span>3.40</span>
                <span>3.50</span>
            </div>
        </div>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        container = soup.find('div', class_='event-row')
        
        match_data = await hollywoodbets_scraper._parse_match(container)
        
        assert match_data is not None
        assert match_data['home_team'] == 'Chelsea'
        assert match_data['away_team'] == 'Arsenal'
        assert match_data['league'] == 'Premier League'
    
    def test_parse_odds_value(self, hollywoodbets_scraper):
        """Test odds value parsing"""
        from bs4 import BeautifulSoup
        
        # Test with valid odds
        html = '<span class="odds-value">2.50</span>'
        soup = BeautifulSoup(html, 'html.parser')
        elem = soup.find('span')
        
        result = hollywoodbets_scraper._parse_odds_value(elem)
        assert result == 2.5
        
        # Test with invalid odds
        html = '<span class="odds-value">N/A</span>'
        soup = BeautifulSoup(html, 'html.parser')
        elem = soup.find('span')
        
        result = hollywoodbets_scraper._parse_odds_value(elem)
        assert result is None
        
        # Test with fractional odds
        html = '<span class="odds-value">5/2</span>'
        soup = BeautifulSoup(html, 'html.parser')
        elem = soup.find('span')
        
        result = hollywoodbets_scraper._parse_odds_value(elem)
        assert result == 2.5

class TestBetwayScraper:
    """Test BetwayScraper class"""
    
    @pytest.fixture
    def betway_scraper(self):
        return BetwayScraper()
    
    @pytest.fixture
    def mock_html_content(self):
        """Mock HTML content for Betway"""
        return """
        <html>
            <div class="eventItem">
                <div class="teams">
                    <span class="home">Manchester City</span>
                    <span class="away">Tottenham</span>
                </div>
                <div class="time">20:00</div>
                <div class="league">Premier League</div>
                <div class="odds">
                    <span>1.80</span>
                    <span>3.60</span>
                    <span>4.20</span>
                </div>
            </div>
        </html>
        """
    
    @pytest.mark.asyncio
    async def test_scrape_matches(self, betway_scraper, mock_html_content):
        """Test match scraping"""
        with patch.object(betway_scraper, 'fetch_page', 
                         AsyncMock(return_value=mock_html_content)):
            matches = await betway_scraper.scrape_matches()
            
            assert len(matches) == 1
            match = matches[0]
            
            assert match['home_team'] == 'Manchester City'
            assert match['away_team'] == 'Tottenham'
            assert match['league'] == 'Premier League'
            assert match['source'] == 'betway'
            assert 'odds' in match
    
    @pytest.mark.asyncio
    async def test_parse_betway_event(self, betway_scraper):
        """Test individual event parsing"""
        from bs4 import BeautifulSoup
        
        html = """
        <div class="eventItem">
            <div class="teams">
                <span class="home">Liverpool</span>
                <span class="away">Everton</span>
            </div>
            <div class="time">12:30</div>
            <div class="league">Premier League</div>
            <div class="odds">
                <span>1.50</span>
                <span>4.00</span>
                <span>6.50</span>
            </div>
        </div>
        """
        
        soup = BeautifulSoup(html, 'html.parser')
        container = soup.find('div', class_='eventItem')
        
        match_data = await betway_scraper._parse_betway_event(container)
        
        assert match_data is not None
        assert match_data['home_team'] == 'Liverpool'
        assert match_data['away_team'] == 'Everton'
        assert match_data['league'] == 'Premier League'
    
    def test_parse_odds_value_fractional(self, betway_scraper):
        """Test fractional odds parsing"""
        from bs4 import BeautifulSoup
        
        # Test fractional odds
        html = '<span class="odds">5/2</span>'
        soup = BeautifulSoup(html, 'html.parser')
        elem = soup.find('span')
        
        result = betway_scraper._parse_odds_value(elem)
        assert result == 3.5  # 5/2 + 1 = 2.5 + 1 = 3.5

class TestFootballDataScraper:
    """Test FootballDataScraper class"""
    
    @pytest.fixture
    def football_data_scraper(self):
        return FootballDataScraper()
    
    @pytest.mark.asyncio
    async def test_scrape_football_data(self, football_data_scraper):
        """Test football data scraping"""
        with patch.object(football_data_scraper, '_scrape_team_statistics',
                         AsyncMock(return_value={'Team1': {}})), \
             patch.object(football_data_scraper, '_scrape_injuries',
                         AsyncMock(return_value={})), \
             patch.object(football_data_scraper, '_scrape_transfers',
                         AsyncMock(return_value={})):
            
            data = await football_data_scraper.scrape_football_data(['Team1', 'Team2'])
            
            assert 'injuries' in data
            assert 'transfers' in data
    
    def test_classify_injury_severity(self, football_data_scraper):
        """Test injury severity classification"""
        # Test severe injury
        result = football_data_scraper._classify_injury_severity(
            'Out', 'Serious knee injury, season ending'
        )
        assert result == 'severe'
        
        # Test moderate injury
        result = football_data_scraper._classify_injury_severity(
            'Doubtful', 'Hamstring strain, 2-3 weeks'
        )
        assert result == 'moderate'
        
        # Test minor injury
        result = football_data_scraper._classify_injury_severity(
            'Questionable', 'Minor knock'
        )
        assert result == 'minor'
    
    def test_estimate_return_date(self, football_data_scraper):
        """Test return date estimation"""
        # Test with days
        result = football_data_scraper._estimate_return_date('Out for 7 days')
        assert result is not None
        
        # Test with weeks
        result = football_data_scraper._estimate_return_date('Out for 2 weeks')
        assert result is not None
        
        # Test with unknown duration
        result = football_data_scraper._estimate_return_date('Out indefinitely')
        assert result is None
    
    def test_parse_record(self, football_data_scraper):
        """Test record parsing"""
        result = football_data_scraper._parse_record('12-5-3')
        assert result == (12, 5, 3)
        
        result = football_data_scraper._parse_record('invalid')
        assert result == (0, 0, 0)
    
    def test_parse_goals(self, football_data_scraper):
        """Test goals parsing"""
        result = football_data_scraper._parse_goals('35:20')
        assert result == (35, 20)
        
        result = football_data_scraper._parse_goals('invalid')
        assert result == (0, 0)

@pytest.mark.integration
class TestScraperIntegration:
    """Integration tests for scrapers"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_scraping(self):
        """Test end-to-end scraping workflow"""
        # Initialize scrapers
        hollywoodbets = HollywoodbetsScraper()
        betway = BetwayScraper()
        football_data = FootballDataScraper()
        
        try:
            # Initialize scrapers
            await hollywoodbets.initialize()
            await betway.initialize()
            await football_data.initialize()
            
            # Test initialization
            assert hollywoodbets.browser is not None or hollywoodbets.session is not None
            assert betway.browser is not None or betway.session is not None
            assert football_data.browser is not None or football_data.session is not None
            
            # Note: Actual scraping tests would require mocking or test environments
            # as we don't want to hit real websites in tests
            
        finally:
            # Cleanup
            await hollywoodbets.close()
            await betway.close()
            await football_data.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_scraping(self):
        """Test concurrent scraping"""
        import asyncio
        
        async def mock_scrape(scraper_name):
            await asyncio.sleep(0.1)  # Simulate scraping delay
            return [{'home_team': f'TeamA_{scraper_name}', 
                    'away_team': f'TeamB_{scraper_name}'}]
        
        # Create mock scrapers
        scrapers = {
            'hollywoodbets': Mock(scrape_matches=lambda: mock_scrape('hollywoodbets')),
            'betway': Mock(scrape_matches=lambda: mock_scrape('betway'))
        }
        
        # Run concurrent scraping
        tasks = [scrapers[name].scrape_matches() for name in scrapers]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1

class TestDataValidation:
    """Test data validation functions"""
    
    def test_match_data_validation(self):
        """Test match data validation"""
        from utils.validators import validate_match_data
        
        # Valid match data
        valid_match = {
            'home_team': 'Manchester United',
            'away_team': 'Liverpool',
            'match_date': datetime.now().isoformat(),
            'league': 'Premier League',
            'home_odds': 2.5,
            'draw_odds': 3.2,
            'away_odds': 2.8
        }
        
        is_valid, errors = validate_match_data(valid_match)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid match data
        invalid_match = {
            'home_team': '',  # Empty team name
            'away_team': 'Liverpool',
            'match_date': 'invalid-date'
        }
        
        is_valid, errors = validate_match_data(invalid_match)
        assert not is_valid
        assert len(errors) > 0
    
    def test_odds_validation(self):
        """Test odds validation"""
        from utils.validators import validate_odds_data
        
        # Valid odds
        valid_odds = {
            'home_win': 2.5,
            'draw': 3.2,
            'away_win': 2.8
        }
        
        is_valid, errors = validate_odds_data(valid_odds)
        assert is_valid
        assert len(errors) == 0
        
        # Invalid odds
        invalid_odds = {
            'home_win': 0.5,  # Less than 1.0
            'draw': 1001.0,   # More than 1000
            'away_win': 'invalid'
        }
        
        is_valid, errors = validate_odds_data(invalid_odds)
        assert not is_valid
        assert len(errors) > 0

# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
