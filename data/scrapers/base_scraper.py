import asyncio
import aiohttp
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import json
import time

logger = logging.getLogger(__name__)

class BaseScraper(ABC):
    def __init__(self, use_playwright: bool = True):
        self.use_playwright = use_playwright
        self.session = None
        self.playwright = None
        self.browser = None
        self.context = None
        
    async def initialize(self):
        """Initialize scraping session"""
        if self.use_playwright:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            self.context = await self.browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
        else:
            self.session = aiohttp.ClientSession()
            
    async def close(self):
        """Close all connections"""
        if self.session:
            await self.session.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    @abstractmethod
    async def scrape_matches(self) -> List[Dict]:
        """Scrape today's matches"""
        pass
    
    @abstractmethod
    async def scrape_odds(self, match_url: str) -> Dict:
        """Scrape odds for a specific match"""
        pass
    
    async def fetch_page(self, url: str, use_playwright: bool = None) -> str:
        """Fetch page content"""
        use_playwright = use_playwright if use_playwright is not None else self.use_playwright
        
        if use_playwright:
            page = await self.context.new_page()
            try:
                await page.goto(url, wait_until="networkidle")
                content = await page.content()
                return content
            finally:
                await page.close()
        else:
            async with self.session.get(url) as response:
                return await response.text()
    
    def parse_json_ld(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Parse JSON-LD structured data"""
        script = soup.find('script', type='application/ld+json')
        if script:
            try:
                return json.loads(script.string)
            except json.JSONDecodeError:
                pass
        return None
