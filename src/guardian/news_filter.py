import requests
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict

class NewsFilter:
    def __init__(self, api_key: str = None):
        self.logger = logging.getLogger("NewsFilter")
        self.api_key = api_key
        self.events = []
        self.blackout_minutes_before = 5
        self.blackout_minutes_after = 5
        
        if self.api_key:
            self.refresh_events()

    def refresh_events(self):
        """Fetches latest economic calendar from Finnhub."""
        if not self.api_key:
            return

        url = "https://finnhub.io/api/v1/calendar/economic"
        params = {
            'token': self.api_key,
            'from': datetime.now().strftime("%Y-%m-%d"),
            'to': (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'economicCalendar' in data:
                self.events = []
                for item in data['economicCalendar']:
                    if item['impact'] in ['high', 'medium']: # Filter for impact
                        self.events.append({
                            "time": item['time'], # Format: "2023-10-27 14:30:00"
                            "currency": item['currency'],
                            "impact": item['impact'].upper(),
                            "event": item['event']
                        })
                self.logger.info(f"Refreshed News Events: {len(self.events)} high/med impact events found.")
                
        except Exception as e:
            self.logger.error(f"Failed to fetch news from Finnhub: {e}")

    def is_news_imminent(self, symbol: str, current_time: datetime) -> bool:
        """
        Checks if high-impact news is imminent for the symbol's currency.
        """
        # Extract currencies from symbol (e.g., EURUSD -> EUR, USD)
        # Simple heuristic for now
        currencies = []
        if len(symbol) == 6:
            currencies = [symbol[:3], symbol[3:]]
        elif "USD" in symbol or "NAS" in symbol or "US30" in symbol:
            currencies = ["USD"]
        
        for event in self.events:
            if event['currency'] in currencies and event['impact'] == 'HIGH':
                try:
                    event_time = datetime.strptime(event['time'], "%Y-%m-%d %H:%M:%S")
                    
                    # Check window
                    start_block = event_time - timedelta(minutes=self.blackout_minutes_before)
                    end_block = event_time + timedelta(minutes=self.blackout_minutes_after)
                    
                    if start_block <= current_time <= end_block:
                        self.logger.warning(f"News Blackout: {event['event']} ({event['currency']}) at {event['time']}")
                        return True
                except ValueError:
                    continue # Skip if time format parsing fails
                    
        return False
