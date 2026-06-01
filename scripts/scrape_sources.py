import os
import requests
import json
from pathlib import Path

def scrape():
    """Fetches source documents from Wikipedia and saves them to data/raw/papers."""
    print("Initializing Wikipedia scraper...")
    
    # Target topics
    topics = [
        "Alan Turing",
        "Turing machine",
        "Turing test",
        "Cryptanalysis of the Enigma",
        "Automatic Computing Engine",
        "Bletchley Park",
        "Morphogenesis"
    ]
    
    # Ensure directory exists
    output_dir = Path(__file__).parent.parent / "data" / "raw" / "papers"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Wikipedia API endpoint for plain text extracts
    base_url = "https://en.wikipedia.org/w/api.php"
    
    for topic in topics:
        print(f"Fetching '{topic}'...")
        params = {
            "action": "query",
            "format": "json",
            "titles": topic,
            "prop": "extracts",
            "explaintext": True,
            "redirects": 1
        }
        
        try:
            headers = {"User-Agent": "DigitalTwinTuringBot/1.0 (contact@example.com)"}
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Extract the text
            pages = data["query"]["pages"]
            page_id = list(pages.keys())[0]
            
            if page_id == "-1":
                print(f"  [Error] Page not found: {topic}")
                continue
                
            extract = pages[page_id].get("extract", "")
            if not extract:
                print(f"  [Error] No content found for: {topic}")
                continue
                
            # Save to file following the convention: YEAR - Title.txt
            # Using 2024 as the year for recent encyclopedic knowledge
            filename = f"2024 - Wikipedia - {topic.replace('/', '_')}.txt"
            filepath = output_dir / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(extract)
                
            print(f"  [Success] Saved to {filename} ({len(extract)} chars)")
            
        except Exception as e:
            print(f"  [Error] Error fetching {topic}: {e}")

if __name__ == "__main__":
    scrape()
