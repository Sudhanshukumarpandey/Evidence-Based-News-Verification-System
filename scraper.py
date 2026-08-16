import requests
from bs4 import BeautifulSoup

def scrape_article(url):
    # Standard desktop User-Agent header to avoid basic crawler blocks from cloudflare/web servers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        # 5-second timeout keeps response times fast and avoids hanging the Streamlit thread
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch webpage: {e}")
        
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Try to extract the h1 title, fallback to HTML title tag if h1 doesn't exist
    title_el = soup.find("h1")
    title = title_el.get_text().strip() if title_el else ""
    if not title and soup.title:
        title = soup.title.get_text().strip()
        
    # Extract paragraphs and filter out empty blocks/boilerplate text
    paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
    text = " ".join([p for p in paragraphs if p])
    
    return title, text
