import requests
import sqlite3
import json
import os
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
from dotenv import load_dotenv

load_dotenv()

newsapi    = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
AV_KEY     = os.getenv("ALPHA_VANTAGE_KEY")
HEADERS    = {"User-Agent": "etf-sentiment-bot contact@youremail.com"}

def load_holdings(config_path="etf_holdings.json"):
    with open(config_path, "r") as f:
        return json.load(f)["holdings"]

def setup_db():
    conn = sqlite3.connect("etf_news.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker    TEXT,
            headline  TEXT,
            source    TEXT,
            date      TEXT,
            url       TEXT,
            body      TEXT,
            sentiment REAL DEFAULT NULL,
            UNIQUE(ticker, headline)
        )
    """)
    conn.commit()
    conn.close()
    print("Database ready.")

def save_article(conn, ticker, headline, source, date, url, body):
    try:
        conn.execute("""
            INSERT OR IGNORE INTO articles
            (ticker, headline, source, date, url, body)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticker, headline, source, date, url, body or ""))
        return conn.execute("SELECT changes()").fetchone()[0]
    except Exception as e:
        print(f"    Save error: {e}")
        return 0

def fetch_newsapi(ticker, keywords, days_back=3):
    query     = " OR ".join(keywords)
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    conn      = sqlite3.connect("etf_news.db")
    saved     = 0

    try:
        response = newsapi.get_everything(
            q=query,
            from_param=from_date,
            language="en",
            sort_by="publishedAt",
            page_size=20,
        )
        for item in response.get("articles", []):
            saved += save_article(
                conn,
                ticker,
                item["title"],
                item["source"]["name"],
                item["publishedAt"][:10],
                item["url"],
                item.get("description", ""),
            )
    except Exception as e:
        print(f"    NewsAPI error: {e}")

    conn.commit()
    conn.close()
    return saved

def fetch_alphavantage(ticker):
    conn = sqlite3.connect("etf_news.db")
    saved = 0

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "tickers":  ticker,
                "limit":    50,
                "apikey":   AV_KEY,
            }
        )
        data = response.json()
        for item in data.get("feed", []):
            date_raw = item["time_published"][:8]
            date_fmt = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:]}"
            saved += save_article(
                conn,
                ticker,
                item["title"],
                item["source"],
                date_fmt,
                item["url"],
                item.get("summary", ""),
            )
    except Exception as e:
        print(f"    Alpha Vantage error: {e}")

    conn.commit()
    conn.close()
    return saved

def fetch_sec(ticker, cik, filing_type="8-K", count=5):
    cik_clean = str(cik).lstrip("0")
    conn      = sqlite3.connect("etf_news.db")
    saved     = 0

    try:
        response = requests.get(
            f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json",
            headers=HEADERS
        )
        if response.status_code != 200:
            return 0

        filings = response.json().get("filings", {}).get("recent", {})
        forms   = filings.get("form", [])
        dates   = filings.get("filingDate", [])
        accnums = filings.get("accessionNumber", [])
        docs    = filings.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form != filing_type:
                continue
            if saved >= count:
                break

            accnum  = accnums[i].replace("-", "")
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accnum}/{docs[i]}"

            try:
                soup    = BeautifulSoup(
                    requests.get(doc_url, headers=HEADERS).text,
                    "html.parser"
                )
                excerpt = soup.get_text(separator=" ", strip=True)[:1000]
                saved  += save_article(
                    conn,
                    ticker,
                    f"{ticker} {filing_type} filing — {dates[i]}",
                    "SEC EDGAR",
                    dates[i],
                    doc_url,
                    excerpt,
                )
                time.sleep(0.5)
            except Exception as e:
                print(f"    SEC filing error: {e}")

    except Exception as e:
        print(f"    SEC error: {e}")

    conn.commit()
    conn.close()
    return saved

def print_summary():
    conn = sqlite3.connect("etf_news.db")
    rows = conn.execute("""
        SELECT ticker, COUNT(*) as total,
               SUM(CASE WHEN sentiment IS NULL THEN 1 ELSE 0 END) as unscored
        FROM articles
        GROUP BY ticker
        ORDER BY total DESC
    """).fetchall()
    conn.close()

    print("\n--- Database summary ---")
    print(f"{'Ticker':<8} {'Total':>6} {'Unscored':>9}")
    print("-" * 26)
    for row in rows:
        print(f"{row[0]:<8} {row[1]:>6} {row[2]:>9}")

if __name__ == "__main__":
    setup_db()
    holdings = load_holdings()
    print(f"\nFetching data for: {list(holdings.keys())}\n")

    for ticker, info in holdings.items():
        keywords = info["keywords"]
        cik      = info.get("cik")
        print(f"[{ticker}]")

        n = fetch_newsapi(ticker, keywords)
        print(f"  NewsAPI:       +{n} articles")

        a = fetch_alphavantage(ticker)
        print(f"  Alpha Vantage: +{a} articles")

        if cik:
            s = fetch_sec(ticker, cik)
            print(f"  SEC EDGAR:     +{s} filings")
        else:
            print(f"  SEC EDGAR:     skipped (no CIK)")

        time.sleep(1)

    print_summary()
    print("\nAll done. Run sentiment_scorer.py to score new articles.")