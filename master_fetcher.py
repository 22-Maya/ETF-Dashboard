"""Collect news, SEC filings and latest prices for the configured holdings.

Sources (each is optional and skipped when not configured):
    * NewsAPI        - needs NEWSAPI_KEY
    * Alpha Vantage  - needs ALPHA_VANTAGE_KEY
    * SEC EDGAR      - needs a CIK in etf_holdings.json and SEC_USER_AGENT
    * Yahoo Finance  - latest closes via yfinance (no key)

Everything is stored in the local SQLite database (``etf_news.db``).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from newsapi import NewsApiClient

from sentiment_utils import BASE_DIR, DB_PATH, get_connection

load_dotenv()
logger = logging.getLogger(__name__)

CONFIG_PATH = BASE_DIR / "etf_holdings.json"
REQUEST_TIMEOUT = 20  # seconds, applied to every HTTP request
NEWS_DAYS_BACK = 3
SEC_FILING_TYPE = "8-K"
SEC_FILINGS_PER_TICKER = 5
SEC_EXCERPT_CHARS = 1000
SEC_PAUSE_SECONDS = 0.5  # SEC asks for <= 10 requests per second
PER_TICKER_PAUSE_SECONDS = 1.0

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_MAX_ARTICLES = 25  # newest articles kept per ticker per run
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")


class RateLimited(Exception):
    """Raised when a data provider reports that its request quota is used up."""


# ── Config and database ─────────────────────────────────────────────────────


def load_holdings(config_path: Path | str = CONFIG_PATH) -> dict[str, dict]:
    """Return ``{ticker: {"keywords": [...], "cik": "..."}}`` from the config."""
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)["holdings"]


def setup_db() -> None:
    """Create the tables used by the pipeline if they do not exist."""
    with get_connection() as conn:
        conn.execute(
            """
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
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                ticker     TEXT,
                date       TEXT,
                close      REAL,
                pct_change REAL,
                PRIMARY KEY (ticker, date)
            )
            """
        )
    logger.info("Database ready: %s", DB_PATH.name)


def save_article(
    conn: sqlite3.Connection,
    ticker: str,
    headline: str | None,
    source: str | None,
    date: str | None,
    url: str | None,
    body: str | None,
) -> int:
    """Insert one article; returns 1 if it was new and 0 if it was a duplicate."""
    if not headline:
        return 0
    try:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO articles (ticker, headline, source, date, url, body)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticker, headline, source, date, url, body or ""),
        )
        return cursor.rowcount
    except sqlite3.Error as exc:
        logger.warning("Could not save article for %s: %s", ticker, exc)
        return 0


# ── Sources ─────────────────────────────────────────────────────────────────


def fetch_newsapi(
    client: NewsApiClient, ticker: str, keywords: list[str], days_back: int = NEWS_DAYS_BACK
) -> int:
    """Fetch recent NewsAPI articles for a ticker. Returns the number saved."""
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    try:
        response = client.get_everything(
            q=" OR ".join(keywords),
            from_param=from_date,
            language="en",
            sort_by="publishedAt",
            page_size=20,
        )
    except Exception as exc:  # the client raises several provider-specific errors
        logger.warning("NewsAPI error for %s: %s", ticker, exc)
        return 0

    saved = 0
    with get_connection() as conn:
        for item in response.get("articles", []):
            saved += save_article(
                conn,
                ticker,
                item.get("title"),
                (item.get("source") or {}).get("name"),
                (item.get("publishedAt") or "")[:10],
                item.get("url"),
                item.get("description"),
            )
    return saved

def fetch_finnhub(ticker: str, days_back: int = NEWS_DAYS_BACK) -> int:
    """Fetch recent Finnhub company news. Raises RateLimited on HTTP 429."""
    now = datetime.now()
    try:
        response = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker,
                "from": (now - timedelta(days=days_back)).strftime("%Y-%m-%d"),
                "to": now.strftime("%Y-%m-%d"),
            },
            headers={"X-Finnhub-Token": FINNHUB_API_KEY},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        logger.warning("Finnhub error for %s: %s", ticker, exc)
        return 0

    if response.status_code == 429:
        raise RateLimited("Finnhub rate limit reached")
    if response.status_code != 200:
        logger.warning("Finnhub returned HTTP %s for %s", response.status_code, ticker)
        return 0
    try:
        items = response.json()
    except ValueError:
        return 0
    if not isinstance(items, list):
        return 0

    saved = 0
    with get_connection() as conn:
        for item in items[:FINNHUB_MAX_ARTICLES]:
            timestamp = item.get("datetime")  # Unix seconds
            date = (
                datetime.fromtimestamp(timestamp, tz=UTC).strftime("%Y-%m-%d")
                if timestamp
                else None
            )
            saved += save_article(
                conn, ticker, item.get("headline"), item.get("source"),
                date, item.get("url"), item.get("summary"),
            )
    return saved

def fetch_sec(
    ticker: str,
    cik: str,
    filing_type: str = SEC_FILING_TYPE,
    count: int = SEC_FILINGS_PER_TICKER,
) -> int:
    """Fetch recent SEC filings of one type and store a text excerpt of each."""
    headers = {"User-Agent": SEC_USER_AGENT}
    cik_plain = str(cik).lstrip("0")

    try:
        response = requests.get(
            f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json",
            headers=headers,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code != 200:
            logger.warning("SEC returned HTTP %s for %s", response.status_code, ticker)
            return 0
        recent = response.json().get("filings", {}).get("recent", {})
    except (requests.RequestException, ValueError) as exc:
        logger.warning("SEC error for %s: %s", ticker, exc)
        return 0

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    documents = recent.get("primaryDocument", [])

    saved = 0
    with get_connection() as conn:
        for i, form in enumerate(forms):
            if form != filing_type:
                continue
            if saved >= count:
                break
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_plain}/"
                f"{accessions[i].replace('-', '')}/{documents[i]}"
            )
            try:
                page = requests.get(doc_url, headers=headers, timeout=REQUEST_TIMEOUT)
                text = BeautifulSoup(page.text, "html.parser").get_text(separator=" ", strip=True)[
                    :SEC_EXCERPT_CHARS
                ]
            except requests.RequestException as exc:
                logger.warning("SEC filing error for %s: %s", ticker, exc)
                continue
            saved += save_article(
                conn,
                ticker,
                f"{ticker} {filing_type} filing — {dates[i]}",
                "SEC EDGAR",
                dates[i],
                doc_url,
                text,
            )
            time.sleep(SEC_PAUSE_SECONDS)
    return saved


def update_prices(tickers: list[str], period: str = "10d") -> int:
    """Store recent closes and daily % change so the dashboard can show prices."""
    try:
        data = yf.download(
            tickers, period=period, auto_adjust=True, progress=False, group_by="column"
        )
    except Exception as exc:
        logger.warning("Price download failed: %s", exc)
        return 0
    if data.empty:
        logger.warning("No price data returned by yfinance.")
        return 0

    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if not isinstance(data.columns, pd.MultiIndex):
        close.columns = tickers[:1]
    pct = close.pct_change() * 100.0

    rows = []
    for ticker in close.columns:
        for date, value in close[ticker].dropna().items():
            change = pct.at[date, ticker]
            rows.append(
                (
                    ticker,
                    date.strftime("%Y-%m-%d"),
                    float(value),
                    None if pd.isna(change) else float(change),
                )
            )

    with get_connection() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO prices (ticker, date, close, pct_change) VALUES (?, ?, ?, ?)",
            rows,
        )
    return len(rows)


# ── Reporting and entry point ───────────────────────────────────────────────


def print_summary() -> None:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT ticker, COUNT(*) AS total,
                   SUM(CASE WHEN sentiment IS NULL THEN 1 ELSE 0 END) AS unscored
            FROM articles
            GROUP BY ticker
            ORDER BY total DESC
            """
        ).fetchall()

    print("\n--- Database summary ---")
    print(f"{'Ticker':<8} {'Total':>6} {'Unscored':>9}")
    print("-" * 26)
    for ticker, total, unscored in rows:
        print(f"{ticker:<8} {total:>6} {unscored:>9}")


def main() -> None:
    setup_db()
    holdings = load_holdings()
    logger.info("Fetching data for %d tickers", len(holdings))

    news_client = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None
    use_alpha_vantage = bool(ALPHA_VANTAGE_KEY)
    if news_client is None:
        logger.info("NEWSAPI_KEY not set - skipping NewsAPI.")
    if not use_alpha_vantage:
        logger.info("ALPHA_VANTAGE_KEY not set - skipping Alpha Vantage.")

    use_sec = bool(SEC_USER_AGENT) and "example.com" not in SEC_USER_AGENT
    if not use_sec:
        logger.info(
            "SEC_USER_AGENT not set to a real contact - skipping SEC EDGAR. "
            "SEC requires a descriptive user agent with a contact email."
        )

    for ticker, info in holdings.items():
        logger.info("[%s]", ticker)

        if news_client:
            logger.info(
                "  NewsAPI:       +%d articles",
                fetch_newsapi(news_client, ticker, info["keywords"]),
            )

        if use_finnhub:
		try:
                	logger.info("  Finnhub:       +%d articles", fetch_finnhub(ticker))
            	except RateLimited as exc:
                	logger.warning("  Finnhub rate limit reached; skipping it for the "
                               	"rest of this run (%s)", exc)
                	use_finnhub = False

        cik = info.get("cik")
        if use_sec and cik:
            logger.info("  SEC EDGAR:     +%d filings", fetch_sec(ticker, cik))

        time.sleep(PER_TICKER_PAUSE_SECONDS)

    logger.info("Prices: %d rows stored", update_prices(list(holdings)))
    print_summary()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
