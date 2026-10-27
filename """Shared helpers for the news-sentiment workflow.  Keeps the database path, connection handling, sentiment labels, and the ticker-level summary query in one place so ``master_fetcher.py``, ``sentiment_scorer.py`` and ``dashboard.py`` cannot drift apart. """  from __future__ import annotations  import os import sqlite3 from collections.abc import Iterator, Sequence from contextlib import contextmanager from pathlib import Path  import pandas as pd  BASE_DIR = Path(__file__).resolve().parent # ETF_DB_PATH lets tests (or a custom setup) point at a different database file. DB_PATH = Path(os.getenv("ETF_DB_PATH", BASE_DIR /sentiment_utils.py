"""Shared helpers for the news-sentiment workflow.

Keeps the database path, connection handling, sentiment labels, and the
ticker-level summary query in one place so ``master_fetcher.py``,
``sentiment_scorer.py`` and ``dashboard.py`` cannot drift apart.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
# ETF_DB_PATH lets tests (or a custom setup) point at a different database file.
DB_PATH = Path(os.getenv("ETF_DB_PATH", BASE_DIR / "etf_news.db"))

# VADER compound-score cut-offs used to bucket average sentiment.
# These are descriptive rules of thumb, not a validated trading signal.
STRONG_THRESHOLD = 0.5
MILD_THRESHOLD = 0.2

SENTIMENT_LABELS = {
    "very_positive": "Very positive",
    "positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative",
    "very_negative": "Very negative",
    "no_data": "No data",
}

SENTIMENT_EMOJI = {
    "Very positive": "🟢",
    "Positive": "🟩",
    "Neutral": "🟡",
    "Negative": "🟠",
    "Very negative": "🔴",
    "No data": "⬜",
}


@contextmanager
def get_connection(db_path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection that is always committed and closed.

    ``db_path`` defaults to the module-level ``DB_PATH`` (looked up at call time
    so tests can point it at a temporary database).
    """
    conn = sqlite3.connect(db_path or DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def db_has_articles(db_path: Path | str | None = None) -> bool:
    """Return True if the database file exists and has an ``articles`` table."""
    if not Path(db_path or DB_PATH).exists():
        return False
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='articles'"
        ).fetchone()
    return row is not None


def sentiment_label(score: float | None) -> str:
    """Map an average VADER compound score to a descriptive label."""
    if score is None or pd.isna(score):
        return SENTIMENT_LABELS["no_data"]
    if score > STRONG_THRESHOLD:
        return SENTIMENT_LABELS["very_positive"]
    if score > MILD_THRESHOLD:
        return SENTIMENT_LABELS["positive"]
    if score > -MILD_THRESHOLD:
        return SENTIMENT_LABELS["neutral"]
    if score > -STRONG_THRESHOLD:
        return SENTIMENT_LABELS["negative"]
    return SENTIMENT_LABELS["very_negative"]


def sentiment_summary(
    tickers: Sequence[str] | None = None,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    """Ticker-level sentiment: overall, today, and trailing 7-day averages."""
    query = """
        SELECT
            ticker,
            ROUND(AVG(sentiment), 4) AS avg_sentiment,
            COUNT(*)                 AS article_count,
            ROUND(AVG(CASE WHEN date = DATE('now')
                      THEN sentiment END), 4) AS today_sentiment,
            ROUND(AVG(CASE WHEN date >= DATE('now', '-7 days')
                      THEN sentiment END), 4) AS week_sentiment
        FROM articles
        WHERE sentiment IS NOT NULL
    """
    params: list = []
    if tickers:
        query += f" AND ticker IN ({','.join('?' * len(tickers))})"
        params = list(tickers)
    query += " GROUP BY ticker ORDER BY avg_sentiment DESC"

    with get_connection(db_path) as conn:
        return pd.read_sql(query, conn, params=params)
