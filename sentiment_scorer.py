"""Score stored articles with VADER and print a ticker-level summary."""

from __future__ import annotations

import logging

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sentiment_utils import get_connection, sentiment_label, sentiment_summary

logger = logging.getLogger(__name__)

_analyzer = SentimentIntensityAnalyzer()

MIN_TEXT_LENGTH = 5


def score_text(text: str | None) -> float:
    """Return the VADER compound score (-1 to 1) for ``text``.

    Empty or very short text scores 0.0 (neutral).
    """
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return 0.0
    return round(_analyzer.polarity_scores(text)["compound"], 4)


def score_all_unscored() -> int:
    """Score every article whose ``sentiment`` is NULL. Returns the count scored."""
    with get_connection() as conn:
        unscored = pd.read_sql(
            "SELECT id, ticker, headline, body FROM articles WHERE sentiment IS NULL",
            conn,
        )
        if unscored.empty:
            logger.info("No unscored articles found.")
            return 0

        logger.info("Scoring %d articles...", len(unscored))
        updates = []
        for row in unscored.itertuples(index=False):
            headline = row.headline if isinstance(row.headline, str) else ""
            body = row.body if isinstance(row.body, str) else ""
            score = score_text(f"{headline} {body}".strip())
            updates.append((score, row.id))
            logger.debug("%s | %+.4f | %s", row.ticker, score, headline[:60])

        conn.executemany("UPDATE articles SET sentiment = ? WHERE id = ?", updates)
    logger.info("Done scoring.")
    return len(updates)


def print_summary() -> None:
    """Print average sentiment and 7-day momentum for each ticker."""
    summary = sentiment_summary()
    if summary.empty:
        print("No scored articles yet.")
        return

    print("\n--- Sentiment summary (descriptive labels, not trading signals) ---")
    for row in summary.itertuples(index=False):
        if pd.notna(row.today_sentiment) and pd.notna(row.week_sentiment):
            momentum = f"{row.today_sentiment - row.week_sentiment:+.4f}"
        else:
            momentum = "N/A"
        print(
            f"\nTicker:              {row.ticker}"
            f"\nAvg sentiment:       {row.avg_sentiment:+.4f}"
            f"\nArticles:            {row.article_count}"
            f"\nLabel:               {sentiment_label(row.avg_sentiment)}"
            f"\nToday vs 7-day avg:  {momentum}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    score_all_unscored()
    print_summary()
