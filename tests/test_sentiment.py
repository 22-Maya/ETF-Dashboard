import sqlite3

import pandas as pd
import pytest

import master_fetcher as mf
import sentiment_scorer as scorer
import sentiment_utils as su


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Point every module at a throwaway SQLite file."""
    path = tmp_path / "test_news.db"
    monkeypatch.setattr(su, "DB_PATH", path)
    mf.setup_db()
    return path


def insert(db, rows):
    with su.get_connection() as conn:
        for row in rows:
            mf.save_article(conn, *row)


@pytest.mark.parametrize(
    "score,label",
    [
        (0.8, "Very positive"),
        (0.3, "Positive"),
        (0.0, "Neutral"),
        (-0.3, "Negative"),
        (-0.8, "Very negative"),
        (None, "No data"),
        (float("nan"), "No data"),
    ],
)
def test_sentiment_label(score, label):
    assert su.sentiment_label(score) == label


def test_score_text_edge_cases():
    assert scorer.score_text(None) == 0.0
    assert scorer.score_text("   ") == 0.0
    assert scorer.score_text("Great results, excellent growth and strong demand") > 0.3
    assert scorer.score_text("Terrible losses, fraud investigation and lawsuit") < -0.3


def test_setup_creates_tables_and_is_idempotent(db):
    mf.setup_db()
    with su.get_connection() as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"articles", "prices"} <= tables
    assert su.db_has_articles()


def test_db_has_articles_false_without_file(tmp_path, monkeypatch):
    monkeypatch.setattr(su, "DB_PATH", tmp_path / "missing.db")
    assert not su.db_has_articles()


def test_save_article_skips_duplicates_and_missing_headlines(db):
    with su.get_connection() as conn:
        assert mf.save_article(conn, "HON", "Headline", "src", "2026-01-01", "u", "body") == 1
        assert mf.save_article(conn, "HON", "Headline", "src", "2026-01-01", "u", "body") == 0
        assert mf.save_article(conn, "HON", None, "src", "2026-01-01", "u", "body") == 0
        assert mf.save_article(conn, "HON", "No body", "src", "2026-01-02", "u", None) == 1


def test_scoring_pipeline_and_summary(db):
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    insert(
        db,
        [
            ("AAA", "Record profits and excellent growth", "s", today, "u1", "Strong demand"),
            ("AAA", "Terrible quarter, fraud probe", "s", today, "u2", None),
            ("BBB", "Company announces update", "s", today, "u3", ""),
        ],
    )
    assert scorer.score_all_unscored() == 3
    assert scorer.score_all_unscored() == 0  # nothing left to score

    summary = su.sentiment_summary().set_index("ticker")
    assert set(summary.index) == {"AAA", "BBB"}
    assert summary.loc["AAA", "article_count"] == 2
    assert summary.loc["AAA", "today_sentiment"] == summary.loc["AAA", "avg_sentiment"]

    only_bbb = su.sentiment_summary(["BBB"])
    assert list(only_bbb["ticker"]) == ["BBB"]


def test_scoring_handles_null_headline_and_body(db):
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO articles (ticker, headline, body) VALUES ('X', NULL, NULL)")
    assert scorer.score_all_unscored() == 1


def test_load_holdings_matches_portfolio_csv():
    holdings = mf.load_holdings()
    tickers = set(pd.read_csv(mf.BASE_DIR / "holdings.csv")["ticker"])
    assert set(holdings) == tickers
    assert all(info["keywords"] for info in holdings.values())


def test_update_prices_stores_close_and_pct_change(db, monkeypatch):
    idx = pd.bdate_range("2026-01-05", periods=3)
    cols = pd.MultiIndex.from_product([["Close"], ["AAA", "BBB"]])
    fake = pd.DataFrame([[10.0, 20.0], [11.0, 19.0], [12.1, 19.0]], index=idx, columns=cols)
    monkeypatch.setattr(mf.yf, "download", lambda *a, **k: fake)

    assert mf.update_prices(["AAA", "BBB"]) == 6
    with su.get_connection() as conn:
        rows = pd.read_sql("SELECT * FROM prices WHERE date = '2026-01-07'", conn).set_index(
            "ticker"
        )
    assert rows.loc["AAA", "close"] == pytest.approx(12.1)
    assert rows.loc["AAA", "pct_change"] == pytest.approx(10.0)
    assert rows.loc["BBB", "pct_change"] == pytest.approx(0.0)

    # Re-running replaces rows instead of duplicating them
    assert mf.update_prices(["AAA", "BBB"]) == 6
    with su.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 6


def test_update_prices_handles_empty_download(db, monkeypatch):
    monkeypatch.setattr(mf.yf, "download", lambda *a, **k: pd.DataFrame())
    assert mf.update_prices(["AAA"]) == 0
