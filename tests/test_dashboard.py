import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import master_fetcher as mf  # noqa: E402
import sentiment_scorer as scorer  # noqa: E402
import sentiment_utils as su  # noqa: E402

APP = str(su.BASE_DIR / "dashboard.py")


@pytest.fixture(autouse=True)
def fresh_cache():
    import streamlit as st

    st.cache_data.clear()
    yield
    st.cache_data.clear()


def test_dashboard_without_database_shows_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(su, "DB_PATH", tmp_path / "missing.db")
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert any("No news database found" in i.value for i in app.info)


def test_dashboard_with_data_renders_without_exceptions(tmp_path, monkeypatch):
    monkeypatch.setattr(su, "DB_PATH", tmp_path / "news.db")
    mf.setup_db()
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    with su.get_connection() as conn:
        for ticker in ("HON", "MSI", "AXON"):
            mf.save_article(
                conn, ticker, f"{ticker} posts strong growth", "wire", today, "u", "great"
            )
            mf.save_article(
                conn, ticker, f"{ticker} faces lawsuit", "wire", today, "u2", "terrible"
            )
        conn.execute("INSERT INTO prices VALUES ('HON', ?, 210.5, 1.25)", (today,))
    scorer.score_all_unscored()

    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert len(app.dataframe) >= 1  # summary table rendered
    assert any("Research prototype" in i.value for i in app.info)
