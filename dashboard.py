"""Streamlit dashboard: ticker-level news sentiment and portfolio performance charts.

Run with ``streamlit run dashboard.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sentiment_utils import (
    MILD_THRESHOLD,
    SENTIMENT_EMOJI,
    db_has_articles,
    get_connection,
    sentiment_label,
    sentiment_summary,
)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "etf_holdings.json"
HOLDINGS_CSV = BASE_DIR / "holdings.csv"
OUTPUT_DIR = BASE_DIR / "etf_output"

CACHE_SECONDS = 300
CATEGORY_COLUMNS = 4

DISCLAIMER = (
    "Research prototype, not investment advice. Sentiment labels summarise VADER scores of "
    "news text. They are descriptive and have not been validated as predictors of returns. "
    "Portfolio charts are a historical simulation of a configured portfolio, not a live fund."
)

# (file in etf_output/, heading, caption)
CHARTS = [
    (
        "01_indexed_performance.png",
        "Indexed performance vs benchmarks",
        "Portfolio and benchmarks rebased to 100 at the start of the backtest window.",
    ),
    (
        "02_relative_strength.png",
        "Relative strength vs each benchmark",
        "A ratio above 1.0 means the portfolio is ahead of that benchmark since the window start.",
    ),
    (
        "03_volatility_bar.png",
        "Annualized volatility",
        "Volatility of daily returns over the trailing window. Lower means less volatile.",
    ),
    (
        "04_drawdown_comparison.png",
        "Drawdown comparison",
        "Decline from the previous peak at each point in time.",
    ),
    (
        "05_rolling_beta_vs_spy.png",
        "Rolling beta vs SPY",
        "Beta above 1.0 means the portfolio has moved more than SPY.",
    ),
    (
        "06_rolling_correlation_vs_spy.png",
        "Rolling correlation vs SPY",
        "How closely daily portfolio returns move with SPY.",
    ),
    (
        "07_rolling_beta_vs_lock.png",
        "Rolling beta vs LOCK.L",
        "Beta relative to the L&G Cyber Security ETF (LOCK.L).",
    ),
    (
        "08_rolling_correlation_vs_lock.png",
        "Rolling correlation vs LOCK.L",
        "How closely daily portfolio returns move with LOCK.L.",
    ),
    (
        "09_concentration.png",
        "Concentration by theme and top holdings",
        "Target weight by theme and the ten largest holdings.",
    ),
    (
        "10_performance_attribution.png",
        "Performance attribution",
        "Each holding's contribution to total return, computed from the rebalanced simulation.",
    ),
    (
        "11_holdings_overlap.png",
        "Holdings overlap vs peer ETFs",
        "Share of portfolio holdings that also appear in each peer ETF's holdings list.",
    ),
    (
        "12_metrics_table.png",
        "Risk-adjusted metrics",
        "CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, alpha and beta.",
    ),
    ("13_sector_exposure_donut.png", "Sector exposure", "Target portfolio weight by sector."),
]


# ── Data loading ────────────────────────────────────────────────────────────


@st.cache_data
def load_config() -> dict:
    """ETF universe from etf_holdings.json, with a de-duplicated ticker list per ETF."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        etfs = json.load(f)["etfs"]
    for etf in etfs.values():
        tickers = [t for group in etf["categories"].values() for t in group]
        etf["tickers"] = list(dict.fromkeys(tickers))
    return etfs


@st.cache_data
def load_target_weights() -> pd.Series:
    """Target portfolio weights (in %) from holdings.csv."""
    df = pd.read_csv(HOLDINGS_CSV).drop_duplicates(subset="ticker")
    return pd.Series(df["weight"].to_numpy() * 100, index=df["ticker"], name="weight")


@st.cache_data(ttl=CACHE_SECONDS)
def load_sentiment(tickers: tuple[str, ...]) -> pd.DataFrame:
    return sentiment_summary(list(tickers))


@st.cache_data(ttl=CACHE_SECONDS)
def load_latest_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Latest stored close and daily % change; empty if prices were never fetched."""
    placeholders = ",".join("?" * len(tickers))
    query = f"""
        SELECT ticker, close, pct_change
        FROM prices
        WHERE ticker IN ({placeholders})
          AND date = (SELECT MAX(date) FROM prices p2 WHERE p2.ticker = prices.ticker)
    """
    try:
        with get_connection() as conn:
            return pd.read_sql(query, conn, params=list(tickers))
    except Exception:  # no prices table yet: run master_fetcher.py
        return pd.DataFrame(columns=["ticker", "close", "pct_change"])


@st.cache_data(ttl=CACHE_SECONDS)
def load_recent_articles(ticker: str, limit: int = 8) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(
            """
            SELECT date, headline, source, ROUND(sentiment, 4) AS sentiment
            FROM articles
            WHERE ticker = ? AND sentiment IS NOT NULL
            ORDER BY date DESC
            LIMIT ?
            """,
            conn,
            params=(ticker, limit),
        )


@st.cache_data(ttl=CACHE_SECONDS)
def load_sentiment_history(ticker: str) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql(
            """
            SELECT date, ROUND(AVG(sentiment), 4) AS avg_sentiment
            FROM articles
            WHERE ticker = ? AND sentiment IS NOT NULL
            GROUP BY date
            ORDER BY date ASC
            """,
            conn,
            params=(ticker,),
        )


@st.cache_data(ttl=CACHE_SECONDS)
def latest_article_date() -> str | None:
    with get_connection() as conn:
        return conn.execute("SELECT MAX(date) FROM articles").fetchone()[0]


# ── Small helpers ───────────────────────────────────────────────────────────


def label_with_emoji(score: float | None) -> str:
    label = sentiment_label(score)
    return f"{SENTIMENT_EMOJI[label]} {label}"


def sentiment_color(score: float) -> str:
    if score > MILD_THRESHOLD:
        return "#2ecc71"
    if score < -MILD_THRESHOLD:
        return "#e74c3c"
    return "#f39c12"


def format_momentum(today: float | None, week: float | None) -> str:
    if pd.notna(today) and pd.notna(week):
        return f"{today - week:+.4f}"
    return "N/A"


# ── ETF sentiment tab ───────────────────────────────────────────────────────


def render_etf_tab(etf_key: str) -> None:
    etf = load_config()[etf_key]
    tickers = tuple(etf["tickers"])
    weights = load_target_weights()

    st.markdown(f"### {etf['name']}")
    st.caption(etf["description"])
    st.divider()

    if not db_has_articles():
        st.info(
            "No news database found yet. Run `python master_fetcher.py` and then "
            "`python sentiment_scorer.py` to populate it."
        )
        return

    sentiment_df = load_sentiment(tickers)
    if sentiment_df.empty:
        st.warning(
            f"No scored articles yet for {etf_key}. "
            "Run `master_fetcher.py` and `sentiment_scorer.py` first."
        )
        return

    merged = sentiment_df.merge(load_latest_prices(tickers), on="ticker", how="left")
    merged["target_weight"] = merged["ticker"].map(weights)

    # Overview cards
    st.subheader("Holdings overview")
    cols = st.columns(min(len(merged), 5))
    for i, row in enumerate(merged.itertuples(index=False)):
        with cols[i % len(cols)]:
            st.metric(
                label=f"{SENTIMENT_EMOJI[sentiment_label(row.avg_sentiment)]} {row.ticker}",
                value=f"{row.avg_sentiment:+.4f}",
                delta=f"{row.pct_change:+.2f}%" if pd.notna(row.pct_change) else None,
            )
            st.caption(sentiment_label(row.avg_sentiment))

    st.divider()

    # Summary table
    st.subheader("Sentiment summary")
    table = pd.DataFrame(
        {
            "Ticker": merged["ticker"],
            "Avg sentiment": merged["avg_sentiment"].map("{:+.4f}".format),
            "Sentiment": merged["avg_sentiment"].map(label_with_emoji),
            "Today vs 7-day avg": [
                format_momentum(t, w)
                for t, w in zip(merged["today_sentiment"], merged["week_sentiment"])
            ],
            "Target weight": merged["target_weight"].map(
                lambda v: f"{v:.1f}%" if pd.notna(v) else "N/A"
            ),
            "Articles": merged["article_count"].astype(int),
            "Latest close": merged["close"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "N/A"),
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)

    st.divider()

    # Charts
    left, right = st.columns(2)
    with left:
        st.subheader("Largest target weights")
        top = weights.reindex(list(tickers)).dropna().sort_values(ascending=False).head(12)
        fig = px.bar(
            top.sort_values(),
            orientation="h",
            labels={"value": "Target weight (%)", "index": ""},
        )
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, width="stretch")
        st.caption("Weights come from `holdings.csv`, not from sentiment.")

    with right:
        st.subheader("Average sentiment by ticker")
        fig2 = go.Figure(
            go.Bar(
                x=merged["ticker"],
                y=merged["avg_sentiment"],
                marker_color=[sentiment_color(s) for s in merged["avg_sentiment"]],
            )
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(yaxis_title="VADER compound score", margin=dict(t=20, b=20))
        st.plotly_chart(fig2, width="stretch")

    st.divider()

    # Category breakdown
    st.subheader("Category breakdown")
    scores = merged.set_index("ticker")["avg_sentiment"]
    categories = list(etf["categories"].items())
    for start in range(0, len(categories), CATEGORY_COLUMNS):
        chunk = categories[start : start + CATEGORY_COLUMNS]
        for col, (name, members) in zip(st.columns(len(chunk)), chunk):
            with col:
                st.markdown(f"**{name}**")
                for ticker in members:
                    if ticker in scores.index:
                        emoji = SENTIMENT_EMOJI[sentiment_label(scores[ticker])]
                        st.markdown(f"{emoji} `{ticker}` {scores[ticker]:+.4f}")
                    else:
                        st.markdown(f"⬜ `{ticker}` no data")

    st.divider()

    # Ticker deep dive
    st.subheader("Ticker deep dive")
    selected = st.selectbox("Select a ticker", merged["ticker"].tolist(), key=f"select_{etf_key}")
    if not selected:
        return

    col_articles, col_trend = st.columns(2)
    with col_articles:
        st.markdown("**Recent articles**")
        articles = load_recent_articles(selected)
        if articles.empty:
            st.info("No articles found for this ticker yet.")
        for article in articles.itertuples(index=False):
            dot = SENTIMENT_EMOJI[sentiment_label(article.sentiment)]
            headline = article.headline
            short = headline[:75] + ("..." if len(headline) > 75 else "")
            st.markdown(f"{dot} `{article.sentiment:+.4f}` **{short}**")
            st.caption(f"{article.source} · {article.date}")

    with col_trend:
        st.markdown("**Sentiment trend**")
        history = load_sentiment_history(selected)
        if history.empty:
            st.info("No sentiment history yet.")
        else:
            fig3 = px.line(history, x="date", y="avg_sentiment", markers=True)
            fig3.add_hline(y=0, line_dash="dash", line_color="gray")
            fig3.update_layout(yaxis_title="Avg sentiment", xaxis_title="", margin=dict(t=10, b=10))
            st.plotly_chart(fig3, width="stretch")


# ── Performance tab ─────────────────────────────────────────────────────────


def backtest_caption() -> str:
    """Describe the backtest window from etf_output/run_info.json, if present."""
    info_path = OUTPUT_DIR / "run_info.json"
    base = "Historical simulation using Yahoo Finance prices. Re-run `etf_charts.py` to refresh."
    if not info_path.exists():
        return base
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return (
        f"Backtest window: {info['effective_start']} to {info['end']} "
        f"({info['trading_days']:,} trading days, {info['holdings_with_data']} holdings, "
        f"{info['rebalance_freq']}-rebalanced). {base}"
    )


def render_performance_tab() -> None:
    st.markdown("### Portfolio performance analysis")
    st.caption(backtest_caption())
    st.divider()

    if not OUTPUT_DIR.exists():
        st.warning("No charts found. Run `python etf_charts.py` first to generate them.")
        return

    missing = []
    for filename, title, caption in CHARTS:
        path = OUTPUT_DIR / filename
        if not path.exists():
            missing.append(filename)
            continue
        st.subheader(title)
        st.caption(caption)
        st.image(str(path), width="stretch")
        st.download_button(
            "Download chart",
            data=path.read_bytes(),
            file_name=filename,
            mime="image/png",
            key=filename,
        )
        st.divider()

    if missing:
        names = "\n".join(f"- `{name}`" for name in missing)
        st.warning(
            "These charts were not found in `etf_output/`. "
            f"Re-run `etf_charts.py` to regenerate them:\n\n{names}"
        )

    st.subheader("Metrics snapshot")
    snapshot = OUTPUT_DIR / "snapshot.csv"
    if snapshot.exists():
        st.dataframe(pd.read_csv(snapshot), width="stretch", hide_index=True)
    else:
        st.info("No snapshot found. Run `python etf_charts.py` to create `snapshot.csv`.")


# ── App shell ───────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="Security ETF Research Dashboard", page_icon="📈", layout="wide")
    st.title("Security ETF Research Dashboard")
    st.info(DISCLAIMER)

    if db_has_articles():
        latest = latest_article_date()
        st.caption(f"Latest article in database: {latest or 'n/a'}")

    etfs = load_config()
    tabs = st.tabs(
        [f"{key}: {etf['name']}" for key, etf in etfs.items()] + ["Performance analysis"]
    )
    for tab, key in zip(tabs[:-1], etfs):
        with tab:
            render_etf_tab(key)
    with tabs[-1]:
        render_performance_tab()

    st.divider()
    st.caption("Data: NewsAPI · Alpha Vantage · SEC EDGAR · Yahoo Finance · VADER Sentiment")


main()
