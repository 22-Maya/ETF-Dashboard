import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import json

st.set_page_config(
    page_title="ETF Sentiment Dashboard",
    page_icon="📈",
    layout="wide"
)

# ── Config ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_config():
    with open("etf_holdings.json", "r") as f:
        config = json.load(f)
    etfs = config["etfs"]
    for etf_key, etf_data in etfs.items():
        tickers = []
        for cat_tickers in etf_data["categories"].values():
            tickers.extend(cat_tickers)
        etf_data["tickers"] = list(dict.fromkeys(tickers))
    return etfs

ETF_UNIVERSE = load_config()

# ── DB helpers ─────────────────────────────────────────────────────────────

def get_connection():
    return sqlite3.connect("etf_news.db")

def get_sentiment_for_tickers(tickers):
    conn = get_connection()
    placeholders = ",".join("?" * len(tickers))
    df = pd.read_sql(f"""
        SELECT
            ticker,
            ROUND(AVG(sentiment), 4)  AS avg_sentiment,
            COUNT(*)                  AS article_count,
            ROUND(AVG(CASE WHEN date = DATE('now')
                      THEN sentiment END), 4) AS today_sentiment,
            ROUND(AVG(CASE WHEN date >= DATE('now', '-7 days')
                      THEN sentiment END), 4) AS week_sentiment
        FROM articles
        WHERE sentiment IS NOT NULL
        AND ticker IN ({placeholders})
        GROUP BY ticker
        ORDER BY avg_sentiment DESC
    """, conn, params=tickers)
    conn.close()
    return df

def get_prices_for_tickers(tickers):
    conn = get_connection()
    try:
        placeholders = ",".join("?" * len(tickers))
        df = pd.read_sql(f"""
            SELECT ticker, date, close, pct_change
            FROM prices
            WHERE ticker IN ({placeholders})
            AND date = (
                SELECT MAX(date) FROM prices p2
                WHERE p2.ticker = prices.ticker
            )
        """, conn, params=tickers)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df

def get_recent_articles(ticker, limit=8):
    conn = get_connection()
    df = pd.read_sql("""
        SELECT date, headline, source,
               ROUND(sentiment, 4) AS sentiment
        FROM articles
        WHERE ticker = ? AND sentiment IS NOT NULL
        ORDER BY date DESC
        LIMIT ?
    """, conn, params=(ticker, limit))
    conn.close()
    return df

def get_sentiment_history(ticker):
    conn = get_connection()
    df = pd.read_sql("""
        SELECT date, ROUND(AVG(sentiment), 4) AS avg_sentiment
        FROM articles
        WHERE ticker = ? AND sentiment IS NOT NULL
        GROUP BY date
        ORDER BY date ASC
    """, conn, params=(ticker,))
    conn.close()
    return df

# ── Signal helpers ──────────────────────────────────────────────────────────

def get_signal(score):
    if score is None or pd.isna(score):
        return "NO DATA", "⬜"
    if score > 0.5:
        return "STRONG BUY", "🟢"
    elif score > 0.2:
        return "BUY", "🟩"
    elif score > -0.2:
        return "HOLD", "🟡"
    elif score > -0.5:
        return "SELL", "🟠"
    else:
        return "STRONG SELL", "🔴"

def sentiment_to_weight(df):
    df = df.copy()
    if df.empty:
        return df
    min_s = df["avg_sentiment"].min()
    max_s = df["avg_sentiment"].max()
    if max_s == min_s:
        df["weight"] = round(100 / len(df), 1)
    else:
        df["score_shifted"] = df["avg_sentiment"] - min_s + 0.1
        total = df["score_shifted"].sum()
        df["weight"] = (df["score_shifted"] / total * 100).round(1)
    return df

# ── ETF sentiment tab ───────────────────────────────────────────────────────

def render_etf_tab(etf_key):
    etf     = ETF_UNIVERSE[etf_key]
    tickers = etf["tickers"]

    st.markdown(f"### {etf['emoji']} {etf['name']}")
    st.caption(etf["description"])
    st.divider()

    sentiment_df = get_sentiment_for_tickers(tickers)
    price_df     = get_prices_for_tickers(tickers)

    if sentiment_df.empty:
        st.warning(
            f"No scored articles yet for {etf_key}. "
            "Run `master_fetcher.py` and `sentiment_scorer.py` first."
        )
        return

    sentiment_df = sentiment_to_weight(sentiment_df)

    if not price_df.empty:
        merged = sentiment_df.merge(price_df, on="ticker", how="left")
    else:
        merged = sentiment_df.copy()
        merged["close"]      = None
        merged["pct_change"] = None

    # ── Metric cards ────────────────────────────────────────────────────
    st.subheader("Holdings overview")
    cols = st.columns(min(len(merged), 5))
    for i, (_, row) in enumerate(merged.iterrows()):
        signal, emoji = get_signal(row["avg_sentiment"])
        with cols[i % len(cols)]:
            st.metric(
                label=f"{emoji} {row['ticker']}",
                value=f"{row['avg_sentiment']:+.4f}",
                delta=f"{row['pct_change']:+.2f}%"
                      if pd.notna(row.get("pct_change")) else None,
            )
            st.caption(signal)

    st.divider()

    # ── Signal table ─────────────────────────────────────────────────────
    st.subheader("Signal table")
    table_data = []
    for _, row in merged.iterrows():
        signal, emoji = get_signal(row["avg_sentiment"])
        today_s = row.get("today_sentiment")
        week_s  = row.get("week_sentiment")
        momentum = (
            round(today_s - week_s, 4)
            if pd.notna(today_s) and pd.notna(week_s)
            else None
        )
        table_data.append({
            "Ticker":        row["ticker"],
            "Avg sentiment": f"{row['avg_sentiment']:+.4f}",
            "Signal":        f"{emoji} {signal}",
            "7d momentum":   f"{momentum:+.4f}" if momentum is not None else "N/A",
            "ETF weight":    f"{row['weight']:.1f}%",
            "Articles":      int(row["article_count"]),
            "Latest close":  f"${row['close']:.2f}" if pd.notna(row.get("close")) else "N/A",
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.divider()

    # ── Charts ───────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ETF weight allocation")
        fig = px.pie(
            merged, names="ticker", values="weight",
            hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Sentiment by ticker")
        colors = [
            "#2ecc71" if s > 0.2 else "#e74c3c" if s < -0.2 else "#f39c12"
            for s in merged["avg_sentiment"]
        ]
        fig2 = go.Figure(go.Bar(
            x=merged["ticker"], y=merged["avg_sentiment"],
            marker_color=colors,
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(yaxis_title="Sentiment score", margin=dict(t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ── Category breakdown ───────────────────────────────────────────────
    st.subheader("Category breakdown")
    categories = etf["categories"]
    cat_cols   = st.columns(len(categories))

    for i, (cat_name, cat_tickers) in enumerate(categories.items()):
        with cat_cols[i]:
            st.markdown(f"**{cat_name}**")
            for t in cat_tickers:
                row = merged[merged["ticker"] == t]
                if not row.empty:
                    score         = row.iloc[0]["avg_sentiment"]
                    signal, emoji = get_signal(score)
                    st.markdown(f"{emoji} `{t}` {score:+.4f}")
                else:
                    st.markdown(f"⬜ `{t}` no data")

    st.divider()

    # ── Ticker deep dive ─────────────────────────────────────────────────
    st.subheader("Ticker deep dive")
    selected = st.selectbox(
        "Select a ticker", merged["ticker"].tolist(), key=f"selectbox_{etf_key}"
    )

    if selected:
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Recent articles**")
            articles = get_recent_articles(selected)
            if not articles.empty:
                for _, a in articles.iterrows():
                    dot = (
                        "🟢" if a["sentiment"] > 0.2
                        else "🔴" if a["sentiment"] < -0.2
                        else "🟡"
                    )
                    headline = a["headline"]
                    short = headline[:75] + ("..." if len(headline) > 75 else "")
                    st.markdown(f"{dot} `{a['sentiment']:+.4f}` **{short}**")
                    st.caption(f"{a['source']} · {a['date']}")
            else:
                st.info("No articles found for this ticker yet.")

        with col4:
            st.markdown("**Sentiment trend**")
            history = get_sentiment_history(selected)
            if not history.empty:
                fig3 = px.line(history, x="date", y="avg_sentiment", markers=True)
                fig3.add_hline(y=0, line_dash="dash", line_color="gray")
                fig3.update_layout(
                    yaxis_title="Avg sentiment", xaxis_title="",
                    margin=dict(t=10, b=10),
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No sentiment history yet.")

# ── Performance tab (charts from etf_charts.py) ────────────────────────────

def render_performance_tab():
    st.markdown("### 📊 PTEQ Performance Analysis")
    st.caption(
        "Generated from historical price data via Yahoo Finance. "
        "Re-run `etf_charts.py` to refresh."
    )
    st.divider()

    output_dir = Path(__file__).parent / "etf_output"

    if not output_dir.exists():
        st.warning(
            "No charts found. Run `python etf_charts.py` first to generate them."
        )
        return

    charts = [
        (
            "01_indexed_performance.png",
            "① Indexed Performance vs Benchmarks",
            "How PTEQ has performed relative to SPY, CIBR, HACK, LOCK.L, and BUG since inception (base = 100).",
        ),
        (
            "02_relative_strength.png",
            "② Relative Strength vs Each Benchmark",
            "A ratio above 1.0 means PTEQ is outperforming that benchmark. Below 1.0 means underperforming.",
        ),
        (
            "03_volatility_bar.png",
            "③ Current 252-Day Annualised Volatility",
            "Annualised volatility over a 252-day window for PTEQ and each benchmark. Lower is more stable.",
        ),
        (
            "04_drawdown_comparison.png",
            "④ Drawdown Comparison",
            "Peak-to-trough decline at each point in time. Shallower and shorter drawdowns are better.",
        ),
        (
            "05_rolling_beta_vs_spy.png",
            "⑤ Rolling Beta vs SPY",
            "Beta above 1.0 means PTEQ moves more than SPY. Below 1.0 means it moves less.",
        ),
        (
            "06_rolling_correlation_vs_spy.png",
            "⑥ Rolling Correlation vs SPY",
            "PTEQ provides moderately differentiated exposure — partial overlap with the broad market but meaningful independent return drivers.",
        ),
        (
            "07_rolling_beta_vs_lock.png",
            "⑦ Rolling Beta vs LOCK.L",
            "Beta relative to the L&G Cyber Security ETF (LOCK.L).",
        ),
        (
            "08_rolling_correlation_vs_lock.png",
            "⑧ Rolling Correlation vs LOCK.L",
            "How closely PTEQ tracks LOCK.L over time.",
        ),
        (
            "09_concentration.png",
            "⑨ Concentration by Theme & Top Holdings",
            "Weight distribution across security themes and the top 10 individual holdings.",
        ),
        (
            "10_performance_attribution.png",
            "⑩ Performance Attribution",
            "Top and bottom contributors to PTEQ's total return, weighted by portfolio allocation.",
        ),
        (
            "11_holdings_overlap.png",
            "⑪ Holdings Overlap vs Peer ETFs",
            "Fraction of PTEQ holdings that also appear in each peer ETF.",
        ),
        (
            "12_metrics_table.png",
            "⑫ Risk-Adjusted Metrics",
            "CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, alpha, and beta for PTEQ and all benchmarks.",
        ),
        (
            "13_sector_exposure_donut.png",
            "⑬ Sector Exposure",
            "Portfolio weight broken down by sector (Software, Defense, Semiconductors, etc.).",
        ),
    ]

    missing = []
    for filename, title, caption in charts:
        path = output_dir / filename
        if not path.exists():
            missing.append(filename)
            continue

        st.subheader(title)
        st.caption(caption)
        st.image(str(path), use_container_width=True)

        with open(path, "rb") as f:
            st.download_button(
                label="Download chart",
                data=f,
                file_name=filename,
                mime="image/png",
                key=filename,
            )

        st.divider()

    if missing:
        missing_list = "\n".join(f"- `{f}`" for f in missing)
        st.warning(
            "The following charts were not found in `etf_output/` — "
            f"re-run `etf_charts.py` to regenerate them:\n\n{missing_list}"
        )
    st.subheader("Latest snapshot")
    snapshot_path = output_dir / "snapshot.csv"
    if snapshot_path.exists():
        snap = pd.read_csv(snapshot_path)
        st.dataframe(snap, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No snapshot CSV found. "
            "Add `summary.to_csv(OUTPUT_DIR / 'snapshot.csv', index=False)` "
            "at the end of `etf_charts.py` to populate this table."
        )

# ── App shell ───────────────────────────────────────────────────────────────

st.title("📈 ETF Sentiment Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")

tab_labels = [
    f"{etf['emoji']} {etf_key} — {etf['name']}"
    for etf_key, etf in ETF_UNIVERSE.items()
] + ["📊 Performance Analysis"]

tabs = st.tabs(tab_labels)

for tab, etf_key in zip(tabs[:-1], ETF_UNIVERSE.keys()):
    with tab:
        render_etf_tab(etf_key)

with tabs[-1]:
    render_performance_tab()

st.divider()
st.caption(
    "Data: NewsAPI · Alpha Vantage · SEC EDGAR · Yahoo Finance · VADER Sentiment"
)