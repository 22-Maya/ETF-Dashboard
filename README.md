# ETF Dashboard & Quantitative Portfolio Research

A Python research project for evaluating a thematic security-focused equity portfolio against market and sector ETF benchmarks, alongside a separate financial-news sentiment pipeline.

> **Research prototype, not investment advice.** Historical results are sensitive to data availability, portfolio assumptions, benchmark selection, and survivorship/look-ahead bias. The configured portfolio is not a live fund and its historical simulation should not be interpreted as an investable track record.

## Project overview

This repository contains two connected research workflows:

1. **Portfolio performance analysis** downloads historical price data, constructs a rebalanced portfolio index, compares it with selected benchmarks, and exports performance and risk charts.
2. **News sentiment dashboard** ingests financial news and SEC filing summaries, stores article records in SQLite, scores text with VADER, and displays ticker-level sentiment and price context in Streamlit.

## Research features

### Portfolio analytics

- Quarterly rebalanced portfolio index based on configured holdings and weights.
- Benchmark comparisons against SPY, LOCK.L, CIBR, HACK, and BUG.
- CAGR, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, and alpha/beta calculations.
- Rolling volatility, beta, and correlation views.
- Holdings overlap, sector/theme concentration, and approximate weighted holding-return contributions.
- CSV exports for summary metrics and the latest portfolio snapshot.

### News and sentiment workflow

- Financial-news collection using NewsAPI and Alpha Vantage, with SEC EDGAR filing retrieval where configured.
- SQLite storage for article metadata and sentiment scores.
- VADER sentiment scoring and ticker-level aggregation.
- Streamlit interface for sentiment summaries, recent articles, and portfolio analysis charts.

## Repository structure

```text
ETF-Dashboard/
├── dashboard.py           # Streamlit dashboard
├── etf_charts.py           # Portfolio construction, analytics, and chart exports
├── master_fetcher.py       # News and filing ingestion into SQLite
├── sentiment_scorer.py     # VADER sentiment scoring
├── etf_holdings.json       # ETF universe, holdings, search terms, and SEC CIKs
├── holdings.csv            # Portfolio holdings and weights for performance analysis
├── peer_holdings.csv       # Peer ETF constituents for overlap analysis
├── theme_map.csv           # Holding-to-theme mapping
├── etf_output/             # Example generated charts and CSV summaries
├── .env.example            # Environment variable template
└── requirements.txt        # Python dependencies
```

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\\Scripts\\activate       # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Add API keys to `.env` if you want to enable those sources. Never commit `.env` or API credentials. The portfolio analysis script uses Yahoo Finance data through `yfinance` and does not require the news API keys.

For SEC EDGAR requests, set `SEC_USER_AGENT` to a descriptive application name and a valid contact email, consistent with SEC guidance.

## Run the portfolio analysis

```bash
python etf_charts.py
```

Charts and CSV summaries are written to `etf_output/`. Historical prices are retrieved when the script runs, so results can change as data sources revise or extend their data.

## Run the news sentiment dashboard

The first command creates the local SQLite database and attempts to collect data from the enabled sources:

```bash
python master_fetcher.py
python sentiment_scorer.py
streamlit run dashboard.py
```

The news workflow may have no data when API credentials are absent, provider limits are reached, or no articles are returned. Configure the keys in `.env` and rerun ingestion. The database is intentionally excluded from version control because it is generated locally.

## Example portfolio outputs

These are example outputs committed with the project. They are historical analysis artifacts, not live results or a promise of future performance.

### Indexed performance

![Indexed portfolio and benchmark performance](etf_output/01_indexed_performance.png)

### Drawdown comparison

![Portfolio and benchmark drawdown](etf_output/04_drawdown_comparison.png)

### Rolling beta

![Rolling beta versus benchmark](etf_output/05_rolling_beta_vs_spy.png)

### Holdings overlap

![Holdings overlap](etf_output/11_holdings_overlap.png)

## Methodology and limitations

Read [`docs/methodology.md`](docs/methodology.md) before interpreting the charts or metrics. In particular, the current historical reconstruction applies the configured holdings and weights across the analysis period rather than reconstructing point-in-time historical constituents and weights. That can introduce survivorship and look-ahead bias. Sentiment-based labels are heuristic classifications and have not been demonstrated here to predict future returns.

## Tech stack

Python · pandas · NumPy · Matplotlib · Plotly · Streamlit · SQLite · yfinance · VADER Sentiment · NewsAPI · Alpha Vantage · SEC EDGAR

## Disclaimer

This project is for educational and research purposes only. It is not investment advice, a recommendation to buy or sell securities, or a representation of a live ETF. Data providers may impose access limits, revise historical data, or return incomplete records. Validate assumptions and calculations independently before relying on any output.
