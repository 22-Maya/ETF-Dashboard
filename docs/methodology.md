# Methodology and Research Limitations

This document describes what the code in this repository does and where its results should not be over-interpreted.

## Portfolio construction (`analytics.py`, `etf_charts.py`)

Prices are adjusted closes from Yahoo Finance (`yfinance`, `auto_adjust=True`). Target weights come from `holdings.csv` and are normalised to sum to 1.

Defaults (constants at the top of each file):

| Setting | Value |
| --- | --- |
| Requested start | 2020-01-01 |
| Rebalancing | First trading day of each quarter, at that day's close |
| Initial index value | 100 |
| Benchmarks | SPY, LOCK.L, CIBR, HACK, BUG |
| Risk-free rate | 4.5% annualized, constant |
| Minimum weight coverage at start | 90% |
| Transaction costs | None |

**Effective start date.** Some holdings listed after the requested start. The simulation begins on the first date where holdings covering at least 90% of the target weight have prices. A holding without a price is left out and the remaining weights are scaled up; it joins at the next quarterly rebalance after it first trades. Dropping every date before the *newest* holding listed would shrink the history to a few years, so this rule keeps more data at the cost of a short period where the portfolio differs from its target weights.

The window actually used is printed in every chart footnote, shown in the dashboard, and saved in `etf_output/run_info.json` together with each holding's first price date.

**Rebalancing.** On each rebalance date the portfolio is first marked to that day's closing prices, then positions are reset to target weights at those same prices. Between rebalances, share counts are fixed. A stock that stops trading is held flat at its last price (prices are forward-filled).

**Benchmarks** are aligned to the portfolio's window and rebased to 100 on the same first date. Prices are forward-filled across market holidays, which matters for LOCK.L (London calendar).

## Metrics

All use daily returns from the rebased series. "Annualized" uses 252 trading days.

- **CAGR:** `(end / start) ^ (1 / years) - 1`, with years = calendar days / 365.25.
- **Volatility:** sample standard deviation of daily returns × √252.
- **Sharpe:** √252 × mean(excess return) / std(excess return), where excess return = daily return minus the daily equivalent of the risk-free rate.
- **Sortino:** √252 × mean(excess return) / downside deviation. Downside deviation is the root-mean-square of `min(excess, 0)` over **all** observations, with the risk-free rate as the minimum acceptable return.
- **Max drawdown / Calmar:** largest peak-to-trough decline; Calmar = CAGR / |max drawdown|.
- **Alpha / beta:** regression of the portfolio's daily excess returns on SPY's daily excess returns. Beta = cov / var; alpha = (mean y − beta × mean x) × 252. Reported for the portfolio versus SPY only.
- **Rolling beta / correlation:** 63-day and 252-day windows.

The risk-free rate is a fixed assumption, not a historical series, so Sharpe, Sortino and alpha depend on it.

## Attribution

Each holding's contribution is its daily profit and loss, using the shares actually held that day, summed over the window and divided by the starting value. Contributions therefore add up exactly to the portfolio's total return, including the effects of rebalancing and late additions. It is a descriptive decomposition, not a factor model.

## Classification data

Themes, buckets and sectors in `theme_map.csv` are manually assigned labels, not an official classification such as GICS. Treat them as the author's judgment. The overlap chart uses the constituent lists in `peer_holdings.csv`, which are static, partial snapshots rather than full, dated ETF holdings.

## Sentiment workflow

`master_fetcher.py` collects article and filing text from enabled sources and stores it in `etf_news.db`. `sentiment_scorer.py` applies VADER to headline plus description or excerpt. `dashboard.py` averages scores by ticker and maps the average to descriptive labels (Very positive, Positive, Neutral, Negative, Very negative) using fixed cut-offs of ±0.2 and ±0.5.

These labels describe the tone of the text. The repository does not show that they forecast returns, improve portfolio performance, or constitute a trading signal. VADER is a general-purpose lexicon model and is not tuned for financial text.

## Important limitations

1. **Point-in-time holdings:** the current portfolio is applied to historical prices without reconstructing the holdings and weights that were knowable at each date. This introduces survivorship and look-ahead bias.
2. **Short and uneven history:** late listings shorten the usable window, and results over a few years can be dominated by one market regime.
3. **No transaction costs:** commissions, spreads, taxes, slippage and market impact are not modeled, so returns are gross.
4. **Fixed risk-free rate:** a constant assumption, not a time series.
5. **Data quality and coverage:** Yahoo Finance and news APIs may contain gaps, revisions, delays and rate limits. Delisted or acquired stocks are held flat rather than modeled explicitly.
6. **No predictive validation:** sentiment labels have not been tested against subsequent returns on a holdout period.
7. **No significance testing:** there are no confidence intervals or significance tests for outperformance, alpha, or sentiment predictive power.
8. **Manual classifications:** theme and sector labels are subjective.

## Research extensions

Useful next steps include point-in-time constituent data, a walk-forward backtest, transaction-cost modeling, a historical risk-free series, out-of-sample sentiment evaluation, a finance-specific sentiment model, comparison with simple baselines, sensitivity analysis (rebalance frequency, coverage threshold), and statistical uncertainty estimates. Any reported finding should state the test period, data sources, assumptions and limitations.
