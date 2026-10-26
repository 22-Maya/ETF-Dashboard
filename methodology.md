# Methodology and Research Limitations

This document describes the current implementation and identifies limitations that matter when interpreting results.

## Portfolio performance workflow

`etf_charts.py` retrieves historical adjusted price data with `yfinance`, aligns observations across the configured holdings and benchmarks, constructs a portfolio index with the configured rebalancing frequency, and exports charts and summary tables.

The current configuration uses:

- Start date: January 1, 2020
- Rebalancing frequency: quarterly
- Initial index value: 100
- Benchmarks: SPY, LOCK.L, CIBR, HACK, and BUG
- Risk-free-rate assumption: 4.5% annualized, as configured in the script

The risk-free rate is a fixed modeling assumption in the current implementation, not a dynamically sourced historical series. Results therefore depend on that assumption.

## Reported metrics

The analysis includes annualized return (CAGR), annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, alpha/beta estimates, and rolling beta/correlation. The formulas and implementation are in `etf_charts.py` and should be reviewed before using the metrics for decision-making.

The holding contribution chart uses weighted holding returns as an approximate attribution view. It is not a full daily, rebalanced attribution model and may not exactly reconcile to the portfolio's realized return.

## Sentiment workflow

`master_fetcher.py` collects article and filing text from enabled sources and stores records in `etf_news.db`. `sentiment_scorer.py` applies VADER sentiment analysis to available text. `dashboard.py` aggregates scores by ticker and maps average sentiment to heuristic labels.

Those labels are descriptive rules applied to text sentiment. The repository does not demonstrate that they forecast returns, improve portfolio performance, or produce statistically significant trading signals.

## Important limitations

1. **Point-in-time holdings:** The configured portfolio is applied to historical prices without reconstructing the exact holdings and weights that would have been known at every historical date. This can introduce survivorship and look-ahead bias.
2. **No predictive validation:** The sentiment labels have not been validated against subsequent returns using a separate holdout period.
3. **No transaction costs:** The current portfolio simulation does not establish net returns after commissions, bid-ask spreads, taxes, slippage, or market impact.
4. **Fixed risk-free assumption:** The configured risk-free rate is constant rather than a historical time series.
5. **Data quality and coverage:** Price and news sources may contain missing values, revisions, delays, API limits, and inconsistent historical coverage.
6. **Heuristic sentiment labels:** Thresholds used to assign buy/hold/sell labels are rules of thumb, not a validated investment model.
7. **Approximate attribution:** Weighted full-period holding returns do not account for the complete timing and compounding effects of periodic rebalancing.
8. **No statistical significance claims:** The current repository does not provide formal confidence intervals or significance tests for strategy outperformance or sentiment predictive power.

## Research extensions

Useful next steps include point-in-time constituent data, a walk-forward backtest, transaction-cost modeling, out-of-sample signal evaluation, comparison with a simple baseline, sensitivity analysis, and statistical uncertainty estimates. Any reported findings should include the test period, data sources, assumptions, and limitations.
