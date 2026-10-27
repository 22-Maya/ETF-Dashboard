"""Portfolio construction and performance/risk metrics.

Pure pandas/NumPy functions with no plotting or network access, so they can be
unit-tested (see ``tests/test_analytics.py``) and reused outside the chart script.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252
BASE_VALUE = 100.0
REBALANCE_FREQ = "Q"  # pandas period alias: rebalance on the first trading day of each quarter
RISK_FREE_ANNUAL = 0.045  # fixed modeling assumption, not a historical series
MIN_WEIGHT_COVERAGE = 0.90  # start once holdings with this share of target weight have prices


# ── Portfolio construction ──────────────────────────────────────────────────


def dedupe_keep_order(seq: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(seq))


def build_equal_weights(tickers: Sequence[str]) -> pd.Series:
    if len(tickers) == 0:
        raise ValueError("No tickers available to weight.")
    return pd.Series(1.0 / len(tickers), index=list(tickers), dtype=float)


@dataclass
class PortfolioResult:
    """Output of :func:`build_rebalanced_portfolio`."""

    index: pd.Series  # portfolio value, starting at ``base_value``
    shares: pd.DataFrame  # post-rebalance share counts per holding per day
    start: pd.Timestamp  # first date of the simulation
    coverage_at_start: float  # share of target weight with prices on ``start``
    first_prices: pd.Series  # first available price date for each holding


def build_rebalanced_portfolio(
    prices: pd.DataFrame,
    weights: pd.Series,
    rebalance_freq: str = REBALANCE_FREQ,
    base_value: float = BASE_VALUE,
    min_coverage: float = MIN_WEIGHT_COVERAGE,
) -> PortfolioResult:
    """Simulate a periodically rebalanced portfolio.

    Holdings that have not started trading yet are left out and the remaining
    target weights are re-normalised; each holding joins at the first rebalance
    after it has a price. The simulation starts on the first date where holdings
    covering at least ``min_coverage`` of the target weight have prices, so one
    recently listed stock does not truncate the whole history.

    Trades happen at the close of each rebalance date. Prices are forward-filled
    after a holding's first price, so a delisted stock is held flat.
    """
    cols = [c for c in weights.index if c in prices.columns]
    if not cols:
        raise ValueError("None of the holdings tickers are present in the price panel.")

    px = prices[cols].sort_index().ffill()
    w = weights.loc[cols].astype(float)
    w = w / w.sum()

    coverage = px.notna().mul(w, axis=1).sum(axis=1)
    eligible = coverage[coverage >= min_coverage]
    if eligible.empty:
        raise ValueError(
            f"No date has prices for holdings covering {min_coverage:.0%} of the weight."
        )
    start = eligible.index[0]
    px = px.loc[start:]

    periods = px.index.to_period(rebalance_freq)
    rebalance_dates = set(px.groupby(periods).head(1).index) | {start}

    price_arr = px.to_numpy()
    w_arr = w.to_numpy()
    n_days, n_assets = price_arr.shape
    shares_arr = np.zeros((n_days, n_assets))
    values = np.zeros(n_days)

    value = float(base_value)
    current = np.zeros(n_assets)
    for i, dt in enumerate(px.index):
        row = price_arr[i]
        if i > 0:
            # Mark the existing positions to today's close *before* rebalancing,
            # so the rebalance date's market move is not lost.
            value = float(np.nansum(current * row))
        if dt in rebalance_dates:
            live = ~np.isnan(row)
            current = np.zeros(n_assets)
            current[live] = value * (w_arr[live] / w_arr[live].sum()) / row[live]
        shares_arr[i] = current
        values[i] = value

    return PortfolioResult(
        index=pd.Series(values, index=px.index, name="PTEQ"),
        shares=pd.DataFrame(shares_arr, index=px.index, columns=cols),
        start=start,
        coverage_at_start=float(coverage.loc[start]),
        first_prices=prices[cols].apply(lambda s: s.first_valid_index()),
    )


def holding_contributions(
    prices: pd.DataFrame, shares: pd.DataFrame, base_value: float = BASE_VALUE
) -> pd.Series:
    """Each holding's contribution to total portfolio return.

    Uses the share counts actually held each day, so the contributions sum
    exactly to the portfolio's total return over the period (rebalancing and
    compounding included).
    """
    px = prices[shares.columns].reindex(shares.index).ffill()
    pnl = (shares.shift(1) * px.diff()).sum()
    return pnl / base_value


# ── Return and risk metrics ─────────────────────────────────────────────────


def drawdown_series(series: pd.Series) -> pd.Series:
    """Percentage decline from the running peak (0 at highs, negative below)."""
    return series / series.cummax() - 1.0


def _daily_rf(rf_annual: float) -> float:
    return (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0


def rolling_beta_corr(
    etf_ret: pd.Series, bench_ret: pd.Series, window: int = 63
) -> tuple[pd.Series, pd.Series]:
    beta = etf_ret.rolling(window).cov(bench_ret) / bench_ret.rolling(window).var()
    corr = etf_ret.rolling(window).corr(bench_ret)
    return beta, corr


def annualized_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def annualized_return(series: pd.Series) -> float:
    """Compound annual growth rate from a value series."""
    if series.empty:
        return np.nan
    n_days = (series.index[-1] - series.index[0]).days
    if n_days <= 0:
        return np.nan
    years = n_days / 365.25
    return (series.iloc[-1] / series.iloc[0]) ** (1.0 / years) - 1.0


def sharpe_ratio(returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    excess = returns - _daily_rf(rf_annual)
    sd = excess.std()
    if excess.empty or not sd or np.isnan(sd):
        return np.nan
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / sd)


def downside_deviation(excess: pd.Series) -> float:
    """Root-mean-square of negative excess returns, over *all* observations."""
    return float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2)))


def sortino_ratio(returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    """Annualised Sortino ratio with the risk-free rate as the minimum acceptable return."""
    if returns.empty:
        return np.nan
    excess = returns - _daily_rf(rf_annual)
    dd = downside_deviation(excess)
    if dd == 0:
        return np.nan
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / dd)


def calmar_ratio(series: pd.Series) -> float:
    mdd = drawdown_series(series).min()
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return annualized_return(series) / abs(mdd)


def alpha_beta(
    etf_ret: pd.Series, bench_ret: pd.Series, rf_annual: float = RISK_FREE_ANNUAL
) -> tuple[float, float]:
    """Annualised alpha (daily alpha x 252) and beta from daily excess returns."""
    rf = _daily_rf(rf_annual)
    aligned = pd.concat([etf_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan, np.nan
    y = aligned.iloc[:, 0] - rf
    x = aligned.iloc[:, 1] - rf
    beta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    alpha_daily = y.mean() - beta * x.mean()
    return float(alpha_daily * TRADING_DAYS), float(beta)


def build_metrics_table(
    levels: pd.DataFrame, returns: pd.DataFrame, portfolio: str = "PTEQ", market: str = "SPY"
) -> pd.DataFrame:
    """One row of summary statistics per column of ``levels``."""
    rows = []
    for col in levels.columns:
        r = returns[col]
        alpha = beta = np.nan
        if col == portfolio and market in returns.columns:
            alpha, beta = alpha_beta(returns[portfolio], returns[market])
        rows.append(
            {
                "ETF": col,
                "CAGR": annualized_return(levels[col]),
                "Ann Vol": r.std() * np.sqrt(TRADING_DAYS),
                "Sharpe": sharpe_ratio(r),
                "Sortino": sortino_ratio(r),
                "Max DD": drawdown_series(levels[col]).min(),
                "Calmar": calmar_ratio(levels[col]),
                f"Alpha (vs {market})": alpha,
                f"Beta (vs {market})": beta,
            }
        )
    return pd.DataFrame(rows).set_index("ETF")


# ── Composition helpers ─────────────────────────────────────────────────────


def holdings_overlap(
    holdings: Sequence[str], etf_holdings: dict[str, Sequence[str]]
) -> dict[str, float]:
    """Fraction of ``holdings`` that also appears in each peer ETF."""
    holding_set = set(holdings)
    if not holding_set:
        return {name: 0.0 for name in etf_holdings}
    return {
        name: len(holding_set & set(members)) / len(holding_set)
        for name, members in etf_holdings.items()
    }


def describe_correlation(avg_corr: float) -> str:
    """Plain-language label for an average correlation (descriptive only)."""
    if avg_corr >= 0.8:
        return "high"
    if avg_corr >= 0.5:
        return "moderate"
    return "low"
