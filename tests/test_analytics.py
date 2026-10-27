import numpy as np
import pandas as pd
import pytest

import analytics as an


def business_days(n: int, start: str = "2022-01-03") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def random_prices(cols, n=400, seed=0, start="2022-01-03") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0004, 0.015, size=(n, len(cols)))
    return pd.DataFrame(
        100 * np.exp(steps.cumsum(axis=0)), index=business_days(n, start), columns=cols
    )


# ── Portfolio construction ──────────────────────────────────────────────────


def test_flat_prices_keep_value_constant():
    px = pd.DataFrame(50.0, index=business_days(300), columns=["A", "B"])
    result = an.build_rebalanced_portfolio(px, pd.Series({"A": 0.6, "B": 0.4}))
    assert result.index.iloc[0] == pytest.approx(100.0)
    assert result.index.iloc[-1] == pytest.approx(100.0)


def test_single_asset_tracks_price():
    px = random_prices(["A"], n=200)
    result = an.build_rebalanced_portfolio(px, pd.Series({"A": 1.0}))
    expected = 100 * px["A"] / px["A"].iloc[0]
    pd.testing.assert_series_equal(result.index, expected.rename("PTEQ"), check_freq=False)


def test_start_waits_for_enough_weight_coverage():
    px = random_prices(["A", "B", "C"], n=300)
    px.loc[px.index[:100], "C"] = np.nan  # C (40% weight) lists on day 100
    weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})

    result = an.build_rebalanced_portfolio(px, weights, min_coverage=0.9)
    assert result.start == px.index[100]
    assert result.coverage_at_start == pytest.approx(1.0)

    relaxed = an.build_rebalanced_portfolio(px, weights, min_coverage=0.5)
    assert relaxed.start == px.index[0]
    assert relaxed.coverage_at_start == pytest.approx(0.6)


def test_late_listing_joins_at_next_rebalance():
    px = random_prices(["A", "B", "C"], n=300)
    px.loc[px.index[:30], "C"] = np.nan  # C lists on day 30, 4% weight
    weights = pd.Series({"A": 0.48, "B": 0.48, "C": 0.04})

    result = an.build_rebalanced_portfolio(px, weights, min_coverage=0.9)
    assert result.start == px.index[0]
    assert (result.shares["C"].iloc[:30] == 0).all()

    first_held = result.shares["C"].ne(0).idxmax()
    quarter_starts = px.groupby(px.index.to_period("Q")).head(1).index
    assert first_held in quarter_starts
    assert first_held > px.index[30]


def test_contributions_sum_to_total_return_with_rebalancing_and_late_listing():
    px = random_prices(["A", "B", "C", "D"], n=500, seed=3)
    px.loc[px.index[:40], "D"] = np.nan
    weights = pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})

    result = an.build_rebalanced_portfolio(px, weights, min_coverage=0.85)
    contrib = an.holding_contributions(px, result.shares)
    total_return = result.index.iloc[-1] / result.index.iloc[0] - 1.0
    assert contrib.sum() == pytest.approx(total_return, rel=1e-9)


def test_unknown_tickers_raise():
    px = pd.DataFrame(1.0, index=business_days(5), columns=["A"])
    with pytest.raises(ValueError):
        an.build_rebalanced_portfolio(px, pd.Series({"Z": 1.0}))


# ── Metrics ─────────────────────────────────────────────────────────────────


def test_drawdown_series():
    s = pd.Series([100, 110, 99, 120, 90.0], index=business_days(5))
    dd = an.drawdown_series(s)
    assert dd.iloc[1] == 0
    assert dd.iloc[2] == pytest.approx(99 / 110 - 1)
    assert dd.min() == pytest.approx(90 / 120 - 1)


def test_annualized_return_matches_formula():
    idx = pd.DatetimeIndex(["2020-01-01", "2021-01-01"])
    s = pd.Series([100.0, 200.0], index=idx)
    years = (idx[1] - idx[0]).days / 365.25
    assert an.annualized_return(s) == pytest.approx(2 ** (1 / years) - 1)


def test_sharpe_is_nan_for_constant_excess_returns():
    rf = an._daily_rf(an.RISK_FREE_ANNUAL)
    r = pd.Series(rf, index=business_days(50))
    assert np.isnan(an.sharpe_ratio(r))


def test_sortino_uses_rms_of_negative_excess_over_all_observations():
    rf = an._daily_rf(an.RISK_FREE_ANNUAL)
    excess = np.array([0.01, 0.02, -0.01, 0.0, -0.03, 0.015])
    r = pd.Series(excess + rf, index=business_days(len(excess)))
    expected_dd = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    expected = np.sqrt(252) * excess.mean() / expected_dd
    assert an.sortino_ratio(r) == pytest.approx(expected)


def test_sortino_nan_without_downside():
    rf = an._daily_rf(an.RISK_FREE_ANNUAL)
    r = pd.Series([rf + 0.01, rf + 0.02, rf + 0.0], index=business_days(3))
    assert np.isnan(an.sortino_ratio(r))


def test_alpha_beta_recovers_known_relationship():
    rf = an._daily_rf(an.RISK_FREE_ANNUAL)
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(0.0005, 0.01, 500), index=business_days(500))
    y = rf + 2.0 * (x - rf)
    alpha, beta = an.alpha_beta(y, x)
    assert beta == pytest.approx(2.0)
    assert alpha == pytest.approx(0.0, abs=1e-9)


def test_rolling_beta_of_scaled_series():
    rng = np.random.default_rng(2)
    bench = pd.Series(rng.normal(0, 0.01, 300), index=business_days(300))
    beta, corr = an.rolling_beta_corr(1.5 * bench, bench, window=63)
    assert beta.dropna().iloc[-1] == pytest.approx(1.5)
    assert corr.dropna().iloc[-1] == pytest.approx(1.0)


def test_metrics_table_shape_and_columns():
    px = random_prices(["PTEQ", "SPY", "HACK"], n=400, seed=5)
    rets = px.pct_change().dropna()
    table = an.build_metrics_table(px, rets)
    assert list(table.index) == ["PTEQ", "SPY", "HACK"]
    assert {"CAGR", "Sharpe", "Sortino", "Max DD", "Calmar"} <= set(table.columns)
    assert pd.notna(table.loc["PTEQ", "Beta (vs SPY)"])
    assert pd.isna(table.loc["SPY", "Beta (vs SPY)"])


# ── Composition helpers ─────────────────────────────────────────────────────


def test_holdings_overlap():
    result = an.holdings_overlap(["A", "B", "C", "D"], {"X": ["A", "B", "Q"], "Y": ["Z"]})
    assert result == {"X": 0.5, "Y": 0.0}
    assert an.holdings_overlap([], {"X": ["A"]}) == {"X": 0.0}


@pytest.mark.parametrize("value,label", [(0.95, "high"), (0.6, "moderate"), (0.1, "low")])
def test_describe_correlation(value, label):
    assert an.describe_correlation(value) == label
