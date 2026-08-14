# ─────────────────────────────────────────────────────────────────────────────
# UPDATED CODE
# Stack: Python 3.x · matplotlib · pandas · numpy · yfinance
# All charts are standalone PNGs saved to OUTPUT_DIR
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import yfinance as yf

# =============================================================================
# 1) INPUTS
# =============================================================================
START_DATE = "2020-01-01"
END_DATE = None
REBALANCE_FREQ = "Q"
BASE_VALUE = 100.0
OUTPUT_DIR = Path("etf_output")
OUTPUT_DIR.mkdir(exist_ok=True)

BENCHMARKS = [
    "SPY",
    "LOCK.L",
    "CIBR",
    "HACK",
    "BUG",
]

HOLDINGS_CSV = "holdings.csv"
THEME_MAP_CSV = "theme_map.csv"

# Color palette: PTEQ is bold blue, all benchmarks are shades of gray
COLORS = {
    "PTEQ":   "#0057B8",
    "SPY":    "#9E9E9E",
    "CIBR":   "#BDBDBD",
    "HACK":   "#B0B0B0",
    "BUG":    "#C8C8C8",
    "LOCK.L": "#D8D8D8",
}

ACCENT = "#0057B8"          # PTEQ accent / highlight color
MUTED  = "#BDBDBD"          # neutral gray for non-PTEQ bars/lines
RISK_FREE_ANNUAL = 0.045    # approximate current risk-free rate

# Default theme mapping (used when theme_map.csv is unavailable / incomplete)
DEFAULT_THEME_MAP: Dict[str, str] = {
    "PLTR": "AI Vision",
    "AXON": "Physical Security",
    "FTNT": "Cybersecurity",
    "PANW": "Cybersecurity",
    "CGNX": "AI Vision",
    "ADT":  "Physical Security",
    "EVLV": "Biometrics",
    "AUID": "Identity",
}

# Default sector mapping (used for the donut chart when not in holdings CSV)
DEFAULT_SECTOR_MAP: Dict[str, str] = {
    "PLTR": "Software",
    "AXON": "Defense",
    "FTNT": "Software",
    "PANW": "Software",
    "CGNX": "Electronic Equipment",
    "ADT":  "Security Services",
    "EVLV": "Electronic Equipment",
    "AUID": "Software",
}

# =============================================================================
# 2) HELPERS  (unchanged originals + new additions)
# =============================================================================

def dedupe_keep_order(seq: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(seq))


def build_equal_weights(tickers: Sequence[str]) -> pd.Series:
    if len(tickers) == 0:
        raise ValueError("No tickers available to weight.")
    return pd.Series(1.0 / len(tickers), index=list(tickers), dtype=float)


def download_prices(
    display_tickers: Sequence[str],
    benchmarks: Sequence[str],
    start: str = START_DATE,
    end: str | None = END_DATE,
) -> pd.DataFrame:
    yf_tickers = dedupe_keep_order(list(display_tickers) + list(benchmarks))

    data = yf.download(
        yf_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        actions=False,
        threads=True,
    )

    if data.empty:
        raise ValueError("No price data returned from yfinance.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            px = data["Close"].copy()
        elif "Adj Close" in data.columns.get_level_values(0):
            px = data["Adj Close"].copy()
        else:
            raise ValueError("Neither Close nor Adj Close field returned by yfinance.")
    else:
        close_col = "Close" if "Close" in data.columns else "Adj Close"
        px = data[[close_col]].copy()
        px.columns = [yf_tickers[0]]

    wanted = dedupe_keep_order(list(display_tickers) + list(benchmarks))
    px = px.reindex(columns=wanted)

    failed = [c for c in px.columns if px[c].isna().all()]
    if failed:
        print("These tickers failed to download and will be skipped:", failed)
        px = px.drop(columns=failed)

    px = px.ffill().dropna(how="any")
    return px


def build_rebalanced_portfolio_index(
    prices: pd.DataFrame,
    weights: pd.Series,
    rebalance_freq: str = REBALANCE_FREQ,
    base_value: float = BASE_VALUE,
) -> pd.Series:
    cols = [c for c in weights.index if c in prices.columns]
    if len(cols) == 0:
        raise ValueError("None of the holdings tickers are present in the price panel.")

    prices = prices[cols].copy().ffill().dropna()
    weights = weights.loc[cols].astype(float)
    weights = weights / weights.sum()

    rebalance_dates = set(prices.groupby(prices.index.to_period(rebalance_freq)).head(1).index)

    shares = None
    value_series = []
    current_value = float(base_value)

    for dt, row in prices.iterrows():
        if shares is None or dt in rebalance_dates:
            shares = (current_value * weights) / row
        current_value = float((shares * row).sum())
        value_series.append(current_value)

    return pd.Series(value_series, index=prices.index, name="PTEQ")


def max_drawdown(series: pd.Series) -> pd.Series:
    running_max = series.cummax()
    return series / running_max - 1.0


def drawdown_episodes(series: pd.Series) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame()

    peak = series.iloc[0]
    peak_date = series.index[0]
    trough_val = peak
    trough_date = peak_date
    underwater = False
    episodes = []

    for dt, val in series.items():
        if val >= peak:
            if underwater:
                episodes.append({
                    "peak_date": peak_date,
                    "trough_date": trough_date,
                    "recovery_date": dt,
                    "max_drawdown": trough_val / peak - 1.0,
                    "recovery_days": (dt - peak_date).days,
                    "days_to_trough": (trough_date - peak_date).days,
                })
                underwater = False
            peak = val
            peak_date = dt
            trough_val = val
            trough_date = dt
        else:
            underwater = True
            if val < trough_val:
                trough_val = val
                trough_date = dt

    return pd.DataFrame(episodes)


def rolling_beta_corr(
    etf_ret: pd.Series, bench_ret: pd.Series, window: int = 63
) -> Tuple[pd.Series, pd.Series]:
    cov  = etf_ret.rolling(window).cov(bench_ret)
    var  = bench_ret.rolling(window).var()
    beta = cov / var
    corr = etf_ret.rolling(window).corr(bench_ret)
    return beta, corr


def annualized_volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)


def annualized_return(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    n_days = (series.index[-1] - series.index[0]).days
    if n_days <= 0:
        return np.nan
    total_return = series.iloc[-1] / series.iloc[0] - 1.0
    years = n_days / 365.25
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def sharpe_ratio(returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    if returns.empty or returns.std(ddof=0) == 0:
        return np.nan
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    excess = returns - rf_daily
    return np.sqrt(252) * excess.mean() / excess.std(ddof=0)


def sortino_ratio(returns: pd.Series, rf_annual: float = RISK_FREE_ANNUAL) -> float:
    if returns.empty:
        return np.nan
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if downside.empty or downside.std(ddof=0) == 0:
        return np.nan
    return np.sqrt(252) * excess.mean() / downside.std(ddof=0)


def calmar_ratio(series: pd.Series) -> float:
    ann_ret = annualized_return(series)
    mdd = max_drawdown(series).min()
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return ann_ret / abs(mdd)


def alpha_beta(
    etf_ret: pd.Series, bench_ret: pd.Series, rf_annual: float = RISK_FREE_ANNUAL
) -> Tuple[float, float]:
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    aligned = pd.concat([etf_ret, bench_ret], axis=1).dropna()
    if aligned.empty:
        return np.nan, np.nan
    y = aligned.iloc[:, 0] - rf_daily
    x = aligned.iloc[:, 1] - rf_daily
    beta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    alpha_daily = y.mean() - beta * x.mean()
    alpha_ann = alpha_daily * 252
    return alpha_ann, beta


def holdings_overlap(
    holdings: Sequence[str], etf_holdings: Dict[str, Sequence[str]]
) -> Dict[str, float]:
    holding_set = set(holdings)
    result = {}
    for etf_name, etf_list in etf_holdings.items():
        overlap = holding_set & set(etf_list)
        result[etf_name] = len(overlap) / len(holding_set) if holding_set else 0.0
    return result


def save_fig(fig: plt.Figure, filename: str) -> None:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def get_benchmark_panel(
    prices: pd.DataFrame, benchmarks: Sequence[str], base_value: float = BASE_VALUE
) -> pd.DataFrame:
    existing = [c for c in benchmarks if c in prices.columns]
    if not existing:
        raise ValueError("None of the requested benchmarks were downloaded successfully.")
    bench = prices[existing].div(prices[existing].iloc[0]) * base_value
    bench.columns = existing
    return bench


def wrap_label(label: str) -> str:
    words = label.split()
    if len(words) <= 1:
        return label
    total  = sum(len(w) for w in words) + len(words) - 1
    target = total / 2
    best_split = 1
    running = len(words[0])
    best_diff = abs(running - target)
    for i in range(1, len(words)):
        running += 1 + len(words[i])
        diff = abs(running - target)
        if diff < best_diff:
            best_diff = diff
            best_split = i + 1
    line1 = " ".join(words[:best_split])
    line2 = " ".join(words[best_split:])
    return f"{line1}\n{line2}" if line2 else line1


# ── NEW HELPER ────────────────────────────────────────────────────────────────
def style_kpi_card(ax: plt.Axes, label: str, value: str) -> None:
    """Render a single KPI card inside a given Axes."""
    ax.set_facecolor("#EEF4FF")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#AAAAAA")
    ax.text(0.5, 0.62, value, transform=ax.transAxes,
            ha="center", va="center", fontsize=20, fontweight="bold", color=ACCENT)
    ax.text(0.5, 0.24, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color="#555555")


# =============================================================================
# 3) DATA + PORTFOLIO BUILD
# =============================================================================

def main() -> None:
    holdings_df = pd.read_csv(HOLDINGS_CSV)
    theme_df    = pd.read_csv(THEME_MAP_CSV)

    holdings_df = holdings_df.drop_duplicates(subset="ticker")
    raw_holdings = holdings_df["ticker"].astype(str).tolist()
    holdings = dedupe_keep_order(raw_holdings)

    if "weight" in holdings_df.columns:
        weights_raw = pd.Series(
            holdings_df["weight"].astype(float).values,
            index=holdings_df["ticker"].astype(str).values,
            dtype=float,
        )
        weights_raw = weights_raw / weights_raw.sum()
        weights = weights_raw.reindex(holdings).fillna(0.0)
        if weights.sum() <= 0:
            weights = build_equal_weights(holdings)
        else:
            weights = weights / weights.sum()
    else:
        weights = build_equal_weights(holdings)

    # Theme map: CSV first, fall back to built-in defaults
    theme_map: Dict[str, str] = {**DEFAULT_THEME_MAP}
    theme_map.update(
        pd.Series(
            theme_df["theme"].values,
            index=theme_df["ticker"].astype(str).values,
        ).to_dict()
    )

    # Sector map: CSV column "sector" if present, else built-in defaults
    sector_map: Dict[str, str] = {**DEFAULT_SECTOR_MAP}
    if "sector" in holdings_df.columns:
        sector_map.update(
            pd.Series(
                holdings_df["sector"].values,
                index=holdings_df["ticker"].astype(str).values,
            ).to_dict()
        )

    prices = download_prices(holdings, BENCHMARKS, start=START_DATE, end=END_DATE)

    available_holdings = [t for t in holdings if t in prices.columns]
    missing_holdings   = [t for t in holdings if t not in prices.columns]
    if missing_holdings:
        print("Missing holdings after download:", missing_holdings)

    if len(available_holdings) == 0:
        raise ValueError("No holdings survived the download.")

    holdings = available_holdings
    weights  = weights.loc[holdings]
    weights  = weights / weights.sum()

    etf = build_rebalanced_portfolio_index(
        prices[holdings],
        weights,
        rebalance_freq=REBALANCE_FREQ,
        base_value=BASE_VALUE,
    ).rename("PTEQ")

    benchmarks_panel = get_benchmark_panel(prices, BENCHMARKS, base_value=BASE_VALUE)
    combined = pd.concat([etf, benchmarks_panel], axis=1).dropna()
    ret      = combined.pct_change().dropna()
    etf_ret  = ret["PTEQ"]

    # =========================================================================
    # CHART 1 — INDEXED PERFORMANCE (PTEQ highlighted, benchmarks in gray)
    # CHANGE: PTEQ bold blue on top (zorder=10), benchmarks muted gray
    # =========================================================================
    benchmarks_in_panel = [c for c in benchmarks_panel.columns if c in combined.columns]

    fig, ax = plt.subplots(figsize=(12, 6))

    for col in benchmarks_in_panel:                      # benchmarks behind PTEQ
        ax.plot(
            combined.index, combined[col],
            label=col,
            color=COLORS.get(col, MUTED),
            linewidth=1.5, alpha=0.65, zorder=1,
        )

    ax.plot(                                             # PTEQ on top, thick + bold
        combined.index, combined["PTEQ"],
        label="PTEQ",
        color=ACCENT,
        linewidth=3.5, zorder=10,
    )

    ax.set_title("Indexed Performance: PTEQ vs Benchmarks (Base = 100)",
                 fontsize=16, fontweight="bold")
    ax.set_ylabel("Indexed Value")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    save_fig(fig, "01_indexed_performance.png")

    # =========================================================================
    # CHART 2 — RELATIVE STRENGTH: 5 SMALL MULTIPLES
    # CHANGE: one sub-panel per benchmark, shared styling, fill above/below 1.0
    # =========================================================================
    bench_cols = list(benchmarks_in_panel)
    n    = len(bench_cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        sharex=True, sharey=False,
    )
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, b in enumerate(bench_cols):
        ax    = axes_flat[i]
        ratio = combined["PTEQ"] / combined[b]
        ax.plot(ratio.index, ratio, color=ACCENT, linewidth=2)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.fill_between(ratio.index, ratio, 1.0,
                        where=(ratio >= 1.0), alpha=0.15, color=ACCENT)
        ax.fill_between(ratio.index, ratio, 1.0,
                        where=(ratio <  1.0), alpha=0.10, color="#EF5350")
        ax.set_title(f"PTEQ / {b}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Ratio")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelrotation=30)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Relative Strength: PTEQ vs Each Benchmark",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "02_relative_strength.png")

    # =========================================================================
    # CHART 3 — VOLATILITY BAR CHART (252-day annualized, PTEQ accented)
    # CHANGE: bar chart replaces rolling line; PTEQ = ACCENT, others = MUTED
    # =========================================================================
    vol_252: Dict[str, float] = {}
    vol_252["PTEQ"] = annualized_volatility(etf_ret, 252).iloc[-1]
    for b in bench_cols:
        if b in ret.columns:
            vol_252[b] = annualized_volatility(ret[b], 252).iloc[-1]

    vol_series = pd.Series(vol_252)
    bar_colors = [ACCENT if t == "PTEQ" else COLORS.get(t, MUTED) for t in vol_series.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(vol_series.index, vol_series.values * 100,
                  color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Current 252-Day Annualized Volatility", fontsize=14, fontweight="bold")
    ax.set_ylabel("Volatility (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, axis="y", alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    save_fig(fig, "03_volatility_bar.png")

    # =========================================================================
    # CHART 4 — DRAWDOWN + SUMMARY TABLE
    # CHANGE: table now populates all benchmark rows (not just PTEQ)
    # =========================================================================
    dd_panel = pd.DataFrame(index=combined.index)
    for col in combined.columns:
        dd_panel[col] = max_drawdown(combined[col]) * 100.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 9),
                              gridspec_kw={"height_ratios": [3, 1]})

    ax_dd = axes[0]
    for col in dd_panel.columns:
        lw    = 3.0  if col == "PTEQ" else 1.5
        alpha = 1.0  if col == "PTEQ" else 0.65
        z     = 10   if col == "PTEQ" else 1
        ax_dd.plot(dd_panel.index, dd_panel[col], label=col,
                   color=COLORS.get(col, MUTED), linewidth=lw, alpha=alpha, zorder=z)
    ax_dd.set_title("Drawdown Comparison: PTEQ vs Benchmarks",
                    fontsize=14, fontweight="bold")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.grid(True, alpha=0.3)
    ax_dd.legend(ncol=2)

    ax_tbl = axes[1]
    ax_tbl.axis("off")
    mdd_rows = [(col, f"{dd_panel[col].min():.1f}%") for col in dd_panel.columns]
    tbl = ax_tbl.table(
        cellText=[[r[1] for r in mdd_rows]],
        colLabels=[r[0] for r in mdd_rows],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)
    # Highlight PTEQ column
    pteq_col_idx = [r[0] for r in mdd_rows].index("PTEQ")
    tbl[(0, pteq_col_idx)].set_facecolor(ACCENT)
    tbl[(0, pteq_col_idx)].set_text_props(color="white", fontweight="bold")
    tbl[(1, pteq_col_idx)].set_facecolor("#E8F0FE")
    tbl[(1, pteq_col_idx)].set_text_props(fontweight="bold")
    ax_tbl.set_title("Maximum Drawdown by ETF", fontsize=11, pad=4)

    fig.tight_layout()
    save_fig(fig, "04_drawdown_comparison.png")

    # =========================================================================
    # CHART 5 — ROLLING BETA vs SPY + KPI CARDS
    # CHANGE: three KPI cards (Current / Average / Maximum) below the chart
    # =========================================================================
    if "SPY" in ret.columns:
        beta63_spy,  corr63_spy  = rolling_beta_corr(etf_ret, ret["SPY"], window=63)
        beta252_spy, corr252_spy = rolling_beta_corr(etf_ret, ret["SPY"], window=252)

        fig = plt.figure(figsize=(12, 10))
        gs  = gridspec.GridSpec(3, 3, figure=fig,
                                height_ratios=[4, 1, 0.1],
                                hspace=0.45)

        ax_beta = fig.add_subplot(gs[0, :])
        ax_beta.plot(beta63_spy.index,  beta63_spy,
                     label="63D Beta vs SPY",  color="#4C9BE8", linewidth=1.8, alpha=0.8)
        ax_beta.plot(beta252_spy.index, beta252_spy,
                     label="252D Beta vs SPY", color=ACCENT,    linewidth=2.2)
        ax_beta.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax_beta.set_title("Rolling Beta vs SPY", fontsize=14, fontweight="bold")
        ax_beta.set_ylabel("Beta")
        ax_beta.grid(True, alpha=0.3)
        ax_beta.legend()

        beta_clean = beta252_spy.dropna()
        kpi_values = {
            "Current Beta": f"{beta_clean.iloc[-1]:.2f}",
            "Average Beta": f"{beta_clean.mean():.2f}",
            "Maximum Beta": f"{beta_clean.max():.2f}",
        }
        for idx, (label, value) in enumerate(kpi_values.items()):
            ax_kpi = fig.add_subplot(gs[1, idx])
            style_kpi_card(ax_kpi, label, value)

        save_fig(fig, "05_rolling_beta_vs_spy.png")

        # =====================================================================
        # CHART 6 — ROLLING CORRELATION vs SPY
        # CHANGE: subtitle emphasises "moderately differentiated exposure"
        # =====================================================================
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(corr63_spy.index,  corr63_spy,
                label="63D Corr vs SPY",  color="#4C9BE8", linewidth=1.8, alpha=0.8)
        ax.plot(corr252_spy.index, corr252_spy,
                label="252D Corr vs SPY", color=ACCENT,    linewidth=2.2)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.fill_between(corr252_spy.index, corr252_spy, 0,
                        where=(corr252_spy > 0), alpha=0.08, color=ACCENT)
        ax.set_title(
            "Rolling Correlation vs SPY",
            fontsize=14, fontweight="bold",
        )
        ax.set_xlabel(
            "PTEQ provides moderately differentiated exposure — partial overlap "
            "with broad market but meaningful independent return drivers",
            fontsize=9, color="#555555",
        )
        ax.set_ylabel("Correlation")
        ax.set_ylim(-0.2, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()

        avg_corr = corr252_spy.dropna().mean()
        ax.annotate(
            f"Avg 252D corr: {avg_corr:.2f}\n(moderately differentiated exposure)",
            xy=(corr252_spy.index[len(corr252_spy) // 2], avg_corr),
            xytext=(corr252_spy.index[len(corr252_spy) // 4], avg_corr + 0.12),
            fontsize=9, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#666666"),
        )

        save_fig(fig, "06_rolling_correlation_vs_spy.png")

    if "LOCK.L" in ret.columns:
        beta63_lock,  corr63_lock  = rolling_beta_corr(etf_ret, ret["LOCK.L"], window=63)
        beta252_lock, corr252_lock = rolling_beta_corr(etf_ret, ret["LOCK.L"], window=252)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(beta63_lock.index,  beta63_lock,  label="63D Beta vs LOCK.L",  linewidth=1.8)
        ax.plot(beta252_lock.index, beta252_lock, label="252D Beta vs LOCK.L", linewidth=2.2)
        ax.axhline(1.0, linestyle="--", color="gray")
        ax.set_title("Rolling Beta vs LOCK.L")
        ax.set_ylabel("Beta")
        ax.grid(True, alpha=0.3)
        ax.legend()
        save_fig(fig, "07_rolling_beta_vs_lock.png")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(corr63_lock.index,  corr63_lock,  label="63D Corr vs LOCK.L",  linewidth=1.8)
        ax.plot(corr252_lock.index, corr252_lock, label="252D Corr vs LOCK.L", linewidth=2.2)
        ax.axhline(0.0, linestyle="--", color="gray")
        ax.set_title("Rolling Correlation vs LOCK.L")
        ax.set_ylabel("Correlation")
        ax.grid(True, alpha=0.3)
        ax.legend()
        save_fig(fig, "08_rolling_correlation_vs_lock.png")

    # =========================================================================
    # CHART 7 (was 9) — RISK-ADJUSTED METRICS TABLE  [extended]
    # CHANGE: adds Sortino, Calmar, Alpha, Beta; highlights PTEQ row
    # =========================================================================
    rows = []
    for col in combined.columns:
        r = ret[col] if col in ret.columns else pd.Series(dtype=float)
        s = combined[col]
        mdd = max_drawdown(s).min()
        ann_alpha, beta_val = (np.nan, np.nan)
        if col == "PTEQ" and "SPY" in ret.columns:
            ann_alpha, beta_val = alpha_beta(etf_ret, ret["SPY"])
        row = {
            "ETF":             col,
            "CAGR":            annualized_return(s),
            "Ann Vol":         r.std(ddof=0) * np.sqrt(252) if not r.empty else np.nan,
            "Sharpe":          sharpe_ratio(r),
            "Sortino":         sortino_ratio(r),
            "Max DD":          mdd,
            "Calmar":          calmar_ratio(s),
            "Alpha (vs SPY)":  ann_alpha,
            "Beta (vs SPY)":   beta_val,
        }
        rows.append(row)

    metrics_df = pd.DataFrame(rows).set_index("ETF")

    fmt_pct = lambda x: f"{x:.1%}" if pd.notna(x) else "—"
    fmt_2f  = lambda x: f"{x:.2f}"  if pd.notna(x) else "—"
    disp = metrics_df.copy()
    disp["CAGR"]           = disp["CAGR"].map(fmt_pct)
    disp["Ann Vol"]        = disp["Ann Vol"].map(fmt_pct)
    disp["Sharpe"]         = disp["Sharpe"].map(fmt_2f)
    disp["Sortino"]        = disp["Sortino"].map(fmt_2f)
    disp["Max DD"]         = disp["Max DD"].map(fmt_pct)
    disp["Calmar"]         = disp["Calmar"].map(fmt_2f)
    disp["Alpha (vs SPY)"] = disp["Alpha (vs SPY)"].map(fmt_pct)
    disp["Beta (vs SPY)"]  = disp["Beta (vs SPY)"].map(fmt_2f)

    print("\n=== Risk-Adjusted Metrics Summary ===")
    print(disp.to_string())

    metrics_path = OUTPUT_DIR / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path)
    print(f"Saved {metrics_path}")

    fig, ax = plt.subplots(figsize=(15, max(3, len(disp) * 0.65 + 1.5)))
    ax.axis("off")
    tbl = ax.table(
        cellText=disp.reset_index().values,
        colLabels=["ETF"] + list(disp.columns),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)

    # Header row
    for j in range(len(disp.columns) + 1):
        tbl[(0, j)].set_facecolor(ACCENT)
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # PTEQ row highlight
    pteq_row_idx = list(disp.index).index("PTEQ") + 1
    for j in range(len(disp.columns) + 1):
        tbl[(pteq_row_idx, j)].set_facecolor("#E8F0FE")
        tbl[(pteq_row_idx, j)].set_text_props(fontweight="bold")

    ax.set_title("Risk-Adjusted Metrics: PTEQ vs Benchmarks",
                 fontsize=14, fontweight="bold", pad=16)
    save_fig(fig, "12_metrics_table.png")

    snapshot_path = OUTPUT_DIR / "snapshot.csv"
    disp.to_csv(snapshot_path)
    print(f"Saved snapshot -> {snapshot_path}")

    # =========================================================================
    # CHART 8 — CONCENTRATION (theme + top 10)  [preserved, minor cleanup]
    # =========================================================================
    theme_series = pd.Series(
        {t: theme_map.get(t, "Unclassified") for t in weights.index}, name="Theme"
    )
    theme_weights = weights.groupby(theme_series).sum().sort_values(ascending=False)
    top10 = weights.sort_values(ascending=False).head(10)
    wrapped_theme_labels = [wrap_label(lbl) for lbl in theme_weights.index]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bars0 = axes[0].bar(range(len(theme_weights)), theme_weights.values * 100,
                        color="#4C72B0", edgecolor="white", linewidth=0.6)
    axes[0].set_title("Weights by Theme / Sector", fontsize=13, pad=12)
    axes[0].set_ylabel("Weight (%)")
    axes[0].set_xticks(range(len(theme_weights)))
    axes[0].set_xticklabels(wrapped_theme_labels, rotation=45, ha="center",
                             fontsize=8.5, linespacing=1.4)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    axes[0].grid(True, axis="y", alpha=0.3)
    for bar in bars0:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.1f}%",
                     ha="center", va="bottom", fontsize=8)

    bars1 = axes[1].bar(range(len(top10)), top10.values * 100,
                        color="#55A868", edgecolor="white", linewidth=0.6)
    axes[1].set_title("Top 10 Holdings by Weight", fontsize=13, pad=12)
    axes[1].set_ylabel("Weight (%)")
    axes[1].set_xticks(range(len(top10)))
    axes[1].set_xticklabels(top10.index, rotation=0, ha="center", fontsize=9)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    axes[1].grid(True, axis="y", alpha=0.3)
    for bar in bars1:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.1f}%",
                     ha="center", va="bottom", fontsize=8)

    fig.subplots_adjust(bottom=0.22, wspace=0.30)
    save_fig(fig, "09_concentration.png")

    # =========================================================================
    # CHART 9 (NEW) — SECTOR EXPOSURE DONUT CHART
    # =========================================================================
    sector_series = pd.Series(
        {t: sector_map.get(t, "Other") for t in weights.index}, name="Sector"
    )
    sector_weights = weights.groupby(sector_series).sum().sort_values(ascending=False)

    sector_palette = [
        "#0057B8", "#4C9BE8", "#A8D1F5", "#55A868",
        "#C44E52", "#8172B2", "#CCB974", "#64B5CD",
    ]
    n_sectors = len(sector_weights)
    sector_colors = [sector_palette[i % len(sector_palette)] for i in range(n_sectors)]

    fig, ax = plt.subplots(figsize=(8, 7))
    wedges, texts, autotexts = ax.pie(
        sector_weights.values,
        labels=sector_weights.index,
        autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
        colors=sector_colors,
        startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5),
    )
    for t in autotexts:
        t.set_fontsize(8)
        t.set_fontweight("bold")
    ax.set_title("Sector Exposure", fontsize=14, fontweight="bold", pad=16)
    save_fig(fig, "13_sector_exposure_donut.png")

    # =========================================================================
    # CHART 10 (NEW) — PERFORMANCE ATTRIBUTION: TOP / BOTTOM CONTRIBUTORS
    # =========================================================================
    px_sub = prices[holdings].ffill().dropna()
    if len(px_sub) >= 2:
        total_returns = px_sub.iloc[-1] / px_sub.iloc[0] - 1.0
        contrib = (
            total_returns * weights.reindex(px_sub.columns).fillna(0)
        ).dropna().sort_values(ascending=False)

        top_n     = 8
        top_contrib = contrib.head(top_n)
        bot_contrib = contrib.tail(top_n)
        attr_df   = pd.concat([top_contrib, bot_contrib]).drop_duplicates()

        bar_c = ["#2196F3" if v >= 0 else "#EF5350" for v in attr_df.values]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: top contributors
        top_df = attr_df[attr_df >= 0].sort_values(ascending=True)
        axes[0].barh(top_df.index, top_df.values * 100,
                     color="#2196F3", edgecolor="white", linewidth=0.5)
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set_title("Top Contributors (Weighted Return)",
                          fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Contribution to Total Return (%)")
        axes[0].grid(True, axis="x", alpha=0.3)

        # Right: bottom contributors
        bot_df = attr_df[attr_df < 0].sort_values(ascending=False)
        axes[1].barh(bot_df.index, bot_df.values * 100,
                     color="#EF5350", edgecolor="white", linewidth=0.5)
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_title("Bottom Contributors (Weighted Return)",
                          fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Contribution to Total Return (%)")
        axes[1].grid(True, axis="x", alpha=0.3)

        fig.suptitle("Performance Attribution", fontsize=14, fontweight="bold")
        fig.tight_layout()
        save_fig(fig, "10_performance_attribution.png")

    # =========================================================================
    # CHART 11 (NEW) — HOLDINGS OVERLAP vs PEER ETFs
    # Requires peer_holdings.csv [etf, ticker]; generates chart or prints notice
    # =========================================================================
    peer_file = Path("peer_holdings.csv")
    if peer_file.exists():
        peer_df = pd.read_csv(peer_file)
        peer_dict: Dict[str, List[str]] = (
            peer_df.groupby("etf")["ticker"].apply(list).to_dict()
        )
        overlap        = holdings_overlap(holdings, peer_dict)
        overlap_series = pd.Series(overlap).sort_values()

        fig, ax = plt.subplots(figsize=(8, 5))
        bar_c2 = [COLORS.get(e, MUTED) for e in overlap_series.index]
        bars   = ax.barh(overlap_series.index, overlap_series.values * 100,
                         color=bar_c2, edgecolor="white", linewidth=0.6)
        ax.set_title("Holdings Overlap: PTEQ vs Peer ETFs",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Overlap (%)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(True, axis="x", alpha=0.3)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}%", va="center", fontsize=9)

        unique_pct = (1 - max(overlap.values())) * 100 if overlap else 100.0
        ax.annotate(
            f"~{unique_pct:.0f}% of PTEQ holdings\nnot in largest peer ETF",
            xy=(0.97, 0.05), xycoords="axes fraction",
            ha="right", fontsize=9, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEF4FF", edgecolor="#AAAAAA"),
        )
        save_fig(fig, "11_holdings_overlap.png")
    else:
        print("peer_holdings.csv not found — skipping holdings overlap chart (Chart 11).")
        print("  Create peer_holdings.csv with columns [etf, ticker] to enable this chart.")

    print("\nAll charts and data saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()