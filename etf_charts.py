"""Build the PTEQ portfolio index, compare it with benchmark ETFs, and export charts.

Run from anywhere with ``python etf_charts.py``. Outputs (PNG charts, CSV
summaries and ``run_info.json``) are written to ``etf_output/``. Portfolio
construction and all metrics live in ``analytics.py``; this file handles data
download, input files, and plotting.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.ticker import PercentFormatter

from analytics import (
    BASE_VALUE,
    MIN_WEIGHT_COVERAGE,
    REBALANCE_FREQ,
    RISK_FREE_ANNUAL,
    PortfolioResult,
    annualized_volatility,
    build_equal_weights,
    build_metrics_table,
    build_rebalanced_portfolio,
    dedupe_keep_order,
    describe_correlation,
    drawdown_series,
    holding_contributions,
    holdings_overlap,
    rolling_beta_corr,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "etf_output"
HOLDINGS_CSV = BASE_DIR / "holdings.csv"
THEME_MAP_CSV = BASE_DIR / "theme_map.csv"
PEER_HOLDINGS_CSV = BASE_DIR / "peer_holdings.csv"

START_DATE = "2020-01-01"  # first date requested from Yahoo Finance
END_DATE: str | None = None
BENCHMARKS = ["SPY", "LOCK.L", "CIBR", "HACK", "BUG"]
PORTFOLIO_NAME = "PTEQ"
MARKET_BENCHMARK = "SPY"
ROLLING_WINDOWS = (63, 252)
REBALANCE_LABELS = {"M": "Monthly", "Q": "Quarterly", "Y": "Annual", "A": "Annual"}

ACCENT = "#0057B8"  # the portfolio is always drawn in this blue
MUTED = "#BDBDBD"
# Benchmarks use distinct, colour-blind-friendly hues so the legend is readable.
COLORS = {
    PORTFOLIO_NAME: ACCENT,
    "SPY": "#555555",
    "CIBR": "#E69F00",
    "HACK": "#009E73",
    "BUG": "#CC79A7",
    "LOCK.L": "#D55E00",
}
SECTOR_PALETTE = [
    "#0057B8",
    "#4C9BE8",
    "#A8D1F5",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#CCB974",
    "#64B5CD",
]


# =============================================================================
# Inputs and data download
# =============================================================================


@dataclass
class Inputs:
    weights: pd.Series
    theme_map: dict[str, str]
    sector_map: dict[str, str]


def load_inputs(holdings_csv: Path = HOLDINGS_CSV, theme_csv: Path = THEME_MAP_CSV) -> Inputs:
    """Read target weights (holdings.csv) and theme/sector labels (theme_map.csv)."""
    holdings_df = pd.read_csv(holdings_csv).drop_duplicates(subset="ticker")
    tickers = dedupe_keep_order(holdings_df["ticker"].astype(str).tolist())

    if "weight" in holdings_df.columns:
        weights = pd.Series(
            holdings_df["weight"].astype(float).to_numpy(),
            index=holdings_df["ticker"].astype(str).to_numpy(),
        ).reindex(tickers)
        weights = weights / weights.sum() if weights.sum() > 0 else build_equal_weights(tickers)
    else:
        weights = build_equal_weights(tickers)

    theme_df = pd.read_csv(theme_csv).drop_duplicates(subset="ticker").set_index("ticker")
    theme_map = theme_df["theme"].to_dict()
    sector_map = theme_df["sector"].to_dict() if "sector" in theme_df.columns else {}

    unmapped = [t for t in tickers if t not in theme_map]
    if unmapped:
        logger.warning(
            "Holdings missing from %s (shown as 'Unclassified'): %s", theme_csv.name, unmapped
        )
    return Inputs(weights=weights, theme_map=theme_map, sector_map=sector_map)


def download_prices(
    holdings: Sequence[str],
    benchmarks: Sequence[str],
    start: str = START_DATE,
    end: str | None = END_DATE,
) -> pd.DataFrame:
    """Download adjusted closes. Tickers keep their own start dates (NaN before listing)."""
    wanted = dedupe_keep_order(list(holdings) + list(benchmarks))
    data = yf.download(
        wanted,
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
        field = "Close" if "Close" in data.columns.get_level_values(0) else "Adj Close"
        px = data[field].copy()
    else:
        field = "Close" if "Close" in data.columns else "Adj Close"
        px = data[[field]].copy()
        px.columns = [wanted[0]]

    px = px.reindex(columns=wanted)
    failed = [c for c in px.columns if px[c].isna().all()]
    if failed:
        logger.warning("These tickers failed to download and will be skipped: %s", failed)
        px = px.drop(columns=failed)
    return px.dropna(how="all").ffill()


# =============================================================================
# Analysis container
# =============================================================================


@dataclass
class Analysis:
    prices: pd.DataFrame
    portfolio: PortfolioResult
    levels: pd.DataFrame  # PTEQ + benchmarks, rebased to BASE_VALUE
    returns: pd.DataFrame
    weights: pd.Series  # weights of holdings with price data
    theme_map: dict[str, str]
    sector_map: dict[str, str]
    benchmarks: list[str]
    note: str  # footnote printed under each chart

    @property
    def etf_ret(self) -> pd.Series:
        return self.returns[PORTFOLIO_NAME]


def prepare_analysis(prices: pd.DataFrame, inputs: Inputs) -> Analysis:
    holdings = [t for t in inputs.weights.index if t in prices.columns]
    missing = [t for t in inputs.weights.index if t not in prices.columns]
    if missing:
        logger.warning("Missing holdings after download: %s", missing)
    if not holdings:
        raise ValueError("No holdings survived the download.")

    weights = inputs.weights.loc[holdings]
    weights = weights / weights.sum()

    portfolio = build_rebalanced_portfolio(prices[holdings], weights)

    benchmarks = [b for b in BENCHMARKS if b in prices.columns]
    if not benchmarks:
        raise ValueError("None of the requested benchmarks were downloaded successfully.")

    combined = pd.concat(
        [portfolio.index, prices[benchmarks].ffill().loc[portfolio.start :]], axis=1
    ).dropna()
    levels = combined.div(combined.iloc[0]) * BASE_VALUE
    returns = levels.pct_change().dropna()

    note = (
        f"{levels.index[0]:%Y-%m-%d} to {levels.index[-1]:%Y-%m-%d} | "
        f"{REBALANCE_LABELS.get(REBALANCE_FREQ, REBALANCE_FREQ)} rebalancing, "
        "no transaction costs | "
        "Yahoo Finance adjusted prices | Research only, not investment advice"
    )
    return Analysis(
        prices,
        portfolio,
        levels,
        returns,
        weights,
        inputs.theme_map,
        inputs.sector_map,
        benchmarks,
        note,
    )


# =============================================================================
# Plot helpers
# =============================================================================


def save_fig(fig: plt.Figure, filename: str, note: str | None = None) -> None:
    if note:
        fig.text(0.5, -0.01, note, ha="center", va="top", fontsize=7, color="#777777")
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def wrap_label(label: str) -> str:
    """Split a label into two balanced lines."""
    words = label.split()
    if len(words) <= 1:
        return label
    target = (sum(len(w) for w in words) + len(words) - 1) / 2
    best_split, running = 1, len(words[0])
    best_diff = abs(running - target)
    for i in range(1, len(words)):
        running += 1 + len(words[i])
        if abs(running - target) < best_diff:
            best_diff, best_split = abs(running - target), i + 1
    return f"{' '.join(words[:best_split])}\n{' '.join(words[best_split:])}"


def style_kpi_card(ax: plt.Axes, label: str, value: str) -> None:
    """Render a single KPI card inside ``ax``."""
    ax.set_facecolor("#EEF4FF")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#AAAAAA")
    ax.text(
        0.5,
        0.62,
        value,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=ACCENT,
    )
    ax.text(
        0.5,
        0.24,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )


def color_for(name: str) -> str:
    return COLORS.get(name, MUTED)


# =============================================================================
# Charts
# =============================================================================


def plot_indexed_performance(a: Analysis) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in a.benchmarks:
        ax.plot(
            a.levels.index,
            a.levels[col],
            label=col,
            color=color_for(col),
            linewidth=1.5,
            alpha=0.8,
            zorder=1,
        )
    ax.plot(
        a.levels.index,
        a.levels[PORTFOLIO_NAME],
        label=PORTFOLIO_NAME,
        color=ACCENT,
        linewidth=3.5,
        zorder=10,
    )
    ax.set_title(
        f"Indexed Performance: {PORTFOLIO_NAME} vs Benchmarks (Base = 100)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_ylabel("Indexed Value")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    save_fig(fig, "01_indexed_performance.png", a.note)


def plot_relative_strength(a: Analysis) -> None:
    n = len(a.benchmarks)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharex=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, bench in zip(axes_flat, a.benchmarks):
        ratio = a.levels[PORTFOLIO_NAME] / a.levels[bench]
        ax.plot(ratio.index, ratio, color=ACCENT, linewidth=2)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.fill_between(ratio.index, ratio, 1.0, where=(ratio >= 1.0), alpha=0.15, color=ACCENT)
        ax.fill_between(ratio.index, ratio, 1.0, where=(ratio < 1.0), alpha=0.10, color="#EF5350")
        ax.set_title(f"{PORTFOLIO_NAME} / {bench}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Ratio")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelrotation=30)
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Relative Strength: {PORTFOLIO_NAME} vs Each Benchmark",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "02_relative_strength.png", a.note)


def plot_volatility_bar(a: Analysis) -> None:
    window = min(252, len(a.returns))
    vols = (
        pd.Series(
            {
                col: annualized_volatility(a.returns[col], window).iloc[-1]
                for col in [PORTFOLIO_NAME, *a.benchmarks]
            }
        )
        * 100
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        vols.index,
        vols.values,
        color=[color_for(t) for t in vols.index],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_title(f"Current {window}-Day Annualized Volatility", fontsize=14, fontweight="bold")
    ax.set_ylabel("Volatility (%)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(True, axis="y", alpha=0.3)
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9, fontweight="bold")
    save_fig(fig, "03_volatility_bar.png", a.note)


def plot_drawdown(a: Analysis) -> None:
    dd = a.levels.apply(drawdown_series) * 100.0

    fig, (ax_dd, ax_tbl) = plt.subplots(
        2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [3, 1]}
    )
    for col in dd.columns:
        is_port = col == PORTFOLIO_NAME
        ax_dd.plot(
            dd.index,
            dd[col],
            label=col,
            color=color_for(col),
            linewidth=3.0 if is_port else 1.5,
            alpha=1.0 if is_port else 0.8,
            zorder=10 if is_port else 1,
        )
    ax_dd.set_title(
        f"Drawdown Comparison: {PORTFOLIO_NAME} vs Benchmarks", fontsize=14, fontweight="bold"
    )
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.grid(True, alpha=0.3)
    ax_dd.legend(ncol=2)

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=[[f"{dd[c].min():.1f}%" for c in dd.columns]],
        colLabels=list(dd.columns),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    idx = list(dd.columns).index(PORTFOLIO_NAME)
    table[(0, idx)].set_facecolor(ACCENT)
    table[(0, idx)].set_text_props(color="white", fontweight="bold")
    table[(1, idx)].set_facecolor("#E8F0FE")
    table[(1, idx)].set_text_props(fontweight="bold")
    ax_tbl.set_title("Maximum Drawdown by ETF", fontsize=11, pad=4)

    fig.tight_layout()
    save_fig(fig, "04_drawdown_comparison.png", a.note)


def plot_rolling_beta(a: Analysis, bench: str, filename: str, show_kpis: bool = False) -> None:
    short, long = ROLLING_WINDOWS
    beta_s, _ = rolling_beta_corr(a.etf_ret, a.returns[bench], short)
    beta_l, _ = rolling_beta_corr(a.etf_ret, a.returns[bench], long)
    long_clean = beta_l.dropna()
    show_kpis = show_kpis and not long_clean.empty

    if show_kpis:
        fig = plt.figure(figsize=(12, 10))
        grid = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[4, 1], hspace=0.35)
        ax = fig.add_subplot(grid[0, :])
    else:
        fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        beta_s.index,
        beta_s,
        label=f"{short}D Beta vs {bench}",
        color="#4C9BE8",
        linewidth=1.8,
        alpha=0.8,
    )
    ax.plot(beta_l.index, beta_l, label=f"{long}D Beta vs {bench}", color=ACCENT, linewidth=2.2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"Rolling Beta vs {bench}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Beta")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if show_kpis:
        kpis = {
            "Current Beta": long_clean.iloc[-1],
            "Average Beta": long_clean.mean(),
            "Maximum Beta": long_clean.max(),
        }
        for i, (label, value) in enumerate(kpis.items()):
            style_kpi_card(fig.add_subplot(grid[1, i]), label, f"{value:.2f}")
    save_fig(fig, filename, a.note)


def plot_rolling_corr(a: Analysis, bench: str, filename: str) -> None:
    short, long = ROLLING_WINDOWS
    _, corr_s = rolling_beta_corr(a.etf_ret, a.returns[bench], short)
    _, corr_l = rolling_beta_corr(a.etf_ret, a.returns[bench], long)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        corr_s.index,
        corr_s,
        label=f"{short}D Corr vs {bench}",
        color="#4C9BE8",
        linewidth=1.8,
        alpha=0.8,
    )
    ax.plot(corr_l.index, corr_l, label=f"{long}D Corr vs {bench}", color=ACCENT, linewidth=2.2)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"Rolling Correlation vs {bench}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Correlation")
    ax.set_ylim(-0.2, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    clean = corr_l.dropna()
    if not clean.empty:
        avg = clean.mean()
        ax.axhline(avg, color=ACCENT, linestyle=":", linewidth=1)
        ax.text(
            0.01,
            0.04,
            f"Average {long}D correlation: {avg:.2f} ({describe_correlation(avg)})",
            transform=ax.transAxes,
            fontsize=9,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEF4FF", edgecolor="#AAAAAA"),
        )
    save_fig(fig, filename, a.note)


def format_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Human-readable version of the metrics table (strings, '—' for missing)."""
    pct = lambda x: f"{x:.1%}" if pd.notna(x) else "—"  # noqa: E731
    num = lambda x: f"{x:.2f}" if pd.notna(x) else "—"  # noqa: E731
    pct_cols = {"CAGR", "Ann Vol", "Max DD"} | {c for c in metrics if c.startswith("Alpha")}
    return metrics.apply(lambda col: col.map(pct if col.name in pct_cols else num))


def plot_metrics_table(a: Analysis, metrics: pd.DataFrame) -> None:
    disp = format_metrics(metrics)
    fig, ax = plt.subplots(figsize=(15, max(3, len(disp) * 0.65 + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=disp.reset_index().to_numpy(),
        colLabels=["ETF", *disp.columns],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    n_cols = len(disp.columns) + 1
    port_row = list(disp.index).index(PORTFOLIO_NAME) + 1
    for j in range(n_cols):
        table[(0, j)].set_facecolor(ACCENT)
        table[(0, j)].set_text_props(color="white", fontweight="bold")
        table[(port_row, j)].set_facecolor("#E8F0FE")
        table[(port_row, j)].set_text_props(fontweight="bold")
    ax.set_title(
        f"Risk-Adjusted Metrics: {PORTFOLIO_NAME} vs Benchmarks",
        fontsize=14,
        fontweight="bold",
        pad=16,
    )
    save_fig(fig, "12_metrics_table.png", a.note)


def plot_concentration(a: Analysis) -> None:
    themes = pd.Series({t: a.theme_map.get(t, "Unclassified") for t in a.weights.index})
    theme_weights = a.weights.groupby(themes).sum().sort_values(ascending=False)
    top10 = a.weights.sort_values(ascending=False).head(10)

    fig, (ax_theme, ax_top) = plt.subplots(1, 2, figsize=(16, 6))

    bars = ax_theme.bar(
        range(len(theme_weights)),
        theme_weights.to_numpy() * 100,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.6,
    )
    ax_theme.set_title("Weights by Theme", fontsize=13, pad=12)
    ax_theme.set_xticks(range(len(theme_weights)))
    ax_theme.set_xticklabels(
        [wrap_label(t) for t in theme_weights.index],
        rotation=45,
        ha="center",
        fontsize=8.5,
        linespacing=1.4,
    )
    ax_theme.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)

    bars = ax_top.bar(
        range(len(top10)), top10.to_numpy() * 100, color="#55A868", edgecolor="white", linewidth=0.6
    )
    ax_top.set_title("Top 10 Holdings by Weight", fontsize=13, pad=12)
    ax_top.set_xticks(range(len(top10)))
    ax_top.set_xticklabels(top10.index, fontsize=9)
    ax_top.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)

    for ax in (ax_theme, ax_top):
        ax.set_ylabel("Weight (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(True, axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.22, wspace=0.30)
    save_fig(fig, "09_concentration.png")


def plot_sector_donut(a: Analysis) -> None:
    sectors = pd.Series({t: a.sector_map.get(t, "Unclassified") for t in a.weights.index})
    sector_weights = a.weights.groupby(sectors).sum().sort_values(ascending=False)
    colors = [SECTOR_PALETTE[i % len(SECTOR_PALETTE)] for i in range(len(sector_weights))]

    fig, ax = plt.subplots(figsize=(9, 7))
    _, _, autotexts = ax.pie(
        sector_weights.to_numpy(),
        labels=sector_weights.index,
        autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
        colors=colors,
        startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5),
    )
    for text in autotexts:
        text.set_fontsize(8)
        text.set_fontweight("bold")
        text.set_color("white")
    ax.set_title("Sector Exposure (target weights)", fontsize=14, fontweight="bold", pad=16)
    save_fig(fig, "13_sector_exposure_donut.png")


def plot_attribution(a: Analysis, top_n: int = 8) -> None:
    contrib = holding_contributions(a.prices, a.portfolio.shares).sort_values(ascending=False)
    total = a.portfolio.index.iloc[-1] / a.portfolio.index.iloc[0] - 1.0
    gains = contrib[contrib > 0].head(top_n)
    losses = contrib[contrib < 0].tail(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, series, color, title in (
        (axes[0], gains.sort_values(), "#2196F3", "Top Contributors"),
        (axes[1], losses.sort_values(ascending=False), "#EF5350", "Bottom Contributors"),
    ):
        if series.empty:
            ax.text(0.5, 0.5, "None", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.barh(
                series.index, series.to_numpy() * 100, color=color, edgecolor="white", linewidth=0.5
            )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Contribution to total return (percentage points)")
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        f"Performance Attribution (total portfolio return {total:.1%})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    save_fig(fig, "10_performance_attribution.png", a.note)


def plot_overlap(a: Analysis, peer_csv: Path = PEER_HOLDINGS_CSV) -> None:
    if not peer_csv.exists():
        logger.info(
            "%s not found - skipping holdings overlap chart. Create it with "
            "columns [etf, ticker] to enable it.",
            peer_csv.name,
        )
        return

    peers = pd.read_csv(peer_csv).groupby("etf")["ticker"].apply(list).to_dict()
    overlap = holdings_overlap(list(a.weights.index), peers)
    series = pd.Series(overlap).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        series.index,
        series.to_numpy() * 100,
        color=[color_for(e) for e in series.index],
        edgecolor="white",
        linewidth=0.6,
    )
    ax.set_title(f"Holdings Overlap: {PORTFOLIO_NAME} vs Peer ETFs", fontsize=13, fontweight="bold")
    ax.set_xlabel("Share of portfolio holdings also held by the peer ETF")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(True, axis="x", alpha=0.3)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)

    not_in_top = (1 - max(overlap.values())) * 100
    ax.annotate(
        f"{not_in_top:.0f}% of holdings are not in the\npeer ETF with the highest overlap",
        xy=(0.97, 0.05),
        xycoords="axes fraction",
        ha="right",
        fontsize=9,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#EEF4FF", edgecolor="#AAAAAA"),
    )
    save_fig(
        fig, "11_holdings_overlap.png", "Peer lists come from peer_holdings.csv (partial snapshot)"
    )


# =============================================================================
# Outputs
# =============================================================================


def write_run_info(a: Analysis) -> None:
    """Record the effective backtest window so the dashboard and README can show it."""
    info = {
        "requested_start": START_DATE,
        "effective_start": f"{a.levels.index[0]:%Y-%m-%d}",
        "end": f"{a.levels.index[-1]:%Y-%m-%d}",
        "trading_days": int(len(a.levels)),
        "holdings_with_data": int(len(a.weights)),
        "weight_coverage_at_start": round(a.portfolio.coverage_at_start, 4),
        "min_weight_coverage": MIN_WEIGHT_COVERAGE,
        "rebalance_freq": REBALANCE_FREQ,
        "risk_free_annual": RISK_FREE_ANNUAL,
        "first_price_date": {
            t: (d.strftime("%Y-%m-%d") if pd.notna(d) else None)
            for t, d in a.portfolio.first_prices.sort_values().items()
        },
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    path = OUTPUT_DIR / "run_info.json"
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved %s", path)


def export_metrics(a: Analysis) -> pd.DataFrame:
    metrics = build_metrics_table(a.levels, a.returns, PORTFOLIO_NAME, MARKET_BENCHMARK)
    disp = format_metrics(metrics)
    print("\n=== Risk-Adjusted Metrics Summary ===")
    print(disp.to_string())
    metrics.to_csv(OUTPUT_DIR / "metrics_summary.csv")
    disp.to_csv(OUTPUT_DIR / "snapshot.csv")
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUTPUT_DIR.mkdir(exist_ok=True)

    inputs = load_inputs()
    prices = download_prices(list(inputs.weights.index), BENCHMARKS)
    a = prepare_analysis(prices, inputs)

    late = a.portfolio.first_prices[a.portfolio.first_prices > a.portfolio.start]
    logger.info(
        "Backtest window: %s to %s (%d holdings with data, %.0f%% of target weight at start)",
        a.levels.index[0].date(),
        a.levels.index[-1].date(),
        len(a.weights),
        a.portfolio.coverage_at_start * 100,
    )
    if len(late):
        logger.info(
            "Holdings added after the start date: %s", {t: str(d.date()) for t, d in late.items()}
        )

    plot_indexed_performance(a)
    plot_relative_strength(a)
    plot_volatility_bar(a)
    plot_drawdown(a)
    if MARKET_BENCHMARK in a.returns.columns:
        plot_rolling_beta(a, MARKET_BENCHMARK, "05_rolling_beta_vs_spy.png", show_kpis=True)
        plot_rolling_corr(a, MARKET_BENCHMARK, "06_rolling_correlation_vs_spy.png")
    if "LOCK.L" in a.returns.columns:
        plot_rolling_beta(a, "LOCK.L", "07_rolling_beta_vs_lock.png")
        plot_rolling_corr(a, "LOCK.L", "08_rolling_correlation_vs_lock.png")
    plot_concentration(a)
    plot_sector_donut(a)
    plot_attribution(a)
    plot_overlap(a)
    metrics = export_metrics(a)
    plot_metrics_table(a, metrics)
    write_run_info(a)

    logger.info("\nAll charts and data saved to: %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
