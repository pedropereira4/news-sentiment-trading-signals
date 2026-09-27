"""Markdown report + chart for an event-study run."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Reference palette (validated for colour-vision deficiencies and contrast on the surface).
SURFACE, INK, INK_2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
SERIES = {"positive": "#2a78d6", "negative": "#eb6834"}
GROUPS = {"large_cap": "Large caps", "small_mid_cap": "Small caps (S&P 600 sample)"}


MIN_N_PLOT = 5


def plot_summary(summary: pd.DataFrame, horizons: list[str], primary: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True, facecolor=SURFACE)
    x = np.arange(len(horizons))
    for ax, (group, title) in zip(axes, GROUPS.items(), strict=True):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=AXIS, linewidth=1, zorder=1)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        n_primary = 0
        for k, (direction, color) in enumerate(SERIES.items()):
            d = summary[(summary["cap_group"] == group) & (summary["direction"] == direction)]
            d = d.set_index("horizon").reindex(horizons)
            n_primary += int(d.loc[primary, "n"]) if pd.notna(d.loc[primary, "n"]) else 0
            xs = x + (k - 0.5) * 0.12  # nudge so the two sets of whiskers never overlap
            # A mean of one or two events has no usable CI and would set the y-scale on its own.
            enough = d["n"].fillna(0).to_numpy() >= MIN_N_PLOT
            mean = np.where(enough, d["mean_bps"].to_numpy(dtype=float), np.nan)
            err = np.vstack(
                [mean - d["ci_low_bps"].to_numpy(float), d["ci_high_bps"].to_numpy(float) - mean]
            )
            ax.plot(xs, mean, color=color, linewidth=2, zorder=3,
                    label=f"{direction.capitalize()} strong signals")  # fmt: skip
            ax.errorbar(xs, mean, yerr=np.nan_to_num(err), fmt="o", color=color, markersize=6,
                        elinewidth=1.5, capsize=0, zorder=4,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)  # fmt: skip
        ax.set_title(f"{title}\n{n_primary} strong signals at {primary}", color=INK,
                     fontsize=10.5, loc="left")  # fmt: skip
        ax.set_xticks(x, horizons, color=MUTED)
        ax.tick_params(axis="y", colors=MUTED, length=0)
        ax.tick_params(axis="x", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        if summary[summary["cap_group"] == group]["n"].sum() == 0:
            ax.text(0.5, 0.5, "no strong signals yet", transform=ax.transAxes, ha="center",
                    color=MUTED, fontsize=9)  # fmt: skip
    axes[0].set_ylabel("Mean abnormal return vs SPY (bps)", color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, ncols=2, fontsize=9,
               labelcolor=INK_2)  # fmt: skip
    note = (
        "Points: mean; whiskers: 95% CI. Horizon after publication. "
        f"Cells with fewer than {MIN_N_PLOT} events are only in the table."
    )
    fig.text(0.01, 0.01, note, color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _fmt(v: float, digits: int = 1) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def render_markdown(
    *,
    run_date: date,
    pilot: bool,
    protocol,
    window: tuple[date, date],
    n_signals: dict[str, int],
    excluded: dict[str, int],
    status_counts: pd.DataFrame,
    summary: pd.DataFrame,
    regression,
    chart_file: str,
    stopping: str,
) -> str:
    lines = [f"# Event study - {run_date.isoformat()}", ""]
    if pilot:
        lines += [
            "> **PILOT RUN - not the registered analysis.** It includes data from before "
            f"{protocol.data.collection_start} and exists only to check the code on real data.",
            "",
        ]
    lines += [
        f"**Hypothesis (registered {protocol.registered_on}):** {protocol.hypothesis.strip()}",
        "",
        f"**Window:** {window[0]} to {window[1]} · **Stopping rule:** {stopping}",
        "",
        "## Data",
        "",
        "| Group | Signals used |",
        "| --- | --- |",
        *[f"| {GROUPS.get(g, g)} | {n} |" for g, n in n_signals.items()],
        "",
        "Excluded before analysis: "
        + ", ".join(f"{n} {reason}" for reason, n in excluded.items())
        + ".",
        "",
        "| Horizon | ok | pending (window not over) | no price |",
        "| --- | --- | --- | --- |",
    ]
    for h, r in status_counts.iterrows():
        lines.append(f"| {h} | {r.get('ok', 0)} | {r.get('pending', 0)} | {r.get('no_price', 0)} |")
    lines += [
        "",
        "## Mean abnormal return of strong signals",
        "",
        f"![Mean abnormal return by horizon]({chart_file})",
        "",
        "| Horizon | Group | Direction | n | Mean (bps) | 95% CI (bps) | Windows with a trade |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary.itertuples(index=False):
        traded = "–" if np.isnan(r.share_traded) else f"{r.share_traded:.0%}"
        lines.append(
            f"| {r.horizon} | {GROUPS[r.cap_group]} | {r.direction} | {r.n} | {_fmt(r.mean_bps)} "
            f"| {_fmt(r.ci_low_bps)} to {_fmt(r.ci_high_bps)} | {traded} |"
        )
    lines += ["", f"## Primary test ({protocol.analysis.primary_horizon})", ""]
    if regression is None:
        lines.append(
            "Not enough data yet for the registered regression (it needs at least 20 events "
            "with prices, both groups, and more than one trading day)."
        )
    else:
        lines += [
            "OLS of the abnormal return (bps) on the signal score, with a score × small-cap "
            f"interaction. n = {regression.nobs} events over G = {regression.n_clusters} "
            "trading days. Each coefficient uses the largest of the classic, HC1 and "
            "day-clustered standard errors, with a t distribution on "
            f"{regression.df} degrees of freedom (conservative on purpose: clustered errors "
            "understate uncertainty when there are few trading days).",
            "",
            "| Term | Estimate (bps) | 95% CI | p-value | SE used |",
            "| --- | --- | --- | --- | --- |",
        ]
        names = {"Intercept": "Intercept", "score": "Score (large caps)", "small": "Small cap",
                 "score:small": "**Score × small cap**"}  # fmt: skip
        for term, label in names.items():
            lines.append(
                f"| {label} | {regression.params[term]:.1f} | {regression.ci_low[term]:.1f} to "
                f"{regression.ci_high[term]:.1f} | {regression.pvalues[term]:.3f} "
                f"| {regression.se_method[term]} |"
            )
        lines += [
            "",
            "The **score × small cap** row answers the hypothesis: how many extra basis points a "
            "one-unit change in sentiment moves a small cap, compared with a large cap.",
        ]
    lines += [
        "",
        "## Method",
        "",
        "- Price at time *t*: close of the last 1-minute bar that ended at or before *t* "
        "(a bar's close is only known when the bar ends).",
        "- Abnormal return: stock return minus SPY return over the same window.",
        "- 5min / 1h: wall-clock windows. 1d / 5d: same New York clock time N trading days "
        "later, with the calendar taken from SPY's regular sessions.",
        "- Windows that end after the last available price are `pending`, never filled in.",
        '- "Windows with a trade": share of events where the stock traded inside the window; '
        "for thinly traded stocks a zero return can mean no trade, not no reaction.",
    ]
    return "\n".join(lines) + "\n"
