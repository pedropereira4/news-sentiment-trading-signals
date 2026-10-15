"""Markdown report + confusion-matrix chart for the LLM evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sentiment_pipeline.analysis.report import INK, INK_2, MUTED, SURFACE
from sentiment_pipeline.evaluation.metrics import CLASSES, Score

# One-hue sequential ramp (reference palette, blue 100 -> 700): share of the human row.
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def plot_confusions(scores: dict[str, Score], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("ramp", RAMP)
    fig, axes = plt.subplots(
        1, len(scores), figsize=(4.4 * len(scores), 4.2), facecolor=SURFACE, squeeze=False
    )
    for ax, (name, s) in zip(axes[0], scores.items(), strict=True):
        m = s.matrix
        share = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        ax.pcolormesh(share[::-1], cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                r = len(CLASSES) - 1 - i
                dark = share[i, j] > 0.45
                ax.text(
                    j + 0.5,
                    r + 0.55,
                    f"{m[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if dark else INK,
                    fontweight="bold",
                )
                ax.text(j + 0.5, r + 0.3, f"{share[i, j]:.0%}", ha="center", va="center",
                        fontsize=8.5, color="white" if dark else INK_2)  # fmt: skip
        ticks = np.arange(len(CLASSES)) + 0.5
        ax.set_xticks(ticks, CLASSES, color=INK_2)
        ax.set_yticks(ticks, CLASSES[::-1], color=INK_2)
        ax.tick_params(length=0)
        ax.set_xlabel(f"{name} label", color=INK_2)
        ax.set_ylabel("Human label", color=INK_2)
        ax.set_title(f"{name}\naccuracy {s.accuracy:.0%} · κ {s.kappa:.2f} · n = {s.n}",
                     loc="left", color=INK, fontsize=10.5)  # fmt: skip
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_aspect("equal")
    fig.text(0.01, 0.01, "Cells: count and share of the human row (each row sums to 100%).",
             color=MUTED, fontsize=8)  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _ci(s: Score, key: str, fmt: str) -> str:
    lo, hi = s.ci[key]
    value = getattr(s, key)
    if np.isnan(value):
        return "– (undefined: one class only)"
    if np.isnan(lo):
        return f"{value:{fmt}}"
    return f"{value:{fmt}} ({lo:{fmt}} to {hi:{fmt}})"


def score_row(name: str, s: Score, latency: str = "–", cost: str = "–") -> str:
    return (
        f"| {name} | {s.n} | {_ci(s, 'accuracy', '.0%')} | {_ci(s, 'kappa', '.2f')} | "
        f"{_ci(s, 'macro_f1', '.2f')} | {latency} | {cost} |"
    )


def render_markdown(
    *,
    n_sample: int,
    n_labelled: int,
    n_skipped: int,
    seed: int,
    overall: list[str],
    by_group: list[str],
    attribution: list[str],
    chart_file: str,
    finbert_included: bool,
) -> str:
    head = "| Model | n | Accuracy (95% CI) | Cohen's κ (95% CI) | Macro-F1 (95% CI) | Time per item | Cost per item |"  # noqa: E501
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    lines = [
        "# LLM label evaluation",
        "",
        f"Blind human labels for a random sample of the event study's signals: {n_sample} "
        f"(article, company) pairs, half large caps and half small caps, seed {seed}. "
        f"{n_labelled} labelled ({n_skipped} not labelled or skipped).",
        "",
        "## Sentiment for the company",
        "",
        "Items the human marked as *not about the company* are left out here (see below).",
        "",
        head,
        sep,
        *overall,
        "",
        "### By group",
        "",
        head,
        sep,
        *by_group,
        "",
        f"![Confusion matrices]({chart_file})",
        "",
        "## Does the LLM attach news to the right company?",
        "",
        "| Group | LLM signals checked | Not about the company (95% CI) |",
        "| --- | --- | --- |",
        *attribution,
        "",
        "## Method",
        "",
        "- One annotator labelled each (article, company) pair from the headline and summary "
        "only, as an investor at publication time, without seeing the LLM's label or any price.",
        "- Gemini's label is the per-company `sentiment` the pipeline stored for that pair.",
    ]
    if finbert_included:
        lines.append(
            "- FinBERT (ProsusAI/finbert) labels the same text (headline + start of the "
            "summary). It has no notion of *which* company, so an article gets one label for "
            "every company it mentions."
        )
    lines += [
        "- Majority baseline: always predict the most common human label.",
        "- Cohen's κ corrects agreement for chance (0 = chance, 1 = perfect). Intervals: "
        "2,000 bootstrap resamples; Wilson interval for proportions.",
        "- Limits: a single annotator (no inter-annotator agreement), and labels judge the "
        "news, not the price reaction.",
    ]
    return "\n".join(lines) + "\n"
