"""sp-eval: evaluate the LLM's per-company labels against blind human labels.

sp-eval sample            # draw the random sample (once) -> eval/sample.csv
sp-eval label             # label it in the terminal; resumable, saves every answer
sp-eval score [--finbert] # metrics vs the human labels -> reports/llm_eval.md + .png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment_pipeline.analysis.protocol import load_protocol
from sentiment_pipeline.common import setup_logging
from sentiment_pipeline.config import PostgresSettings
from sentiment_pipeline.evaluation.labeling import read_labels, run_labelling
from sentiment_pipeline.evaluation.metrics import CLASSES, Score, score, wilson
from sentiment_pipeline.evaluation.report import plot_confusions, render_markdown, score_row
from sentiment_pipeline.evaluation.sample import build_blind_sample, model_text

SEED = 20261015
EVAL_DIR = Path("eval")
SAMPLE, LABELS, FINBERT_CACHE = (
    EVAL_DIR / "sample.csv",
    EVAL_DIR / "labels.csv",
    EVAL_DIR / "finbert.csv",
)
GROUP_NAMES = {"large_cap": "Large caps", "small_mid_cap": "Small caps"}

_LLM_SQL = """
SELECT s.article_id, s.ticker, s.sentiment AS llm_label, a.llm_latency_ms
FROM ticker_signals s JOIN articles a ON a.id = s.article_id
WHERE s.article_id = ANY(%s)
"""


def _connect():
    from sentiment_pipeline.writer.postgres_writer import connect

    return connect(PostgresSettings().postgres_conninfo)


def cmd_sample(args) -> None:
    if SAMPLE.exists() and not args.force:
        raise SystemExit(f"{SAMPLE} already exists; labels refer to it. Use --force to redraw.")
    protocol = load_protocol(args.study)
    conn = _connect()
    sample = build_blind_sample(conn, protocol, args.n_per_group, args.seed)
    conn.close()
    EVAL_DIR.mkdir(exist_ok=True)
    sample.to_csv(SAMPLE, index=False)
    counts = sample["cap_group"].map(GROUP_NAMES).value_counts().to_dict()
    print(f"Wrote {SAMPLE}: {len(sample)} items {counts}. Next: sp-eval label")


def load_sample() -> pd.DataFrame:
    if not SAMPLE.exists():
        raise SystemExit("No sample yet: run `sp-eval sample` first.")
    return pd.read_csv(SAMPLE, dtype={"article_id": str, "ticker": str})


def cmd_label(args) -> None:
    run_labelling(load_sample(), LABELS)


def assemble(sample: pd.DataFrame, labels: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    """One row per labelled item, with the human label and Gemini's label."""
    out = sample.merge(labels[["item", "label"]], on="item", how="inner")
    return out.merge(llm, on=["article_id", "ticker"], how="left")


def evaluate(
    data: pd.DataFrame,
    models: dict[str, str],
    n_boot: int = 2000,
) -> tuple[dict[str, Score], dict[tuple[str, str], Score]]:
    """Scores overall and per group for each model column, plus the majority baseline."""
    rated = data[data["label"].isin(CLASSES)]
    overall, by_group = {}, {}
    for name, col in models.items():
        overall[name] = score(rated["label"], rated[col], n_boot)
        for g in GROUP_NAMES:
            sub = rated[rated["cap_group"] == g]
            by_group[(name, g)] = score(sub["label"], sub[col], n_boot)
    majority = rated["label"].mode().iloc[0] if len(rated) else "neutral"
    overall[f"Always '{majority}' (baseline)"] = score(
        rated["label"], np.full(len(rated), majority), n_boot
    )
    return overall, by_group


def cmd_score(args) -> None:
    sample = load_sample()
    labels = read_labels(LABELS)
    if labels.empty:
        raise SystemExit("No labels yet: run `sp-eval label` first.")
    conn = _connect()
    with conn.cursor() as cur:
        cur.execute(_LLM_SQL, (list(sample["article_id"].unique()),))
        llm = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
    conn.close()
    data = assemble(sample, labels, llm)

    models = {"Gemini 3.1 Flash Lite": "llm_label"}
    if args.finbert:
        from sentiment_pipeline.evaluation.finbert import load_finbert, run_finbert

        texts = data.set_index("item").apply(lambda r: model_text(r["title"], r["summary"]), axis=1)
        fb = run_finbert(texts, load_finbert(), FINBERT_CACHE)
        data = data.merge(
            fb.rename(columns={"label": "finbert_label", "latency_ms": "finbert_ms"}),
            left_on="item", right_index=True, how="left",
        )  # fmt: skip
        models["FinBERT"] = "finbert_label"

    overall, by_group = evaluate(data, models)
    llm_ms = f"{data['llm_latency_ms'].median():.0f} ms"
    extras = {"Gemini 3.1 Flash Lite": (llm_ms, f"~${args.llm_cost:.4f}")}
    if args.finbert:
        extras["FinBERT"] = (f"{data['finbert_ms'].median():.0f} ms (CPU)", "$0 (local)")
    overall_rows = [score_row(n, s, *extras.get(n, ("–", "–"))) for n, s in overall.items()]
    group_rows = [score_row(f"{n}, {GROUP_NAMES[g]}", s) for (n, g), s in by_group.items()]

    attribution = []
    for g, name in [(None, "All"), *GROUP_NAMES.items()]:
        sub = data if g is None else data[data["cap_group"] == g]
        p, lo, hi = wilson(int((sub["label"] == "not_about").sum()), len(sub))
        attribution.append(f"| {name} | {len(sub)} | {p:.0%} ({lo:.0%} to {hi:.0%}) |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chart = out / "llm_eval.png"
    plot_confusions({n: overall[n] for n in models}, chart)
    report = out / "llm_eval.md"
    report.write_text(
        render_markdown(
            n_sample=len(sample),
            n_labelled=len(data),
            n_skipped=len(sample) - len(data),
            seed=SEED,
            overall=overall_rows,
            by_group=group_rows,
            attribution=attribution,
            chart_file=chart.name,
            finbert_included=args.finbert,
        ),
        encoding="utf-8",
    )
    print(f"Report: {report}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", default="config/study.yaml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sample", help="draw the random sample once")
    p.add_argument("--n-per-group", type=int, default=100)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--force", action="store_true", help="redraw (invalidates existing labels)")
    sub.add_parser("label", help="label the sample in the terminal (resumable)")
    p = sub.add_parser("score", help="compare models with the human labels")
    p.add_argument("--finbert", action="store_true", help="also run the FinBERT baseline")
    p.add_argument("--llm-cost", type=float, default=0.0002, help="USD per item (billing)")
    p.add_argument("--out", default="reports")
    args = parser.parse_args(argv)
    setup_logging(PostgresSettings().log_level)
    {"sample": cmd_sample, "label": cmd_label, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
