"""sp-event-study: run the registered event study, or check progress without looking.

sp-event-study --status     # how far the stopping rule is (no returns are computed)
sp-event-study              # the registered analysis (data from collection_start)
sp-event-study --pilot      # same code on ALL data, labelled as a pilot
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from sentiment_pipeline.analysis.data import load_bars, load_signals, write_event_returns
from sentiment_pipeline.analysis.event_study import (
    compute_event_returns,
    primary_regression,
    summarize,
)
from sentiment_pipeline.analysis.prices import ET, PriceIndex
from sentiment_pipeline.analysis.protocol import Protocol, load_protocol
from sentiment_pipeline.analysis.report import plot_summary, render_markdown
from sentiment_pipeline.common import setup_logging
from sentiment_pipeline.config import PostgresSettings
from sentiment_pipeline.storage.postgres import ensure_schema
from sentiment_pipeline.writer.postgres_writer import connect

log = logging.getLogger("event-study")


def stopping_status(protocol: Protocol, n_small: int, today: date) -> tuple[bool, str]:
    rule = protocol.stopping_rule
    reached = n_small >= rule.min_small_cap_signals or today > rule.max_end_date
    days_left = (rule.max_end_date - today).days
    text = (
        f"{n_small}/{rule.min_small_cap_signals} small-cap signals, "
        f"{max(days_left, 0)} days to {rule.max_end_date}"
        + (" - REACHED" if reached else " - collecting")
    )
    return reached, text


def collection_counts(signals: pd.DataFrame, protocol: Protocol) -> dict[str, int]:
    """Signals per group inside the registered collection period.

    The stopping rule counts only these, even in a pilot run that also looks at older data.
    """
    out = {"large_cap": 0, "small_mid_cap": 0}
    if signals.empty:
        return out
    # Same boundary as load_signals: midnight in New York on the first collection day.
    start = pd.Timestamp(datetime.combine(protocol.data.collection_start, time(0), tzinfo=ET))
    counted = signals.loc[signals["published_at"] >= start, "cap_group"].value_counts()
    return {g: int(counted.get(g, 0)) for g in out}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--status", action="store_true", help="progress only, no results")
    parser.add_argument("--pilot", action="store_true", help="use all data, labelled as pilot")
    parser.add_argument("--study", default="config/study.yaml")
    parser.add_argument("--out", default="reports")
    parser.add_argument(
        "--as-of", type=date.fromisoformat, help="run as if today were this date (YYYY-MM-DD)"
    )
    args = parser.parse_args(argv)

    settings = PostgresSettings()
    setup_logging(settings.log_level)
    protocol = load_protocol(args.study)
    today = args.as_of or datetime.now(UTC).date()
    until = min(today, protocol.stopping_rule.max_end_date)
    since = None if args.pilot else protocol.data.collection_start

    conn = connect(settings.postgres_conninfo)
    ensure_schema(conn)
    signals, excluded = load_signals(conn, protocol, since, until)
    groups = signals["cap_group"].value_counts() if not signals.empty else pd.Series(dtype=int)
    counts = collection_counts(signals, protocol)
    reached, stopping = stopping_status(protocol, counts["small_mid_cap"], today)

    if args.status:
        print(f"Stopping rule: {stopping}")
        print(f"Large-cap signals: {counts['large_cap']}")
        if not reached:
            print("Keep collecting. (--status never computes returns, so it is safe to run.)")
        return
    if not reached and not args.pilot:
        log.warning("Stopping rule not reached yet (%s): results are interim.", stopping)
    if signals.empty:
        raise SystemExit("No eligible signals in the window yet.")

    t_min, t_max = signals["published_at"].min(), signals["published_at"].max()
    tickers = sorted(set(signals["ticker"]) | {protocol.data.benchmark})
    bars = load_bars(conn, tickers, t_min - timedelta(days=7), t_max + timedelta(days=12))
    index = PriceIndex(bars, benchmark=protocol.data.benchmark)
    horizons = protocol.analysis.horizons

    results = compute_event_returns(signals, index, horizons)
    label = "pilot" if args.pilot else "registered"
    write_event_returns(conn, results, label)
    merged = results.merge(signals, on=["article_id", "ticker"], how="left")

    summary = summarize(merged, horizons, protocol.analysis.strong_signal)
    regression = primary_regression(merged, protocol.analysis.primary_horizon)
    status_counts = (
        results.groupby(["horizon", "status"]).size().unstack(fill_value=0)
        .reindex([h.label for h in horizons]).fillna(0).astype(int)
    )  # fmt: skip

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"event_study_{today.isoformat()}" + ("_pilot" if args.pilot else "")
    chart = out / f"{stem}.png"
    plot_summary(summary, [h.label for h in horizons], protocol.analysis.primary_horizon, chart)
    report = out / f"{stem}.md"
    report.write_text(
        render_markdown(
            run_date=today,
            pilot=args.pilot,
            protocol=protocol,
            window=(t_min.date() if args.pilot else protocol.data.collection_start, until),
            n_signals={g: int(groups.get(g, 0)) for g in ("large_cap", "small_mid_cap")},
            excluded=excluded,
            status_counts=status_counts,
            summary=summary,
            regression=regression,
            chart_file=chart.name,
            stopping=stopping,
        ),
        encoding="utf-8",
    )
    print(f"Report: {report}")
    conn.close()


if __name__ == "__main__":
    main()
