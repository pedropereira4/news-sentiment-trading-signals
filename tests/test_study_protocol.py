"""The study protocol is data the analysis will read: keep it valid and consistent."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from sentiment_pipeline.watchlist import load_watchlist

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = yaml.safe_load((ROOT / "config" / "study.yaml").read_text(encoding="utf-8"))


def test_protocol_dates_are_consistent():
    data, stop = PROTOCOL["data"], PROTOCOL["stopping_rule"]
    assert PROTOCOL["registered_on"] <= data["collection_start"] < stop["max_end_date"]
    assert isinstance(stop["max_end_date"], date)
    assert stop["min_small_cap_signals"] > 0


def test_protocol_matches_the_pipeline_configuration():
    wl = load_watchlist(str(ROOT / PROTOCOL["data"]["watchlist"]))
    assert PROTOCOL["data"]["benchmark"] == wl.benchmark
    assert PROTOCOL["analysis"]["primary_horizon"] in PROTOCOL["analysis"]["horizons"]
    assert isinstance(PROTOCOL["amendments"], list)
