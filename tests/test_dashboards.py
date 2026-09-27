"""Grafana dashboards are generated from code; the committed JSON must match the generator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_builder(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_study_dashboard_json_is_up_to_date():
    builder = load_builder("build_study_dashboard")
    from sentiment_pipeline.analysis.protocol import load_protocol

    expected = builder.build(load_protocol(ROOT / "config" / "study.yaml"))
    committed = json.loads(builder.OUT.read_text(encoding="utf-8"))
    assert committed == expected, "run: python scripts/build_study_dashboard.py"


def test_study_dashboard_shows_no_returns():
    """Looking at returns during collection is what the stopping rule protects against."""
    text = (ROOT / "grafana" / "dashboards" / "event-study-collection.json").read_text()
    assert "event_returns" not in text and "abnormal" not in text


def test_every_grafana_panel_uses_a_provisioned_datasource():
    uids = set()
    for f in (ROOT / "grafana" / "provisioning" / "datasources").glob("*.yml"):
        uids |= {line.split(":", 1)[1].strip() for line in f.read_text().splitlines()
                 if line.strip().startswith("uid:")}  # fmt: skip
    for f in (ROOT / "grafana" / "dashboards").glob("*.json"):
        for p in json.loads(f.read_text(encoding="utf-8"))["panels"]:
            assert p["datasource"]["uid"] in uids, (f.name, p["title"])


def test_news_dashboard_json_is_up_to_date():
    builder = load_builder("build_dashboard")
    committed = ROOT / "grafana" / "dashboards" / "news-sentiment.json"
    assert json.loads(committed.read_text(encoding="utf-8")) == builder.build(), (
        "run: python scripts/build_dashboard.py"
    )
