"""The small-cap sample must be reproducible and match the committed watchlist."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "build_watchlist", ROOT / "scripts" / "build_watchlist.py"
)
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


def test_pool_excludes_other_sectors_and_duplicate_share_classes():
    pool = builder.load_pool()
    tickers = {t for entries in pool.values() for t, _ in entries}
    assert "UAA" not in tickers and "CWEN.A" not in tickers  # second share classes
    assert "AAT" not in tickers  # Real Estate is not in the study
    assert set(pool) == set(builder.SECTORS)


def test_sample_is_stratified_and_reproducible():
    pool = builder.load_pool()
    first = builder.sample_small_caps(pool)
    assert first == builder.sample_small_caps(pool)  # same seed, same draw
    assert Counter(s for _, _, s in first) == {s: builder.PER_SECTOR for s in builder.SECTORS}
    assert len({t for t, _, _ in first}) == len(first)
    assert builder.sample_small_caps(pool, seed=1) != first


def test_committed_watchlist_is_the_generated_one():
    """Editing watchlist.yaml by hand would silently break reproducibility."""
    expected = builder.render(builder.sample_small_caps(builder.load_pool()))
    assert (ROOT / "config" / "watchlist.yaml").read_text(encoding="utf-8") == expected


def test_short_names_drop_legal_suffixes():
    assert builder.short_name("Axcelis Technologies Inc.") == "Axcelis Technologies"
    assert builder.short_name("OneSpaWorld Holdings Limited") == "OneSpaWorld"
    assert builder.short_name("Etsy") == "Etsy"
