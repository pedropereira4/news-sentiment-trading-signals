"""Event study on synthetic prices where the right answer is known in advance."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentiment_pipeline.analysis.event_study import (
    compute_event_returns,
    is_strong,
    mean_ci,
    primary_regression,
    summarize,
)
from sentiment_pipeline.analysis.prices import PriceIndex
from sentiment_pipeline.analysis.protocol import Horizon, StrongSignal, load_protocol

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [Horizon.parse(h) for h in ("5min", "1h", "1d", "5d")]

# New York is UTC-4 in late September: the regular session is 13:30-20:00 UTC.
FRI, MON = datetime(2026, 9, 25, tzinfo=UTC), datetime(2026, 9, 28, tzinfo=UTC)


def session(day: datetime, every_min: int = 1) -> pd.DatetimeIndex:
    open_, close = day.replace(hour=13, minute=30), day.replace(hour=20, minute=0)
    return pd.date_range(open_, close, freq=f"{every_min}min", inclusive="left")


def bars(ticker: str, times, price_at) -> pd.DataFrame:
    return pd.DataFrame(
        {"ticker": ticker, "ts": times, "close": [float(price_at(t)) for t in times]}
    )


def days(*d: datetime, every_min: int = 1) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.concatenate([session(x, every_min) for x in d]))


FRI_MON_TUE = days(FRI, MON, MON + timedelta(days=1))
STEP = MON.replace(hour=15)  # everything moves at 15:00 UTC on Monday


def spy_price(t):
    return 101.0 if t >= STEP else 100.0  # market +1%


def aaa_price(t):
    return 52.0 if t >= STEP else 50.0  # stock +4% -> abnormal +3%


@pytest.fixture
def index() -> PriceIndex:
    frames = [
        bars("SPY", FRI_MON_TUE, spy_price),
        bars("AAA", FRI_MON_TUE, aaa_price),
        # Illiquid: one trade every 10 minutes.
        bars("BBB", days(FRI, MON, MON + timedelta(days=1), every_min=10), lambda t: 20.0),
    ]
    return PriceIndex(pd.concat(frames, ignore_index=True))


def signal(ticker: str, t: datetime, article_id: str = "a1") -> pd.DataFrame:
    return pd.DataFrame([{"article_id": article_id, "ticker": ticker, "published_at": t}])


# ---------------------------------------------------------------- point-in-time prices
def test_price_at_t_only_uses_bars_that_ended_before_t():
    t_news = MON.replace(hour=14, minute=30, second=20)
    b = pd.DataFrame(
        {
            "ticker": ["SPY", "SPY", "X", "X"],
            "ts": [t_news - timedelta(minutes=1, seconds=20), t_news - timedelta(seconds=20)] * 2,
            "close": [100.0, 100.0, 10.0, 99.0],  # the 14:30 bar traded AFTER the news
        }
    )
    idx = PriceIndex(b)
    assert idx.asof("X", t_news, timedelta(days=1)).price == 10.0
    assert idx.asof("X", t_news.replace(minute=31, second=0), timedelta(days=1)).price == 99.0


def test_stale_or_missing_prices_are_not_used(index):
    t = MON.replace(hour=14)
    assert index.asof("NOPE", t, timedelta(days=4)) is None
    assert index.asof("AAA", t + timedelta(days=10), timedelta(days=4)) is None


def test_trading_days_skip_the_weekend(index):
    fri = FRI.replace(hour=15)  # 11:00 in New York
    assert index.add_trading_days(fri, 1) == MON.replace(hour=15)
    sat = fri + timedelta(days=1)
    assert index.add_trading_days(sat, 1) == MON.replace(hour=15)
    assert index.add_trading_days(fri, 5) is None  # beyond the known calendar


# ---------------------------------------------------------------- abnormal returns
def test_known_abnormal_returns_per_horizon(index):
    out = compute_event_returns(
        signal("AAA", MON.replace(hour=14, minute=30)), index, HORIZONS
    ).set_index("horizon")
    assert out.loc["5min", "abnormal_ret"] == pytest.approx(0.0)  # before the move
    assert out.loc["1h", "ret"] == pytest.approx(0.04)
    assert out.loc["1h", "spy_ret"] == pytest.approx(0.01)
    assert out.loc["1h", "abnormal_ret"] == pytest.approx(0.03)  # market move removed
    assert out.loc["1d", "abnormal_ret"] == pytest.approx(0.03)
    assert out.loc["1d", "t1"] == MON.replace(hour=14, minute=30) + timedelta(days=1)
    assert out.loc["5d", "status"] == "pending"  # never filled with the last known price
    assert pd.isna(out.loc["5d", "abnormal_ret"])


def test_weekend_news_is_measured_from_the_last_trade_before_it(index):
    out = compute_event_returns(
        signal("AAA", FRI.replace(hour=23)),
        index,
        HORIZONS,  # Friday evening, after close
    ).set_index("horizon")
    assert out.loc["1d", "p0"] == 50.0  # Friday's last trade
    assert out.loc["1d", "t1"] == MON.replace(hour=23)
    assert out.loc["1d", "abnormal_ret"] == pytest.approx(0.03)


def test_no_trade_in_window_is_flagged_not_hidden(index):
    out = compute_event_returns(
        signal("BBB", MON.replace(hour=14, minute=31)), index, HORIZONS
    ).set_index("horizon")
    assert out.loc["5min", "ret"] == 0.0
    assert not out.loc["5min", "traded_in_window"]  # "no move" here means "no trade"
    assert out.loc["1h", "traded_in_window"]


def test_ticker_without_prices_is_no_price(index):
    out = compute_event_returns(signal("ZZZ", MON.replace(hour=14)), index, HORIZONS)
    assert set(out[out["horizon"] != "5d"]["status"]) == {"no_price"}


# ---------------------------------------------------------------- statistics
def test_mean_ci_matches_a_hand_calculation():
    mean, low, high, n = mean_ci(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert (mean, n) == (2.5, 4)
    # t(0.975, 3) = 3.182, sd = 1.291 -> half width 2.054
    assert low == pytest.approx(2.5 - 2.054, abs=1e-3) and high == pytest.approx(4.554, abs=1e-3)
    assert mean_ci(pd.Series([], dtype=float))[3] == 0


def _synthetic_panel(n=400, seed=7):
    rng = np.random.default_rng(seed)
    score = rng.uniform(-1, 1, n)
    small = rng.integers(0, 2, n)
    # Effect: 10 bps per unit of score for large caps, 50 bps for small caps.
    ar_bps = 10 * score + 40 * score * small + rng.normal(0, 5, n)
    t0 = [
        datetime(2026, 9, 28, 14, tzinfo=UTC) + timedelta(days=int(d))
        for d in rng.integers(0, 25, n)
    ]
    return pd.DataFrame(
        {
            "status": "ok",
            "horizon": "1d",
            "abnormal_ret": ar_bps / 1e4,
            "score": score,
            "cap_group": np.where(small == 1, "small_mid_cap", "large_cap"),
            "t0": t0,
            "is_new_info": True,
            "event_type": "earnings",
            "traded_in_window": True,
        }
    )


def test_primary_regression_recovers_a_planted_group_difference():
    fit = primary_regression(_synthetic_panel(), "1d")
    assert fit.params["score"] == pytest.approx(10, abs=2)
    assert fit.params["score:small"] == pytest.approx(40, abs=3)
    assert fit.ci_low["score:small"] < 40 < fit.ci_high["score:small"]
    assert fit.ci_low["score:small"] > 0 and fit.pvalues["score:small"] < 0.001


def test_primary_regression_refuses_tiny_samples():
    assert primary_regression(_synthetic_panel(n=10), "1d") is None


def test_summary_uses_only_strong_signals():
    panel = _synthetic_panel()
    rule = StrongSignal(min_abs_score=0.6, exclude_event_types=["commentary"])
    table = summarize(panel, [Horizon.parse("1d")], rule)
    assert table["n"].sum() == int(is_strong(panel, rule).sum())
    row = table[(table["cap_group"] == "small_mid_cap") & (table["direction"] == "positive")]
    assert row["mean_bps"].iloc[0] > 30  # 0.6..1.0 x 50 bps


# ---------------------------------------------------------------- protocol
def test_protocol_loads_and_parses_horizons():
    p = load_protocol(ROOT / "config" / "study.yaml")
    assert [h.label for h in p.analysis.horizons] == ["5min", "1h", "1d", "5d"]
    assert p.primary.trading_days == 1
    assert Horizon.parse("1h").offset == timedelta(hours=1)
    with pytest.raises(ValueError):
        Horizon.parse("1w")


def test_inference_never_uses_an_se_smaller_than_plain_ols():
    """With few trading days the clustered SE can collapse; the conservative rule must not."""
    import statsmodels.formula.api as smf

    panel = _synthetic_panel()
    panel["t0"] = [
        datetime(2026, 9, 28, 14, tzinfo=UTC) + timedelta(days=i % 4) for i in range(len(panel))
    ]
    fit = primary_regression(panel, "1d")
    assert fit.n_clusters == 4 and fit.df == 3
    d = panel.assign(
        ar_bps=panel["abnormal_ret"] * 1e4,
        small=(panel["cap_group"] == "small_mid_cap").astype(int),
    )
    classic = smf.ols("ar_bps ~ score * small", data=d).fit()
    assert (fit.se >= classic.bse - 1e-9).all()
    width = fit.ci_high["score:small"] - fit.ci_low["score:small"]
    assert width / (2 * fit.se["score:small"]) == pytest.approx(3.182, abs=1e-3)  # t(0.975, 3)


def test_stopping_rule_counts_only_the_collection_period():
    from sentiment_pipeline.analysis.cli import collection_counts

    p = load_protocol(ROOT / "config" / "study.yaml")  # collection starts 2026-09-28
    signals = pd.DataFrame(
        {
            "published_at": pd.to_datetime(
                # 02:00 UTC on the 28th is still the 27th in New York: not counted.
                ["2026-09-25 15:00", "2026-09-28 02:00", "2026-09-28 13:00", "2026-09-29 15:00"],
                utc=True,
            ),
            "cap_group": ["small_mid_cap", "small_mid_cap", "small_mid_cap", "large_cap"],
        }
    )
    assert collection_counts(signals, p) == {"large_cap": 1, "small_mid_cap": 1}
    assert collection_counts(signals.iloc[0:0], p) == {"large_cap": 0, "small_mid_cap": 0}
