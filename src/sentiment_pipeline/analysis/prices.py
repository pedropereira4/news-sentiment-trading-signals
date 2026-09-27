"""Point-in-time price lookups on 1-minute bars.

Two rules keep the event study free of look-ahead:

1. A bar stamped `ts` holds trades in [ts, ts + 1 min), so its close is only known at
   ts + 1 min. "The price at time t" is the close of the last bar that *ended* at or
   before t - never the bar that contains t, which may include trades after the news.
2. Minutes without trades have no bar (common for small caps), so the price at t is the
   last trade at or before t. Whether any trade happened inside a window is reported
   separately (`trades_between`), because "no move" and "no trade" are different things.

Trading days come from the benchmark's own regular-session bars, so the calendar needs no
holiday list and always matches the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
BAR = pd.Timedelta(minutes=1)
REGULAR_OPEN, REGULAR_CLOSE = time(9, 30), time(16, 0)


@dataclass(frozen=True)
class Quote:
    price: float
    at: datetime  # when that price became known (end of its bar)


class PriceIndex:
    def __init__(self, bars: pd.DataFrame, benchmark: str = "SPY") -> None:
        """`bars`: columns ticker, ts (tz-aware UTC, bar start), close."""
        self.benchmark = benchmark
        self._times: dict[str, np.ndarray] = {}
        self._closes: dict[str, np.ndarray] = {}
        for ticker, g in bars.groupby("ticker", sort=False):
            g = g.sort_values("ts")
            ends = pd.to_datetime(g["ts"], utc=True) + BAR
            self._times[ticker] = ends.to_numpy(dtype="datetime64[ns]")
            self._closes[ticker] = g["close"].to_numpy(dtype=float)
        if benchmark not in self._times:
            raise ValueError(f"no bars for the benchmark {benchmark}")
        self.data_end = _to_dt(self._times[benchmark][-1])
        self.trading_days = self._trading_days(bars[bars["ticker"] == benchmark]["ts"])

    @staticmethod
    def _trading_days(ts: pd.Series) -> list[date]:
        local = pd.to_datetime(ts, utc=True).dt.tz_convert(ET)
        clock = local.dt.time
        in_session = (clock >= REGULAR_OPEN) & (clock < REGULAR_CLOSE)
        return sorted(set(local[in_session].dt.date))

    def asof(self, ticker: str, t: datetime, max_staleness: timedelta) -> Quote | None:
        times = self._times.get(ticker)
        if times is None:
            return None
        i = np.searchsorted(times, _to_np(t), side="right") - 1
        if i < 0:
            return None
        at = _to_dt(times[i])
        if t - at > max_staleness:
            return None
        return Quote(float(self._closes[ticker][i]), at)

    def trades_between(self, ticker: str, start: datetime, end: datetime) -> int:
        """Number of bars that ended in (start, end]."""
        times = self._times.get(ticker)
        if times is None:
            return 0
        lo = np.searchsorted(times, _to_np(start), side="right")
        hi = np.searchsorted(times, _to_np(end), side="right")
        return int(hi - lo)

    def add_trading_days(self, t: datetime, n: int) -> datetime | None:
        """Same New York clock time on the n-th trading day after t's date (None if unknown)."""
        local = t.astimezone(ET)
        later = [d for d in self.trading_days if d > local.date()]
        if len(later) < n:
            return None
        target = datetime.combine(later[n - 1], local.timetz().replace(tzinfo=None), tzinfo=ET)
        return target.astimezone(t.tzinfo)


def _to_np(t: datetime) -> np.datetime64:
    return np.datetime64(pd.Timestamp(t).tz_convert("UTC").tz_localize(None), "ns")


def _to_dt(x: np.datetime64) -> datetime:
    return pd.Timestamp(x).tz_localize("UTC").to_pydatetime()
