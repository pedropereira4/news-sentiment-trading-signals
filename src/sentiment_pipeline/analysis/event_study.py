"""Abnormal returns around each news signal, plus the registered statistics.

Market-adjusted abnormal return over a window [t0, t1]:

    AR = (P_stock(t1) / P_stock(t0) - 1) - (P_spy(t1) / P_spy(t0) - 1)

with every price taken point-in-time (see prices.py). A window whose end lies beyond the
data is `pending`, never filled with the last known price.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from sentiment_pipeline.analysis.prices import ET, PriceIndex
from sentiment_pipeline.analysis.protocol import Horizon, StrongSignal

# How old the last trade may be and still count as "the price at t". Four days covers a
# weekend plus a holiday; older than that means the ticker has no data around the event.
MAX_STALENESS = timedelta(days=4)

RESULT_COLUMNS = [
    "article_id", "ticker", "horizon", "status", "t0", "t1",
    "p0", "p1", "spy_p0", "spy_p1", "ret", "spy_ret", "abnormal_ret", "traded_in_window",
]  # fmt: skip


def window_end(index: PriceIndex, t0, horizon: Horizon):
    if horizon.trading_days:
        return index.add_trading_days(t0, horizon.trading_days)
    return t0 + horizon.offset


def compute_event_returns(
    signals: pd.DataFrame, index: PriceIndex, horizons: list[Horizon]
) -> pd.DataFrame:
    """One row per (signal, horizon). `signals` needs article_id, ticker, published_at."""
    rows = []
    bench = index.benchmark
    for sig in signals.itertuples(index=False):
        t0 = pd.Timestamp(sig.published_at).to_pydatetime()
        start = index.asof(sig.ticker, t0, MAX_STALENESS)
        spy_start = index.asof(bench, t0, MAX_STALENESS)
        for h in horizons:
            row = dict.fromkeys(RESULT_COLUMNS)
            row.update(article_id=sig.article_id, ticker=sig.ticker, horizon=h.label, t0=t0)
            t1 = window_end(index, t0, h)
            row["t1"] = t1
            if t1 is None or t1 > index.data_end:
                row["status"] = "pending"
            elif start is None or spy_start is None:
                row["status"] = "no_price"
            else:
                end = index.asof(sig.ticker, t1, MAX_STALENESS)
                spy_end = index.asof(bench, t1, MAX_STALENESS)
                if end is None or spy_end is None:
                    row["status"] = "no_price"
                else:
                    ret = end.price / start.price - 1
                    spy_ret = spy_end.price / spy_start.price - 1
                    row.update(
                        status="ok",
                        p0=start.price,
                        p1=end.price,
                        spy_p0=spy_start.price,
                        spy_p1=spy_end.price,
                        ret=ret,
                        spy_ret=spy_ret,
                        abnormal_ret=ret - spy_ret,
                        traded_in_window=index.trades_between(sig.ticker, t0, t1) > 0,
                    )
            rows.append(row)
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def is_strong(df: pd.DataFrame, rule: StrongSignal) -> pd.Series:
    mask = df["score"].abs() >= rule.min_abs_score
    if rule.is_new_info:
        mask &= df["is_new_info"].astype(bool)
    if rule.exclude_event_types:
        mask &= ~df["event_type"].isin(rule.exclude_event_types)
    return mask


def mean_ci(values: pd.Series, level: float = 0.95) -> tuple[float, float, float, int]:
    """Mean and t-based confidence interval. Returns (mean, low, high, n)."""
    from scipy import stats

    v = values.dropna().to_numpy(dtype=float)
    n = len(v)
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    mean = float(v.mean())
    if n < 2:
        return (mean, np.nan, np.nan, n)
    half = stats.t.ppf(0.5 + level / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return (mean, mean - half, mean + half, n)


def summarize(
    merged: pd.DataFrame, horizons: list[Horizon], strong_rule: StrongSignal
) -> pd.DataFrame:
    """Mean abnormal return (bps, 95% CI) per horizon x group x direction of strong signals."""
    ok = merged[merged["status"] == "ok"].copy()
    ok["strong"] = is_strong(ok, strong_rule)
    ok = ok[ok["strong"]]
    ok["direction"] = np.where(ok["score"] > 0, "positive", "negative")
    out = []
    for h in horizons:
        for group in ("large_cap", "small_mid_cap"):
            for direction in ("positive", "negative"):
                sel = ok[
                    (ok["horizon"] == h.label)
                    & (ok["cap_group"] == group)
                    & (ok["direction"] == direction)
                ]
                mean, low, high, n = mean_ci(sel["abnormal_ret"] * 1e4)
                traded = float(sel["traded_in_window"].mean()) if n else np.nan
                out.append(
                    dict(horizon=h.label, cap_group=group, direction=direction, n=n,
                         mean_bps=mean, ci_low_bps=low, ci_high_bps=high, share_traded=traded)
                )  # fmt: skip
    return pd.DataFrame(out)


@dataclass
class RegressionResult:
    """OLS estimates with conservative inference (see primary_regression)."""

    params: pd.Series
    se: pd.Series  # the largest of the candidate standard errors, per coefficient
    se_method: pd.Series  # which estimator was the largest
    ci_low: pd.Series
    ci_high: pd.Series
    pvalues: pd.Series
    nobs: int
    n_clusters: int
    df: int


def primary_regression(merged: pd.DataFrame, horizon: str, min_obs: int = 20):
    """AR(bps) ~ score * small_cap. Returns None when there is too little data.

    The coefficient on score:small is the registered test: does a unit of sentiment move
    small caps by more (or less) than large caps?

    Inference is deliberately conservative. Day-clustered standard errors are the textbook
    choice (news on the same day shares market conditions), but with few clusters - a month
    is ~20 trading days - they can badly understate uncertainty: on simulated data with 7
    days they came out 7x smaller than plain OLS. So each coefficient uses the LARGEST of the
    classic, HC1 and day-clustered standard errors, with a t distribution on
    min(G - 1, n - k) degrees of freedom. Missing a real effect is preferred to reporting a
    fake one.
    """
    import statsmodels.formula.api as smf
    from scipy import stats

    d = merged[(merged["status"] == "ok") & (merged["horizon"] == horizon)].copy()
    d = d.dropna(subset=["abnormal_ret", "score", "cap_group"])
    if len(d) < min_obs:
        return None
    d["ar_bps"] = d["abnormal_ret"] * 1e4
    d["small"] = (d["cap_group"] == "small_mid_cap").astype(int)
    d["day"] = pd.to_datetime(d["t0"], utc=True).dt.tz_convert(ET).dt.date.astype(str)
    n_clusters = d["day"].nunique()
    if d["small"].nunique() < 2 or n_clusters < 2:
        return None

    model = smf.ols("ar_bps ~ score * small", data=d)
    fits = {
        "classic": model.fit(),
        "HC1": model.fit(cov_type="HC1"),
        "clustered": model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["day"])[0]}),
    }
    ses = pd.DataFrame({name: f.bse for name, f in fits.items()})
    se, method = ses.max(axis=1), ses.idxmax(axis=1)
    params = fits["classic"].params
    df = int(min(n_clusters - 1, fits["classic"].df_resid))
    t_crit = stats.t.ppf(0.975, df)
    pvalues = pd.Series(2 * stats.t.sf(np.abs(params / se), df), index=params.index)
    return RegressionResult(
        params=params,
        se=se,
        se_method=method,
        ci_low=params - t_crit * se,
        ci_high=params + t_crit * se,
        pvalues=pvalues,
        nobs=int(fits["classic"].nobs),
        n_clusters=n_clusters,
        df=df,
    )
