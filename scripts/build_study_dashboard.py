"""Generate grafana/dashboards/event-study-collection.json from config/study.yaml.

A monitoring dashboard for the data collection of the event study (PostgreSQL). It shows
how much data has arrived and whether it is healthy, and deliberately NO returns: looking at
results while collecting is how a study ends up stopped at a lucky moment.

The protocol values (collection start, news source, LLM model, stopping rule...) are baked
into the SQL, so the dashboard counts exactly what the analysis will use.
Run:  python scripts/build_study_dashboard.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment_pipeline.analysis.protocol import Protocol, load_protocol  # noqa: E402

DS = {"type": "grafana-postgresql-datasource", "uid": "postgres"}
OUT = ROOT / "grafana" / "dashboards" / "event-study-collection.json"

# Categorical colours for the two groups (validated for CVD and contrast on a dark surface).
LARGE, SMALL = "#199e70", "#9085e9"
GROUP_LABEL = {"large_cap": "Large caps", "small_mid_cap": "Small caps"}


def _ny_midnight_utc(day: date) -> str:
    t = datetime.combine(day, time(0), tzinfo=ZoneInfo("America/New_York")).astimezone(UTC)
    return t.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _q(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class Sql:
    """SQL conditions that encode the registered protocol.

    Each method takes an optional table alias (e.g. "v.") for use inside joins.
    """

    def __init__(self, p: Protocol) -> None:
        self.start = f"TIMESTAMPTZ '{p.data.collection_start} 00:00 America/New_York'"
        self.end_date = f"DATE '{p.stopping_rule.max_end_date}'"
        self.target = p.stopping_rule.min_small_cap_signals
        self._p = p
        rule = p.analysis.strong_signal
        self.strong_text = (
            f"|score| >= {rule.min_abs_score}"
            + (", new information" if rule.is_new_info else "")
            + (f", not {'/'.join(rule.exclude_event_types)}" if rule.exclude_event_types else "")
        )

    def reasons(self, a: str = "") -> list[tuple[str, str]]:
        """Same filters, in the same order, as sentiment_pipeline.analysis.data.load_signals."""
        d = self._p.data
        return [
            ("not from the registered news source", f"{a}source <> {_q(d.news_source)}"),
            ("no source timestamp", f"NOT {a}feed_timestamp"),
            ("different LLM model", f"{a}llm_model <> {_q(d.llm_model)}"),
            ("ticker no longer in the watchlist", f"{a}cap_group IS NULL"),
        ]

    def eligible(self, a: str = "") -> str:
        return " AND ".join(f"NOT ({cond})" for _, cond in self.reasons(a))

    def collected(self, a: str = "") -> str:
        """Signals that count for the study: eligible and inside the collection period."""
        return f"{a}published_at >= {self.start} AND {self.eligible(a)}"

    def strong(self, a: str = "") -> str:
        rule = self._p.analysis.strong_signal
        cond = f"abs({a}score) >= {rule.min_abs_score}"
        if rule.is_new_info:
            cond += f" AND {a}is_new_info"
        if rule.exclude_event_types:
            types = ", ".join(_q(e) for e in rule.exclude_event_types)
            cond += f" AND {a}event_type NOT IN ({types})"
        return cond


# ---------------------------------------------------------------- panel helpers
def target(sql: str, fmt: str = "table") -> dict:
    return {
        "refId": "A",
        "datasource": DS,
        "editorMode": "code",
        "rawQuery": True,
        "format": fmt,
        "rawSql": sql.strip(),
    }


def panel(pid, title, ptype, sql, grid, *, fmt="table", description="", **extra) -> dict:
    x, y, w, h = grid
    p = {
        "id": pid,
        "type": ptype,
        "title": title,
        "description": description,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [target(sql, fmt)],
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {},
    }
    p.update(extra)
    return p


def fixed_color(name: str, color: str) -> dict:
    return {
        "matcher": {"id": "byName", "options": name},
        "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}],
    }


GROUP_COLORS = [fixed_color("Large caps", LARGE), fixed_color("Small caps", SMALL)]


def stat(pid, title, sql, x, *, unit=None, description="", color="text", **defaults) -> dict:
    return panel(
        pid,
        title,
        "stat",
        sql,
        (x, 0, 4, 5),
        description=description,
        options={
            "reduceOptions": {"calcs": ["lastNotNull"], "values": False, "fields": ""},
            "colorMode": "value",
            "graphMode": "none",
            "textMode": "value",
            "justifyMode": "center",
        },
        fieldConfig={
            "defaults": {
                "unit": unit or "none",
                "color": {"mode": "fixed", "fixedColor": color},
                **defaults,
            },
            "overrides": [],
        },
    )


# ---------------------------------------------------------------- dashboard
def build(p: Protocol) -> dict:
    s = Sql(p)
    group_label = (
        "CASE cap_group WHEN 'large_cap' THEN 'Large caps' "
        "WHEN 'small_mid_cap' THEN 'Small caps' END"
    )
    panels = [
        # Row 1: where the collection stands.
        panel(
            1,
            "Small-cap signals (stopping rule)",
            "bargauge",
            f"""
SELECT count(*) AS "Small-cap signals"
FROM v_signals
WHERE cap_group = 'small_mid_cap' AND {s.collected()}""",
            (0, 0, 8, 5),
            description=(
                f"Eligible small-cap signals since {p.data.collection_start}. The study stops at "
                f"{s.target} or on {p.stopping_rule.max_end_date}, whichever comes first."
            ),
            options={
                "reduceOptions": {"calcs": ["lastNotNull"], "values": False, "fields": ""},
                "orientation": "horizontal",
                "displayMode": "basic",
                "showUnfilled": True,
                "valueMode": "text",
                "namePlacement": "hidden",
            },
            fieldConfig={
                "defaults": {
                    "min": 0,
                    "max": s.target,
                    "color": {"mode": "fixed", "fixedColor": SMALL},
                    "decimals": 0,
                },
                "overrides": [],
            },
        ),
        stat(
            2,
            "Large-cap signals",
            f"SELECT count(*) FROM v_signals WHERE cap_group = 'large_cap' AND {s.collected()}",
            8,
            color=LARGE,
            description="Eligible large-cap signals since the start of the collection.",
        ),
        stat(
            3,
            "Days left",
            f"SELECT GREATEST({s.end_date} - (now() AT TIME ZONE 'UTC')::date, 0) AS days",
            12,
            description=f"Calendar days until the stopping date ({p.stopping_rule.max_end_date}).",
        ),
        stat(
            4,
            "Latest news stored",
            "SELECT extract(epoch FROM max(stored_at)) * 1000 AS latest FROM articles",
            16,
            unit="dateTimeFromNow",
            description="Last article written. Hours old on a weekday: check it.",
        ),
        stat(
            5,
            "Latest price bar",
            "SELECT extract(epoch FROM max(ts)) * 1000 AS latest FROM price_bars",
            20,
            unit="dateTimeFromNow",
            description=(
                "Newest 1-minute bar from Alpaca. SIP data arrives ~16 minutes late, and "
                "there are no bars outside market hours."
            ),
        ),
        # Row 2: flow over time.
        panel(
            6,
            "Eligible signals per day",
            "timeseries",
            f"""
SELECT date_trunc('day', published_at AT TIME ZONE 'America/New_York')
         AT TIME ZONE 'America/New_York' AS time,
       count(*) FILTER (WHERE cap_group = 'large_cap') AS "Large caps",
       count(*) FILTER (WHERE cap_group = 'small_mid_cap') AS "Small caps"
FROM v_signals
WHERE $__timeFilter(published_at) AND {s.eligible()}
GROUP BY 1
ORDER BY 1""",
            (0, 5, 12, 8),
            fmt="time_series",
            description="New York calendar days. Weekends are quiet.",
            fieldConfig={
                "defaults": {
                    "custom": {
                        "drawStyle": "bars",
                        "fillOpacity": 85,
                        "lineWidth": 0,
                        "barAlignment": 0,
                    },
                    "decimals": 0,
                },
                "overrides": GROUP_COLORS,
            },
            options={"legend": {"displayMode": "list", "placement": "bottom"}},
        ),
        panel(
            7,
            "Mean signal score per day",
            "timeseries",
            f"""
SELECT date_trunc('day', published_at AT TIME ZONE 'America/New_York')
         AT TIME ZONE 'America/New_York' AS time,
       avg(score) FILTER (WHERE cap_group = 'large_cap') AS "Large caps",
       avg(score) FILTER (WHERE cap_group = 'small_mid_cap') AS "Small caps"
FROM v_signals
WHERE $__timeFilter(published_at) AND {s.eligible()}
GROUP BY 1
ORDER BY 1""",
            (12, 5, 12, 8),
            fmt="time_series",
            description="The LLM's average read of the news (-1 to +1). Sentiment, not returns.",
            fieldConfig={
                "defaults": {
                    "min": -1,
                    "max": 1,
                    "decimals": 2,
                    "custom": {
                        "lineWidth": 2,
                        "showPoints": "always",
                        "pointSize": 8,
                        "spanNulls": True,
                    },
                },  # fmt: skip
                "overrides": GROUP_COLORS,
            },
            options={"legend": {"displayMode": "list", "placement": "bottom"}},
        ),
        # Row 3: coverage of the universe.
        panel(
            8,
            "Signals per ticker (collection period)",
            "table",
            f"""
SELECT t.ticker AS "Ticker",
       {group_label.replace("cap_group", "t.cap_group")} AS "Group",
       t.sector AS "Sector",
       count(v.ticker) AS "Signals",
       count(v.ticker) FILTER (WHERE {s.strong("v.")}) AS "Strong",
       round(avg(v.score)::numeric, 2) AS "Mean score",
       max(v.published_at) AS "Last signal"
FROM tickers t
LEFT JOIN v_signals v
  ON v.ticker = t.ticker AND {s.collected("v.")}
GROUP BY t.ticker, t.cap_group, t.sector
ORDER BY t.cap_group DESC, count(v.ticker) DESC, t.ticker""",
            (0, 13, 14, 12),
            description=(
                "Every watchlist ticker, including those with no signals yet: a sample "
                "dominated by a few names says less about small caps in general."
            ),
            options={"showHeader": True, "cellHeight": "sm", "footer": {"show": False}},
            fieldConfig={
                "defaults": {"custom": {"align": "auto"}},
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Signals"},
                        "properties": [
                            {
                                "id": "custom.cellOptions",
                                "value": {
                                    "type": "gauge",
                                    "mode": "basic",
                                    "valueDisplayMode": "text",
                                },
                            },
                            {"id": "color", "value": {"mode": "fixed", "fixedColor": SMALL}},
                            {"id": "custom.width", "value": 160},
                        ],  # fmt: skip
                    },
                    {
                        "matcher": {"id": "byName", "options": "Last signal"},
                        "properties": [{"id": "unit", "value": "dateTimeFromNow"}],
                    },
                    {
                        "matcher": {"id": "byName", "options": "Mean score"},
                        "properties": [{"id": "decimals", "value": 2}],
                    },
                ],
            },
        ),
        panel(
            9,
            "Event types",
            "barchart",
            f"""
SELECT event_type AS "Event type",
       count(*) FILTER (WHERE cap_group = 'large_cap') AS "Large caps",
       count(*) FILTER (WHERE cap_group = 'small_mid_cap') AS "Small caps"
FROM v_signals
WHERE {s.collected()}
GROUP BY event_type
ORDER BY count(*) DESC""",
            (14, 13, 10, 12),
            description="What the news is about, per group (collection period).",
            options={
                "orientation": "horizontal",
                "xField": "Event type",
                "showValue": "never",
                "groupWidth": 0.75,
                "barWidth": 0.9,
                "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            },
            fieldConfig={"defaults": {"decimals": 0}, "overrides": GROUP_COLORS},
        ),
        # Row 4: the data itself.
        panel(
            10,
            "Latest strong signals",
            "table",
            f"""
SELECT published_at AS "Published",
       ticker AS "Ticker",
       {group_label} AS "Group",
       score AS "Score",
       event_type AS "Event",
       publisher AS "Publisher",
       title AS "Headline",
       url
FROM v_signals
WHERE {s.collected()} AND {s.strong()}
ORDER BY published_at DESC
LIMIT 50""",
            (0, 25, 24, 10),
            description=(
                f"Signals that pass the registered 'strong' rule ({s.strong_text}). "
                "Click a headline to open the article."
            ),
            options={"showHeader": True, "cellHeight": "sm"},
            fieldConfig={
                "defaults": {},
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Score"},
                        "properties": [
                            {"id": "decimals", "value": 2},
                            {"id": "custom.width", "value": 70},
                        ],
                    },
                    {
                        "matcher": {"id": "byName", "options": "Published"},
                        "properties": [{"id": "custom.width", "value": 170}],
                    },
                    {
                        "matcher": {"id": "byName", "options": "Ticker"},
                        "properties": [{"id": "custom.width", "value": 70}],
                    },
                    {
                        "matcher": {"id": "byName", "options": "url"},
                        "properties": [{"id": "custom.hidden", "value": True}],
                    },
                    {
                        "matcher": {"id": "byName", "options": "Headline"},
                        "properties": [
                            {
                                "id": "links",
                                "value": [
                                    {
                                        "title": "Open article",
                                        "url": "${__data.fields.url}",
                                        "targetBlank": True,
                                    }
                                ],
                            }
                        ],
                    },
                ],
            },
        ),
        # Row 5: data quality.
        panel(
            11,
            "Signals excluded from the study",
            "table",
            f"""
SELECT CASE
{chr(10).join(f"         WHEN {cond} THEN {_q(reason)}" for reason, cond in s.reasons())}
         ELSE 'eligible'
       END AS "Reason",
       count(*) AS "Signals"
FROM v_signals
WHERE published_at >= {s.start}
GROUP BY 1
ORDER BY 2 DESC""",
            (0, 35, 8, 7),
            description=(
                "Every signal in the collection period, by the first protocol rule it fails. "
                "The analysis applies the same rules in the same order."
            ),
            options={"showHeader": True, "cellHeight": "sm"},
        ),
        panel(
            12,
            "Price data coverage (last 7 days)",
            "table",
            """
SELECT t.ticker AS "Ticker",
       CASE t.cap_group WHEN 'large_cap' THEN 'Large caps'
                        WHEN 'small_mid_cap' THEN 'Small caps'
                        ELSE 'Benchmark' END AS "Group",
       count(b.ts) AS "1-min bars",
       max(b.ts) AS "Latest bar"
FROM (SELECT ticker, cap_group FROM tickers
      UNION ALL SELECT 'SPY', 'benchmark') t
LEFT JOIN price_bars b ON b.ticker = t.ticker AND b.ts >= now() - interval '7 days'
GROUP BY t.ticker, t.cap_group
ORDER BY count(b.ts), t.ticker""",
            (8, 35, 16, 7),
            description=(
                "Thinly traded small caps have minutes without trades, hence fewer bars; "
                "the analysis uses the last trade at or before each moment. Zero bars = problem."
            ),
            options={"showHeader": True, "cellHeight": "sm"},
            fieldConfig={
                "defaults": {},
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "Latest bar"},
                        "properties": [{"id": "unit", "value": "dateTimeFromNow"}],
                    },
                ],
            },
        ),
    ]
    return {
        "uid": "event-study-collection",
        "title": "Event study - data collection",
        "description": (
            "Progress and health of the event-study data collection. "
            "Shows no returns on purpose: the study is analysed once, at the stopping rule."
        ),
        "tags": ["event-study", "postgres"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "5m",
        # Midnight in New York on the first collection day, in UTC.
        "time": {"from": _ny_midnight_utc(p.data.collection_start), "to": "now"},
        "links": [],
        "templating": {"list": []},
        "panels": panels,
    }


def all_sql(dashboard: dict) -> list[tuple[str, str]]:
    """(panel title, SQL) for every panel; used by the tests to run each query."""
    return [(p["title"], t["rawSql"]) for p in dashboard["panels"] for t in p["targets"]]


def main() -> None:
    dashboard = build(load_protocol(ROOT / "config" / "study.yaml"))
    OUT.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
