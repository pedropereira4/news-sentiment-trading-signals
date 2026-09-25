"""Generate grafana/dashboards/news-sentiment.json.

Keeping the dashboard as code makes Flux queries reviewable in diffs.
Run:  python scripts/build_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

DS = {"type": "influxdb", "uid": "influxdb"}
SRC = "filter(fn: (r) => r.source =~ /^${source:regex}$/)"

BASE_ARTICLES = f"""from(bucket: v.defaultBucket)
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "article_sentiment")
  |> {SRC}"""

BASE_TOPICS = f"""from(bucket: v.defaultBucket)
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r._measurement == "topic_mention")
  |> {SRC}"""

SENTIMENT_COLORS = [
    ("positive", "green"),
    ("neutral", "blue"),
    ("negative", "red"),
]


def sentiment_overrides() -> list[dict]:
    return [
        {
            "matcher": {"id": "byRegexp", "options": f".*{label}.*"},
            "properties": [
                {"id": "color", "value": {"mode": "fixed", "fixedColor": color}},
                {"id": "displayName", "value": label},
            ],
        }
        for label, color in SENTIMENT_COLORS
    ]


def panel(
    pid: int, title: str, ptype: str, query: str, grid: tuple[int, int, int, int], **extra
) -> dict:
    x, y, w, h = grid
    p = {
        "id": pid,
        "type": ptype,
        "title": title,
        "datasource": DS,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": [{"refId": "A", "datasource": DS, "query": query}],
        "fieldConfig": {"defaults": {}, "overrides": []},
        "options": {},
    }
    for k, v in extra.items():
        p[k] = v
    return p


def build() -> dict:
    panels = [
        panel(
            1,
            "Articles analysed",
            "stat",
            f'{BASE_ARTICLES}\n  |> filter(fn: (r) => r._field == "score")\n'
            "  |> group()\n  |> count()",
            (0, 0, 6, 4),
            options={"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "none"},
        ),
        panel(
            2,
            "Average sentiment",
            "stat",
            f'{BASE_ARTICLES}\n  |> filter(fn: (r) => r._field == "score")\n'
            "  |> group()\n  |> mean()",
            (6, 0, 6, 4),
            options={"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background"},
            fieldConfig={
                "defaults": {
                    "decimals": 2,
                    "min": -1,
                    "max": 1,
                    # News skews negative; only a sustained slide is worth alarming about.
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "red", "value": None},
                            {"color": "orange", "value": -0.55},
                            {"color": "blue", "value": -0.25},
                            {"color": "green", "value": 0.25},
                        ],
                    },
                },
                "overrides": [],
            },
        ),
        panel(
            3,
            "Negative share",
            "gauge",
            f"""{BASE_ARTICLES}
  |> filter(fn: (r) => r._field == "score")
  |> group()
  |> map(fn: (r) => ({{r with neg: if r.sentiment == "negative" then 1.0 else 0.0}}))
  |> mean(column: "neg")""",
            (12, 0, 6, 4),
            fieldConfig={
                "defaults": {
                    "unit": "percentunit",
                    "min": 0,
                    "max": 1,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [
                            {"color": "green", "value": None},
                            {"color": "orange", "value": 0.55},
                            {"color": "red", "value": 0.75},
                        ],
                    },
                },
                "overrides": [],
            },
        ),
        panel(
            4,
            "LLM latency per headline",
            "stat",
            f'{BASE_ARTICLES}\n  |> filter(fn: (r) => r._field == "llm_latency_ms")\n'
            "  |> group()\n  |> mean()",
            (18, 0, 6, 4),
            options={"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "none"},
            fieldConfig={"defaults": {"unit": "ms", "decimals": 0}, "overrides": []},
        ),
        panel(
            5,
            "Sentiment trend by source (30m mean)",
            "timeseries",
            f"""{BASE_ARTICLES}
  |> filter(fn: (r) => r._field == "score")
  |> group(columns: ["source"])
  |> aggregateWindow(every: 30m, fn: mean, createEmpty: false)""",
            (0, 4, 16, 9),
            fieldConfig={
                "defaults": {
                    "min": -1,
                    "max": 1,
                    "decimals": 2,
                    "displayName": "${__field.labels.source}",
                    "custom": {
                        "lineInterpolation": "smooth",
                        "spanNulls": True,
                        "showPoints": "always",
                        "pointSize": 4,
                    },
                },
                "overrides": [],
            },
        ),
        panel(
            6,
            "Sentiment distribution",
            "piechart",
            f"""{BASE_ARTICLES}
  |> filter(fn: (r) => r._field == "score")
  |> group(columns: ["sentiment"])
  |> count()""",
            (16, 4, 8, 9),
            options={
                "reduceOptions": {"calcs": ["lastNotNull"], "values": False},
                "pieType": "donut",
                "legend": {
                    "displayMode": "table",
                    "placement": "right",
                    "values": ["value", "percent"],
                },
            },
            fieldConfig={"defaults": {}, "overrides": sentiment_overrides()},
        ),
        panel(
            7,
            "Headline volume by sentiment (1h)",
            "timeseries",
            f"""{BASE_ARTICLES}
  |> filter(fn: (r) => r._field == "score")
  |> group(columns: ["sentiment"])
  |> aggregateWindow(every: 1h, fn: count, createEmpty: true)""",
            (0, 13, 16, 8),
            fieldConfig={
                "defaults": {
                    "custom": {
                        "drawStyle": "bars",
                        "fillOpacity": 80,
                        "stacking": {"mode": "normal", "group": "A"},
                    }
                },
                "overrides": sentiment_overrides(),
            },
        ),
        panel(
            8,
            "Top topics",
            "barchart",
            f"""{BASE_TOPICS}
  |> filter(fn: (r) => r._field == "count")
  |> group(columns: ["topic"])
  |> sum()
  |> group()
  |> sort(columns: ["_value"], desc: true)
  |> limit(n: 12)
  |> rename(columns: {{_value: "mentions"}})""",
            (16, 13, 8, 8),
            options={
                "orientation": "horizontal",
                "xField": "topic",
                "showValue": "auto",
                "legend": {"showLegend": False},
            },
        ),
        panel(
            9,
            "Topic sentiment (min. 2 mentions)",
            "table",
            f"""{BASE_TOPICS}
  |> filter(fn: (r) => r._field == "score")
  |> group(columns: ["topic"])
  |> reduce(
      identity: {{total: 0.0, mentions: 0}},
      fn: (r, accumulator) => ({{total: accumulator.total + r._value,
                                 mentions: accumulator.mentions + 1}}))
  |> map(fn: (r) => ({{topic: r.topic, mentions: r.mentions,
                      avg_score: r.total / float(v: r.mentions)}}))
  |> group()
  |> filter(fn: (r) => r.mentions >= 2)
  |> sort(columns: ["avg_score"])""",
            (0, 21, 8, 10),
            transformations=[
                {
                    "id": "organize",
                    "options": {"indexByName": {"topic": 0, "mentions": 1, "avg_score": 2}},
                }
            ],
            fieldConfig={
                "defaults": {},
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "topic"},
                        "properties": [{"id": "custom.width", "value": 150}],
                    },
                    {
                        "matcher": {"id": "byName", "options": "mentions"},
                        "properties": [{"id": "custom.width", "value": 90}],
                    },
                    {
                        "matcher": {"id": "byName", "options": "avg_score"},
                        "properties": [
                            {"id": "decimals", "value": 2},
                            {"id": "min", "value": -1},
                            {"id": "max", "value": 1},
                            # A gauge inside a narrow cell clipped the number; plain
                            # coloured text reads better at this width.
                            {"id": "custom.cellOptions", "value": {"type": "color-text"}},
                            {"id": "color", "value": {"mode": "continuous-RdYlGr"}},
                        ],
                    },
                ],
            },
        ),
        panel(
            10,
            "Latest headlines",
            "table",
            f"""{BASE_ARTICLES}
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> group()
  |> keep(columns: ["_time", "source", "sentiment", "score", "title", "topics", "url"])
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 100)""",
            (8, 21, 16, 10),
            options={"showHeader": True, "cellHeight": "sm"},
            fieldConfig={
                "defaults": {},
                "overrides": [
                    {
                        "matcher": {"id": "byName", "options": "score"},
                        "properties": [
                            {"id": "decimals", "value": 2},
                            {"id": "custom.cellOptions", "value": {"type": "color-text"}},
                            {"id": "color", "value": {"mode": "continuous-RdYlGr"}},
                            {"id": "min", "value": -1},
                            {"id": "max", "value": 1},
                            {"id": "custom.width", "value": 70},
                        ],
                    },
                    {
                        "matcher": {"id": "byName", "options": "url"},
                        "properties": [
                            {"id": "custom.hidden", "value": True},
                        ],
                    },
                    {
                        "matcher": {"id": "byName", "options": "title"},
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
                            },
                        ],
                    },
                ],
            },
        ),
    ]

    return {
        "uid": "news-sentiment",
        "title": "Real-time News Sentiment",
        "tags": ["kafka", "llm", "sentiment"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "30s",
        "time": {"from": "now-24h", "to": "now"},
        "templating": {
            "list": [
                {
                    "name": "source",
                    "label": "Source",
                    "type": "query",
                    "datasource": DS,
                    "query": """import "influxdata/influxdb/schema"
schema.tagValues(bucket: v.defaultBucket, tag: "source",
  predicate: (r) => r._measurement == "article_sentiment", start: -30d)""",
                    "multi": True,
                    "includeAll": True,
                    "allValue": ".*",
                    "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
                    "refresh": 2,
                }
            ]
        },
        "panels": panels,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "grafana" / "dashboards" / "news-sentiment.json"
    out.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
