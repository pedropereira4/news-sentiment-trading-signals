from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from sentiment_pipeline.config import LLMSettings
from sentiment_pipeline.enricher.main import enrich
from sentiment_pipeline.llm.clients import ClassificationBatch, LLMClient, MockClient
from sentiment_pipeline.llm.prompt import LLMParseError, build_user_prompt, parse_results
from sentiment_pipeline.producer.rss_producer import SeenStore, parse_feed
from sentiment_pipeline.schemas import LLMResult, RawArticle, article_id
from sentiment_pipeline.writer.influx_writer import point_time_ns, to_points


def make_article(i: int, title: str = "Markets surge after record growth") -> RawArticle:
    return RawArticle(
        id=article_id(f"https://example.com/{i}", title),
        source="test",
        title=title,
        url=f"https://example.com/{i}",
        published_at=datetime(2026, 9, 17, 12, 0, tzinfo=UTC),
    )


# ---------------------------------------------------------------- parsing
def test_parse_results_handles_code_fences_and_bad_items():
    text = """```json
    {"results": [
        {"index": 0, "sentiment": "Positive", "score": 0.7, "confidence": 0.9,
         "topics": ["AI", "ai", "  Big   Tech "]},
        {"index": 1, "sentiment": "angry", "score": 0.1, "confidence": 0.5, "topics": []},
        {"index": 7, "sentiment": "neutral", "score": 0, "confidence": 0.5, "topics": []}
    ]}
    ```"""
    results = parse_results(text, expected=2)
    assert set(results) == {0}  # invalid label and out-of-range index are dropped
    assert results[0].sentiment == "positive"
    assert results[0].topics == ["ai", "big tech"]


def test_parse_results_accepts_bare_list_and_surrounding_prose():
    text = 'Sure! [{"index": 0, "sentiment": "negative", "score": -0.5, "confidence": 1}] done'
    assert parse_results(text, expected=1)[0].score == -0.5


def test_parse_results_raises_without_json():
    with pytest.raises(LLMParseError):
        parse_results("I cannot help with that", expected=1)


def test_prompt_numbers_headlines_and_strips_html():
    a = make_article(0).model_copy(update={"summary": "<p>Some <b>detail</b></p>"})
    prompt = build_user_prompt([a, make_article(1, "Second")])
    assert "0. [test] Markets surge after record growth — Some detail" in prompt
    assert "1. [test] Second" in prompt


# ---------------------------------------------------------------- enrichment
class FlakyClient(LLMClient):
    """Fails on batches > 1 and never returns a result for titles containing 'bad'."""

    provider = "fake"

    def __init__(self) -> None:
        super().__init__(LLMSettings())

    def classify(self, articles):
        if len(articles) > 1:
            raise RuntimeError("batch too big")
        if "bad" in articles[0].title:
            return ClassificationBatch({}, 10)
        return ClassificationBatch(
            {0: LLMResult(index=0, sentiment="neutral", score=0, confidence=1, topics=["x"])}, 10
        )

    def _complete(self, user_prompt):
        raise NotImplementedError


def test_enrich_splits_failed_batch_and_reports_unclassifiable_items():
    articles = [make_article(0, "good one"), make_article(1, "bad one"), make_article(2, "ok")]
    enriched, errors = enrich(FlakyClient(), articles)
    assert [e.title for e in enriched] == ["good one", "ok"]
    assert list(errors) == [1]


def test_mock_client_end_to_end():
    enriched, errors = enrich(
        MockClient(LLMSettings()),
        [
            make_article(0, "Stocks surge on record AI growth"),
            make_article(1, "Floods and fire leave dozens dead"),
        ],
    )
    assert not errors
    assert enriched[0].sentiment == "positive" and "ai" in enriched[0].topics
    assert enriched[1].sentiment == "negative"


# ---------------------------------------------------------------- producer
RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Hello world</title><link>https://x.com/1</link>
<pubDate>Thu, 17 Sep 2026 10:00:00 GMT</pubDate></item>
<item><title></title><link>https://x.com/2</link></item>
</channel></rss>"""


def test_parse_feed_skips_empty_titles_and_parses_dates():
    arts = parse_feed("src", RSS)
    assert len(arts) == 1
    assert arts[0].published_at == datetime(2026, 9, 17, 10, 0, tzinfo=UTC)
    assert arts[0].id == article_id("https://x.com/1", "Hello world")


def test_seen_store_is_bounded_and_persistent(tmp_path):
    path = tmp_path / "seen.json"
    store = SeenStore(str(path), max_size=2)
    for k in ["a", "b", "c"]:
        store.add(k)
    store.save()
    reloaded = SeenStore(str(path), max_size=2)
    assert "a" not in reloaded and "b" in reloaded and "c" in reloaded


# ---------------------------------------------------------------- writer
def test_points_are_deterministic_for_idempotent_writes():
    raw = make_article(0)
    result = LLMResult(
        index=0, sentiment="positive", score=0.8, confidence=0.9, topics=["markets", "economy"]
    )
    from sentiment_pipeline.schemas import EnrichedArticle

    art = EnrichedArticle.from_parts(raw, result, provider="mock", model="m", latency_ms=5)
    points = to_points(art)
    assert len(points) == 3  # 1 article + 2 topics
    assert point_time_ns(art) == point_time_ns(art.model_copy())
    line = points[0].to_line_protocol()
    assert line.startswith("article_sentiment,llm_model=m,sentiment=positive,source=test ")
    assert 'topics="markets, economy"' in line


def test_config_strips_quotes_from_influx_values(monkeypatch):
    from sentiment_pipeline.config import WriterSettings

    monkeypatch.setenv("INFLUXDB_TOKEN", '"tok en" ')
    monkeypatch.setenv("INFLUXDB_ORG", " sentiment ")
    s = WriterSettings(_env_file=None)
    assert s.influxdb_token == "tok en"
    assert s.influxdb_org == "sentiment"
    assert s.max_point_age_days < 30  # must stay inside the bucket's retention


def test_ollama_client_builds_openai_style_request_and_parses_reply():
    """Fake transport: verifies URL, payload and parsing without a running Ollama."""
    import httpx

    from sentiment_pipeline.llm.clients import OllamaClient, build_client

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"results": [{"index": 0, "sentiment": "negative",'
                            ' "score": -0.8, "confidence": 0.9, "topics": ["war"]}]}'
                        }
                    }
                ]
            },
        )

    settings = LLMSettings(
        _env_file=None,
        llm_provider="ollama",
        llm_model="llama3.2:3b",
        ollama_base_url="http://host.docker.internal:11434",
    )
    client = build_client(settings)
    assert isinstance(client, OllamaClient) and client.provider == "ollama"
    client.http = httpx.Client(transport=httpx.MockTransport(handler))

    out = client.classify([make_article(0, "War escalates")])
    assert seen["url"] == "http://host.docker.internal:11434/v1/chat/completions"
    assert seen["body"]["model"] == "llama3.2:3b"
    assert seen["body"]["max_tokens"] == settings.llm_max_output_tokens  # no runaway replies
    assert seen["body"]["messages"][1]["content"].startswith("Classify these news items:")
    assert out.results[0].sentiment == "negative"
    assert out.results[0].topics == ["war"]


def test_max_poll_interval_covers_worst_case_llm_retries():
    """Kafka must tolerate a batch where every LLM attempt times out."""
    from sentiment_pipeline.config import EnricherSettings

    s = EnricherSettings(_env_file=None, llm_timeout_seconds=120, llm_max_retries=4)
    worst_case_ms = 120 * 4 * 1000
    assert s.max_poll_interval_ms > worst_case_ms
    # Never drops below Kafka's own default, even with a fast provider.
    assert EnricherSettings(_env_file=None, llm_timeout_seconds=5).max_poll_interval_ms >= 300_000


# ---------------------------------------------------------------- schema v2: ticker signals
def test_ticker_signal_normalises_ticker_and_event_type():
    from sentiment_pipeline.schemas import TickerSignal

    base = {"sentiment": "Positive", "score": 0.5, "confidence": 0.8, "is_new_info": True}
    assert TickerSignal(ticker="$nvda ", event_type="M&A", **base).ticker == "NVDA"
    assert TickerSignal(ticker="BRK.B", event_type="M&A", **base).event_type == "m&a"
    assert TickerSignal(ticker="AAPL", event_type="Analyst Rating", **base).event_type == (
        "analyst_rating"
    )
    # An unknown label keeps the signal instead of discarding it.
    assert TickerSignal(ticker="AAPL", event_type="rumour", **base).event_type == "other"
    with pytest.raises(ValueError):
        TickerSignal(ticker="Apple Inc", **base)


def test_llm_result_drops_bad_and_duplicate_signals_but_keeps_the_item():
    r = LLMResult.model_validate(
        {
            "index": 0,
            "sentiment": "negative",
            "score": -0.4,
            "confidence": 0.9,
            "topics": ["contracts"],
            "signals": [
                {
                    "ticker": "MSFT",
                    "sentiment": "positive",
                    "score": 0.6,
                    "confidence": 0.8,
                    "event_type": "product",
                    "is_new_info": True,
                },
                {
                    "ticker": "msft",
                    "sentiment": "negative",
                    "score": -0.1,
                    "confidence": 0.3,
                    "event_type": "product",
                    "is_new_info": True,
                },  # duplicate ticker
                {
                    "ticker": "GOOGL",
                    "sentiment": "negative",
                    "score": -3,
                    "confidence": 0.8,
                    "event_type": "product",
                    "is_new_info": True,
                },  # score out of range
                {
                    "ticker": "not a ticker",
                    "sentiment": "neutral",
                    "score": 0,
                    "confidence": 0.5,
                    "is_new_info": False,
                },
            ],
        }
    )
    assert [s.ticker for s in r.signals] == ["MSFT"]
    assert r.signals[0].score == 0.6  # first occurrence wins


def test_parse_results_reads_per_ticker_signals():
    text = json.dumps(
        {
            "results": [
                {
                    "index": 0,
                    "sentiment": "positive",
                    "score": 0.3,
                    "confidence": 0.8,
                    "topics": ["smartphones"],
                    "signals": [
                        {
                            "ticker": "AAPL",
                            "sentiment": "positive",
                            "score": 0.6,
                            "confidence": 0.8,
                            "event_type": "product",
                            "is_new_info": True,
                        },
                        {
                            "ticker": "SSNLF",
                            "sentiment": "negative",
                            "score": -0.4,
                            "confidence": 0.7,
                            "event_type": "product",
                            "is_new_info": True,
                        },
                    ],
                }
            ]
        }
    )
    result = parse_results(text, expected=1)[0]
    assert {s.ticker: s.sentiment for s in result.signals} == {
        "AAPL": "positive",
        "SSNLF": "negative",
    }


def test_raw_article_normalises_tickers_and_reads_v1_messages():
    art = make_article(0).model_copy(update={"tickers": []})
    art = RawArticle.model_validate(
        {**art.model_dump(), "tickers": ["aapl", "$AAPL", "??", "tsla"]}
    )
    assert art.tickers == ["AAPL", "TSLA"]

    # Messages written before schema v2 are still on the topic: they must keep parsing.
    v1 = {
        "schema_version": 1,
        "id": "abc",
        "source": "bbc_business",
        "title": "Old message",
        "published_at": "2026-09-17T10:00:00Z",
    }
    old = RawArticle.model_validate(v1)
    assert old.tickers == [] and old.timestamp_source == "feed"


def test_prompt_passes_source_tickers_to_the_llm():
    art = RawArticle.model_validate({**make_article(0).model_dump(), "tickers": ["NVDA"]})
    prompt = build_user_prompt([art, make_article(1, "No tickers here")])
    assert "0. [test] Markets surge after record growth (Tickers: NVDA)" in prompt
    assert prompt.endswith("1. [test] No tickers here")  # no hint without tickers


def test_mock_client_emits_one_signal_per_source_ticker():
    art = RawArticle.model_validate(
        {**make_article(0, "Nvidia earnings surge to record").model_dump(), "tickers": ["NVDA"]}
    )
    enriched, errors = enrich(MockClient(LLMSettings()), [art, make_article(1, "No company")])
    assert not errors
    [sig] = enriched[0].signals
    assert (sig.ticker, sig.sentiment, sig.event_type) == ("NVDA", "positive", "earnings")
    assert enriched[1].signals == []
    assert enriched[0].schema_version == 2


def test_parse_feed_flags_entries_without_a_date():
    rss = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
    <item><title>Dated</title><link>https://x.com/1</link>
    <pubDate>Thu, 17 Sep 2026 10:00:00 GMT</pubDate></item>
    <item><title>Undated</title><link>https://x.com/2</link></item>
    </channel></rss>"""
    dated, undated = parse_feed("src", rss)
    assert dated.timestamp_source == "feed"
    assert undated.timestamp_source == "ingested"


def test_writer_emits_one_ticker_signal_point_per_company():
    from sentiment_pipeline.schemas import EnrichedArticle, TickerSignal

    sig = TickerSignal(
        ticker="AAPL",
        sentiment="positive",
        score=0.6,
        confidence=0.8,
        event_type="product",
        is_new_info=True,
    )
    result = LLMResult(
        index=0, sentiment="positive", score=0.3, confidence=0.8, topics=["tech"], signals=[sig]
    )
    art = EnrichedArticle.from_parts(
        make_article(0), result, provider="mock", model="m", latency_ms=5
    )
    points = to_points(art)
    assert len(points) == 3  # article + 1 topic + 1 ticker signal
    line = points[-1].to_line_protocol()
    assert line.startswith(
        "ticker_signal,event_type=product,sentiment=positive,source=test,ticker=AAPL "
    )
    assert "is_new_info=true" in line and "feed_timestamp=true" in line


# ---------------------------------------------------------------- watchlist-aware enrichment
def _signal(ticker: str, score: float = 0.5):
    from sentiment_pipeline.schemas import TickerSignal

    return TickerSignal(
        ticker=ticker,
        sentiment="positive" if score > 0 else "negative",
        score=score,
        confidence=0.8,
        event_type="product",
        is_new_info=True,
    )


def test_placeholder_tickers_are_not_signals():
    from sentiment_pipeline.schemas import normalise_ticker

    assert normalise_ticker("NONE") is None and normalise_ticker("n/a") is None
    r = LLMResult.model_validate(
        {
            "index": 0,
            "sentiment": "neutral",
            "score": 0,
            "confidence": 0.5,
            "signals": [
                {
                    "ticker": "NONE",
                    "sentiment": "neutral",
                    "score": 0,
                    "confidence": 0.5,
                    "is_new_info": False,
                },
                {
                    "ticker": "NVDA",
                    "sentiment": "positive",
                    "score": 0.5,
                    "confidence": 0.8,
                    "is_new_info": True,
                },
            ],
        }
    )
    assert [s.ticker for s in r.signals] == ["NVDA"]


def test_signals_outside_the_watchlist_are_dropped_and_reported():
    from sentiment_pipeline.enricher.main import restrict_to_universe
    from sentiment_pipeline.schemas import EnrichedArticle

    result = LLMResult(
        index=0,
        sentiment="positive",
        score=0.4,
        confidence=0.8,
        signals=[_signal("NVDA"), _signal("TOKIO"), _signal("TWSE", -0.2)],
    )
    art = EnrichedArticle.from_parts(make_article(0), result, provider="m", model="m", latency_ms=1)
    [kept], dropped = restrict_to_universe([art], {"NVDA", "AAPL"})
    assert [s.ticker for s in kept.signals] == ["NVDA"]
    assert sorted(dropped) == ["TOKIO", "TWSE"]
    # No universe configured: nothing is filtered.
    [same], none = restrict_to_universe([art], None)
    assert len(same.signals) == 3 and none == []


def test_prompt_lists_the_companies_of_interest():
    prompt = build_user_prompt([make_article(0)], "NVDA (Nvidia), AAPL (Apple)")
    assert prompt.startswith("Companies of interest: NVDA (Nvidia), AAPL (Apple)\n")
    assert build_user_prompt([make_article(0)]).startswith("Classify these news items:")


def test_repo_watchlist_gives_the_llm_company_names():
    from pathlib import Path

    from sentiment_pipeline.enricher.main import _load_universe

    root = Path(__file__).resolve().parents[1]
    wl = _load_universe(str(root / "config" / "watchlist.yaml"))
    assert all(e.name for e in wl.tickers)
    assert "NVDA (Nvidia)" in wl.prompt_hint()
    assert _load_universe(str(root / "missing.yaml")) is None
    assert _load_universe("") is None
