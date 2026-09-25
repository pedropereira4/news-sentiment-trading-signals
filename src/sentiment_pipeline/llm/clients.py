"""LLM provider clients behind a single `classify()` interface.

- OpenRouterClient: OpenAI-compatible chat completions (any model on OpenRouter)
- AnthropicClient:  Anthropic Messages API
- MockClient:       offline, keyword-based — lets the whole stack run without an API key
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from sentiment_pipeline.config import LLMSettings
from sentiment_pipeline.llm.prompt import SYSTEM_PROMPT, build_user_prompt, parse_results
from sentiment_pipeline.schemas import LLMResult, RawArticle, TickerSignal

log = logging.getLogger(__name__)


@dataclass
class ClassificationBatch:
    results: dict[int, LLMResult]
    latency_ms: int


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 409, 429} or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


class LLMClient(ABC):
    provider: str = "base"

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.model = settings.llm_model

    def classify(self, articles: Sequence[RawArticle]) -> ClassificationBatch:
        start = time.perf_counter()
        text = self._complete_with_retry(build_user_prompt(articles))
        latency = int((time.perf_counter() - start) * 1000)
        return ClassificationBatch(parse_results(text, len(articles)), latency)

    def _complete_with_retry(self, user_prompt: str) -> str:
        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self.settings.llm_max_retries),
            wait=wait_exponential_jitter(initial=1, max=30),
            reraise=True,
            before_sleep=lambda rs: log.warning(
                "LLM call failed (%s), retry %d", rs.outcome.exception(), rs.attempt_number
            ),
        )
        def _call() -> str:
            return self._complete(user_prompt)

        return _call()

    @abstractmethod
    def _complete(self, user_prompt: str) -> str: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class _HttpClient(LLMClient, ABC):
    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        self.http = httpx.Client(timeout=settings.llm_timeout_seconds)

    def close(self) -> None:
        self.http.close()


class OpenAICompatibleClient(_HttpClient, ABC):
    """Any /v1/chat/completions endpoint: OpenRouter, Ollama, vLLM, OpenAI itself."""

    base_url: str = ""

    def headers(self) -> dict[str, str]:
        return {}

    def _complete(self, user_prompt: str) -> str:
        resp = self.http.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=self.headers(),
            json={
                "model": self.model,
                "temperature": self.settings.llm_temperature,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class OpenRouterClient(OpenAICompatibleClient):
    provider = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "X-Title": "realtime-sentiment-pipeline",
        }


class OllamaClient(OpenAICompatibleClient):
    """A model running locally through Ollama — no API key, no cost, works offline."""

    provider = "ollama"

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        self.base_url = f"{settings.ollama_base_url.rstrip('/')}/v1"


class AnthropicClient(_HttpClient):
    provider = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def _complete(self, user_prompt: str) -> str:
        resp = self.http.post(
            self.URL,
            headers={
                "x-api-key": self.settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": self.model,
                "max_tokens": 4096,  # per-ticker signals make the output longer
                "temperature": self.settings.llm_temperature,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}],
            },
        )
        resp.raise_for_status()
        blocks = resp.json()["content"]
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


class MockClient(LLMClient):
    """Deterministic lexicon-based classifier for local development and tests.

    Emits one signal per ticker supplied by the source; it cannot discover companies itself.
    """

    provider = "mock"
    POSITIVE = {
        "win",
        "wins",
        "growth",
        "record",
        "surge",
        "rise",
        "rises",
        "boost",
        "deal",
        "breakthrough",
        "success",
        "peace",
        "gain",
        "gains",
        "launch",
        "approve",
    }
    NEGATIVE = {
        "war",
        "crash",
        "falls",
        "fall",
        "dead",
        "death",
        "killed",
        "crisis",
        "attack",
        "loss",
        "losses",
        "fraud",
        "strike",
        "flood",
        "fire",
        "ban",
        "layoffs",
        "cuts",
    }
    EVENTS = {
        "earnings": {"earnings", "results", "revenue", "profit", "eps"},
        "guidance": {"guidance", "outlook", "forecast"},
        "m&a": {"acquire", "acquires", "acquisition", "merger", "buyout"},
        "analyst_rating": {"upgrade", "upgrades", "downgrade", "downgrades"},
        "regulatory_legal": {"fda", "lawsuit", "probe", "fine", "antitrust"},
    }
    TOPICS = {
        "ai": {"ai", "openai", "anthropic", "chatgpt"},
        "economy": {"inflation", "rates", "economy", "gdp", "market", "markets", "stocks"},
        "politics": {"election", "president", "minister", "government", "parliament"},
        "technology": {"tech", "apple", "google", "microsoft", "software", "chip", "chips"},
    }

    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(settings)
        self.model = "mock-lexicon"

    def classify(self, articles: Sequence[RawArticle]) -> ClassificationBatch:
        results = {}
        for i, a in enumerate(articles):
            words = {w.strip(".,:;!?'\"()").lower() for w in a.title.split()}
            pos, neg = len(words & self.POSITIVE), len(words & self.NEGATIVE)
            score = max(-1.0, min(1.0, (pos - neg) * 0.5))
            sentiment = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
            topics = [t for t, kws in self.TOPICS.items() if words & kws] or ["general"]
            event = next((e for e, kws in self.EVENTS.items() if words & kws), "other")
            signals = [
                TickerSignal(
                    ticker=t,
                    sentiment=sentiment,
                    score=score,
                    confidence=0.5,
                    event_type=event,
                    is_new_info=True,
                )
                for t in a.tickers
            ]
            results[i] = LLMResult(
                index=i,
                sentiment=sentiment,
                score=score,
                confidence=0.5,
                topics=topics,
                signals=signals,
            )
        return ClassificationBatch(results, latency_ms=0)

    def _complete(self, user_prompt: str) -> str:  # pragma: no cover - unused
        raise NotImplementedError


def build_client(settings: LLMSettings) -> LLMClient:
    if settings.llm_provider == "ollama":
        return OllamaClient(settings)
    if settings.llm_provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
        return OpenRouterClient(settings)
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        return AnthropicClient(settings)
    return MockClient(settings)
