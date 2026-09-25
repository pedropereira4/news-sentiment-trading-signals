"""Prompt construction and robust parsing of the LLM's JSON output."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from pydantic import ValidationError

from sentiment_pipeline.schemas import LLMResult, RawArticle

SYSTEM_PROMPT = """You are an equity research analyst. You read financial news and judge what each item means for the stock price of the companies involved.

For each numbered news item return:
- sentiment, score, confidence: the overall tone of the item for markets
  - sentiment: "positive", "neutral" or "negative"
  - score: float from -1.0 (very negative) to 1.0 (very positive); neutral is close to 0
  - confidence: float from 0.0 to 1.0
- topics: 1-3 short, lowercase, canonical topics (e.g. "earnings", "ai", "interest rates")
- signals: one entry per publicly traded US-listed company the item is materially about, with:
  - ticker: the company's US ticker symbol (e.g. "AAPL", "BRK.B")
  - sentiment, score, confidence: the likely impact on THAT company's stock price
  - event_type: one of "earnings", "guidance", "m&a", "analyst_rating", "product", "regulatory_legal", "management", "capital", "macro", "commentary", "other"
  - is_new_info: true if the item reports a new fact or event; false for recaps, opinion pieces, "stocks to watch" lists or news already reported

Rules:
- Judge impact on the stock, not whether the event is good for society. A layoff that cuts costs can be positive for the stock.
- Weigh the whole item: "revenue beats estimates but guidance cut" is usually negative.
- The same item can be positive for one company and negative for another (e.g. a rival losing a contract).
- If an item lists "Tickers:", include a signal for each of them; add others only if clearly the subject.
- Only include companies you can map to a ticker with confidence. No signals is a valid answer (e.g. general politics).
- Reuse the same topic wording for the same concept across items.
- Respond with JSON ONLY, no prose, in exactly this shape:
{"results": [{"index": 0, "sentiment": "positive", "score": 0.6, "confidence": 0.8, "topics": ["earnings"], "signals": [{"ticker": "NVDA", "sentiment": "positive", "score": 0.7, "confidence": 0.85, "event_type": "earnings", "is_new_info": true}]}]}
- Return exactly one result per input index."""


def build_user_prompt(articles: Sequence[RawArticle]) -> str:
    lines = []
    for i, a in enumerate(articles):
        text = a.title.strip()
        if a.summary:
            summary = re.sub(r"<[^>]+>", "", a.summary).strip()[:280]
            if summary:
                text += f" — {summary}"
        line = f"{i}. [{a.source}] {text}"
        if a.tickers:
            line += f" (Tickers: {', '.join(a.tickers)})"
        lines.append(line)
    return "Classify these news items:\n" + "\n".join(lines)


class LLMParseError(ValueError):
    pass


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_json(text: str) -> object:
    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} or [...] block.
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if not match:
            raise LLMParseError(f"No JSON found in LLM output: {text[:200]!r}") from None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise LLMParseError(f"Invalid JSON from LLM: {exc}") from exc


def parse_results(text: str, expected: int) -> dict[int, LLMResult]:
    """Parse the model output into {index: LLMResult}.

    Invalid individual items are skipped (the caller retries/DLQs missing indices)
    instead of failing the whole batch.
    """
    data = _extract_json(text)
    items = data.get("results") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise LLMParseError("LLM JSON has no 'results' list")

    results: dict[int, LLMResult] = {}
    for item in items:
        try:
            r = LLMResult.model_validate(item)
        except ValidationError:
            continue
        if r.index < expected and r.index not in results:
            results[r.index] = r
    return results
