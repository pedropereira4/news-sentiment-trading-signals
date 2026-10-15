"""LLM evaluation: metrics with known answers, a reproducible stratified sample, blind and
resumable labelling, cached FinBERT runs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sentiment_pipeline.evaluation.cli import assemble, evaluate
from sentiment_pipeline.evaluation.finbert import run_finbert
from sentiment_pipeline.evaluation.labeling import read_labels, run_labelling
from sentiment_pipeline.evaluation.metrics import accuracy, cohen_kappa, macro_f1, score, wilson
from sentiment_pipeline.evaluation.report import plot_confusions, render_markdown, score_row
from sentiment_pipeline.evaluation.sample import draw_sample, model_text


# ---------------------------------------------------------------- metrics
def test_kappa_accuracy_and_f1_match_a_hand_calculation():
    m = np.array([[20, 5, 0], [10, 15, 0], [0, 0, 0]])
    assert accuracy(m) == pytest.approx(0.7)
    # p_o = 0.7, p_e = (25*30 + 25*20) / 50^2 = 0.5  ->  kappa = 0.2 / 0.5 = 0.4
    assert cohen_kappa(m) == pytest.approx(0.4)
    # F1: 40/55 and 30/45; the class absent on both sides carries no information
    assert macro_f1(m) == pytest.approx((40 / 55 + 30 / 45) / 2)


def test_perfect_and_chance_models():
    truth = ["negative", "neutral", "positive"] * 20
    perfect = score(truth, truth, n_boot=200)
    assert (perfect.accuracy, perfect.kappa) == (1.0, 1.0)
    assert perfect.ci["kappa"] == (1.0, 1.0)
    constant = score(truth, ["neutral"] * 60, n_boot=200)
    assert constant.accuracy == pytest.approx(1 / 3) and constant.kappa == pytest.approx(0.0)


def test_wilson_interval_is_sensible_at_zero():
    p, lo, hi = wilson(0, 10)
    assert p == 0 and lo == pytest.approx(0, abs=1e-12) and hi == pytest.approx(0.2775, abs=1e-3)


# ---------------------------------------------------------------- sample
def signals(n_large=300, n_small=40):
    t0 = datetime(2026, 9, 28, 14, tzinfo=UTC)
    rows = [
        {"article_id": f"a{i}", "ticker": "NVDA" if i < n_large else "IRDM",
         "cap_group": "large_cap" if i < n_large else "small_mid_cap",
         "published_at": t0 + timedelta(minutes=i)}
        for i in range(n_large + n_small)
    ]  # fmt: skip
    return pd.DataFrame(rows)


def test_sample_is_stratified_reproducible_and_shuffled():
    a = draw_sample(signals(), n_per_group=30, seed=1)
    assert a["cap_group"].value_counts().to_dict() == {"large_cap": 30, "small_mid_cap": 30}
    assert list(a["item"]) == list(range(1, 61))
    pd.testing.assert_frame_equal(a, draw_sample(signals().sample(frac=1, random_state=3), 30, 1))
    assert not a.equals(draw_sample(signals(), 30, seed=2))
    # Shuffled: the first items already mix both groups.
    assert a.head(20)["cap_group"].nunique() == 2


def test_small_group_is_taken_whole_when_short():
    a = draw_sample(signals(n_small=12), n_per_group=30, seed=1)
    assert (a["cap_group"] == "small_mid_cap").sum() == 12


def test_model_text_matches_what_the_llm_saw():
    assert model_text(" Beat ", "<p>Revenue up</p>") == "Beat — Revenue up"
    assert model_text("Beat", None) == "Beat"
    assert len(model_text("T", "x" * 1000)) == len("T — ") + 280


# ---------------------------------------------------------------- labelling
def blind_sample(n=4):
    return pd.DataFrame(
        {
            "item": range(1, n + 1),
            "article_id": [f"a{i}" for i in range(1, n + 1)],
            "ticker": ["NVDA", "IRDM", "AAPL", "SEZL"][:n],
            "company": ["Nvidia", "Iridium", "Apple", "Sezzle"][:n],
            "cap_group": ["large_cap", "small_mid_cap", "large_cap", "small_mid_cap"][:n],
            "published_at": ["2026-09-29 14:00"] * n,
            "publisher": ["Yahoo"] * n,
            "title": [f"Headline {i}" for i in range(1, n + 1)],
            "summary": ["Some summary"] * n,
        }
    )


def test_labelling_saves_each_answer_and_resumes(tmp_path):
    path = tmp_path / "labels.csv"
    shown = []
    answers = iter(["3", "x", "2", "s", "q"])  # pos, invalid, neutral, skip, quit
    assert run_labelling(blind_sample(), path, ask=lambda _: next(answers), show=shown.append) == 2
    assert read_labels(path)[["item", "label"]].values.tolist() == [[1, "positive"], [2, "neutral"]]
    assert any("Please answer" in s for s in shown)
    # Resume: the skipped item 3 is asked again, nothing labelled twice.
    answers = iter(["1", "0"])
    run_labelling(blind_sample(), path, ask=lambda _: next(answers), show=lambda _: None)
    assert read_labels(path)["label"].tolist() == ["positive", "neutral", "negative", "not_about"]


def test_labelling_screen_never_shows_the_llm_output(tmp_path):
    sample = blind_sample(1).assign(sentiment="positive", score=0.9)  # must not leak
    shown = []
    run_labelling(sample.drop(columns=["sentiment", "score"]), tmp_path / "l.csv",
                  ask=lambda _: "q", show=shown.append)  # fmt: skip
    screen = "\n".join(shown)
    assert "Headline 1" in screen and "0.9" not in screen


# ---------------------------------------------------------------- scoring
def test_evaluate_scores_models_by_group_and_excludes_not_about():
    sample = blind_sample()
    labels = pd.DataFrame({"item": [1, 2, 3, 4],
                           "label": ["positive", "negative", "neutral", "not_about"]})  # fmt: skip
    llm = pd.DataFrame({"article_id": ["a1", "a2", "a3", "a4"],
                        "ticker": ["NVDA", "IRDM", "AAPL", "SEZL"],
                        "llm_label": ["positive", "negative", "positive", "neutral"],
                        "llm_latency_ms": [500, 600, 700, 800]})  # fmt: skip
    data = assemble(sample, labels, llm)
    overall, by_group = evaluate(data, {"LLM": "llm_label"}, n_boot=50)
    assert overall["LLM"].n == 3 and overall["LLM"].accuracy == pytest.approx(2 / 3)
    assert by_group[("LLM", "small_mid_cap")].accuracy == 1.0  # item 4 is not_about
    assert any("baseline" in k for k in overall)


def test_finbert_runs_are_cached(tmp_path):
    calls = []

    def fake(texts):
        calls.append(len(texts))
        return ["positive"] * len(texts)

    texts = pd.Series(["a", "b"], index=pd.Index([1, 2], name="item"))
    cache = tmp_path / "finbert.csv"
    assert run_finbert(texts, fake, cache)["label"].tolist() == ["positive", "positive"]
    more = pd.Series(["a", "b", "c"], index=pd.Index([1, 2, 3], name="item"))
    run_finbert(more, fake, cache)
    assert calls == [2, 1]  # only the new item is classified


def test_report_and_chart_render(tmp_path):
    truth = ["negative", "neutral", "positive"] * 10
    s = score(truth, truth[::-1], n_boot=50)
    plot_confusions({"LLM": s, "FinBERT": s}, tmp_path / "c.png")
    md = render_markdown(n_sample=30, n_labelled=30, n_skipped=0, seed=1,
                         overall=[score_row("LLM", s)], by_group=[], attribution=[],
                         chart_file="c.png", finbert_included=True)  # fmt: skip
    assert (tmp_path / "c.png").stat().st_size > 10_000 and "Cohen" in md and "FinBERT" in md
