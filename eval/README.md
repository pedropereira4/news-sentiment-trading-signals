# LLM label evaluation

How often does the LLM's per-company label agree with a human, and how does it compare with
FinBERT? Written before any item was labelled.

**Sample.** 200 (article, company) pairs from the event study's signals (collection period,
protocol filters): 100 large caps and 100 small caps drawn at random with seed `20261015`,
then shuffled, so any labelled prefix is still a random sample of both groups
(`sp-eval sample`).

**Labels.** One annotator, blind: the labelling tool shows the company, headline, summary,
publisher and time, never the LLM's output or any price (`sp-eval label`). The question is how
an investor reading the item at publication time would see it for *that company's* stock:
positive, neutral, negative, or not about the company.

**Metrics** (`sp-eval score --finbert`): accuracy, Cohen's κ and macro-F1 against the human
labels, with 2,000-resample bootstrap intervals, overall and per group; a majority-class
baseline; the share of LLM signals that are not about the company (Wilson interval); time and
cost per item.

**Files.** `labels.csv` (item, article, ticker, label) and `finbert.csv` (FinBERT's labels)
are committed. `sample.csv` holds the article text and stays local (`.gitignore`); it is
rebuilt from the database by the same seed.
