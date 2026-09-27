-- System of record for the event study. Unlike InfluxDB (30-day retention, dashboards),
-- nothing here expires. Every statement is idempotent: the writer runs this on startup.

-- The stock universe, synced from config/watchlist.yaml.
CREATE TABLE IF NOT EXISTS tickers (
    ticker      text PRIMARY KEY,
    name        text,
    cap_group   text NOT NULL CHECK (cap_group IN ('large_cap', 'small_mid_cap')),
    sector      text NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- One row per news item, with the article-level LLM result.
CREATE TABLE IF NOT EXISTS articles (
    id                text PRIMARY KEY,
    source            text NOT NULL,
    publisher         text,
    title             text NOT NULL,
    summary           text,
    url               text,
    source_tickers    text[] NOT NULL DEFAULT '{}',
    published_at      timestamptz NOT NULL,
    timestamp_source  text NOT NULL CHECK (timestamp_source IN ('feed', 'ingested')),
    ingested_at       timestamptz NOT NULL,
    enriched_at       timestamptz NOT NULL,
    sentiment         text NOT NULL CHECK (sentiment IN ('positive', 'neutral', 'negative')),
    score             real NOT NULL CHECK (score BETWEEN -1 AND 1),
    confidence        real NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    topics            text[] NOT NULL DEFAULT '{}',
    llm_provider      text NOT NULL,
    llm_model         text NOT NULL,
    llm_latency_ms    integer NOT NULL,
    schema_version    smallint NOT NULL,
    stored_at         timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS articles_published_at_idx ON articles (published_at);

-- One row per (article, company): the unit of analysis for the event study.
CREATE TABLE IF NOT EXISTS ticker_signals (
    article_id      text NOT NULL REFERENCES articles (id) ON DELETE CASCADE,
    ticker          text NOT NULL,
    -- Denormalised from articles: almost every analysis query filters on it.
    published_at    timestamptz NOT NULL,
    feed_timestamp  boolean NOT NULL,
    sentiment       text NOT NULL CHECK (sentiment IN ('positive', 'neutral', 'negative')),
    score           real NOT NULL CHECK (score BETWEEN -1 AND 1),
    confidence      real NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    event_type      text NOT NULL,
    is_new_info     boolean NOT NULL,
    llm_model       text NOT NULL,
    PRIMARY KEY (article_id, ticker)
);
CREATE INDEX IF NOT EXISTS ticker_signals_ticker_time_idx ON ticker_signals (ticker, published_at);

-- Convenience view for analysis: each signal with its article and the ticker's group/sector.
CREATE OR REPLACE VIEW v_signals AS
SELECT
    s.article_id, s.ticker, s.published_at, s.feed_timestamp,
    s.sentiment, s.score, s.confidence, s.event_type, s.is_new_info, s.llm_model,
    t.cap_group, t.sector,
    a.source, a.publisher, a.title, a.url
FROM ticker_signals s
JOIN articles a ON a.id = s.article_id
LEFT JOIN tickers t ON t.ticker = s.ticker;

-- 1-minute OHLCV bars for the watchlist and the benchmark (Alpaca market data).
-- `ts` is the start of the minute (UTC). Extended-hours bars are kept; the event study
-- decides which session to use.
CREATE TABLE IF NOT EXISTS price_bars (
    ticker       text NOT NULL,
    ts           timestamptz NOT NULL,
    open         double precision NOT NULL,
    high         double precision NOT NULL,
    low          double precision NOT NULL,
    close        double precision NOT NULL,
    volume       bigint NOT NULL,
    trade_count  integer,
    vwap         double precision,
    feed         text NOT NULL,
    PRIMARY KEY (ticker, ts)
);

-- Output of the event study (sp-event-study): one row per (signal, horizon), rewritten on
-- every run. `run_label` says whether it came from the registered analysis or a pilot run.
CREATE TABLE IF NOT EXISTS event_returns (
    article_id        text NOT NULL,
    ticker            text NOT NULL,
    horizon           text NOT NULL,
    status            text NOT NULL CHECK (status IN ('ok', 'pending', 'no_price')),
    t0                timestamptz NOT NULL,
    t1                timestamptz,
    p0                double precision,
    p1                double precision,
    spy_p0            double precision,
    spy_p1            double precision,
    ret               double precision,
    spy_ret           double precision,
    abnormal_ret      double precision,
    traded_in_window  boolean,
    run_label         text NOT NULL,
    computed_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (article_id, ticker, horizon)
);
