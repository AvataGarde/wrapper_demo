-- Algorithmic Gatekeeper Audit — PostgreSQL Schema
-- Run once: psql -d audit_db -f schema.sql

-- ── Layer 1: Queries ──────────────────────────────────────────────────────────
-- The 75 fixed prompts from prompts.yaml (15 topics × 5 framings)
CREATE TABLE IF NOT EXISTS queries (
    id          TEXT PRIMARY KEY,   -- e.g. "uk-elections-2026-neutral"
    topic       TEXT NOT NULL,
    framing     TEXT NOT NULL,      -- neutral | left | right | outlets-national | outlets-local
    text        TEXT NOT NULL
);

-- ── Layer 1: Runs ─────────────────────────────────────────────────────────────
-- One row per individual API call
CREATE TABLE IF NOT EXISTS runs (
    id                  SERIAL PRIMARY KEY,
    query_id            TEXT REFERENCES queries(id),
    provider            TEXT NOT NULL,      -- openai | anthropic | gemini | perplexity
    model               TEXT NOT NULL,
    attempt_no          INT  NOT NULL DEFAULT 1,
    timestamp           TIMESTAMPTZ NOT NULL,
    latency_ms          INT,
    prompt_tokens       INT,
    completion_tokens   INT,
    total_tokens        INT,
    total_cost          FLOAT,
    error               TEXT,
    UNIQUE (query_id, provider, attempt_no)
);

-- ── Layer 1: Raw responses ────────────────────────────────────────────────────
-- Full original API JSON, always preserved
CREATE TABLE IF NOT EXISTS raw_responses (
    run_id  INT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    payload JSONB NOT NULL
);

-- ── Layer 1: Answers ──────────────────────────────────────────────────────────
-- Structured fields extracted from the model's response
CREATE TABLE IF NOT EXISTS answers (
    run_id                          INT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    answer                          TEXT,
    source_selection_justification  TEXT,
    location                        TEXT,
    copyright_subject_matter        TEXT,
    social_media_use                TEXT,
    fair_dealing                    TEXT,
    licensing                       TEXT,
    -- Embedding stored as JSON array of floats (pgvector alternative)
    -- Replace with vector(384) once pgvector is installed
    answer_embedding                JSONB
);

-- ── Layer 1: Reasoning steps ──────────────────────────────────────────────────
-- Chain-of-thought tokens (Perplexity, Gemini thinking)
CREATE TABLE IF NOT EXISTS reasoning_steps (
    id      SERIAL PRIMARY KEY,
    run_id  INT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_no INT NOT NULL,
    content TEXT NOT NULL
);

-- ── Layer 1: Citations ────────────────────────────────────────────────────────
-- One row per cited URL per run
CREATE TABLE IF NOT EXISTS citations (
    id              SERIAL PRIMARY KEY,
    run_id          INT  NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    rank            INT  NOT NULL,          -- order returned by the API (0-based)
    url             TEXT NOT NULL,
    resolved_url    TEXT,
    title           TEXT,
    snippet         TEXT,
    published_date  TEXT,
    domain          TEXT,                   -- extracted from url for easy grouping
    -- Embedding stored as JSON array of floats
    embedding       JSONB
);

-- ── Layer 2: Enrichment ───────────────────────────────────────────────────────
-- Copyright / licensing signals fetched after collection
CREATE TABLE IF NOT EXISTS enrichment (
    citation_id             INT PRIMARY KEY REFERENCES citations(id) ON DELETE CASCADE,
    is_paywalled            BOOLEAN,
    license_type            TEXT,           -- copyright | cc | unknown
    has_robots_tdm          BOOLEAN,        -- robots.txt TDM reservation present
    is_partner_source       BOOLEAN,
    verbatim_ratio          FLOAT,          -- n-gram overlap between answer and page text
    attribution_present     BOOLEAN,
    attribution_clickable   BOOLEAN,
    raw_signals             JSONB           -- full extracted signals for reference
);

-- ── Layer 4: Similarities ─────────────────────────────────────────────────────
-- Cosine similarity scores between query, answer, and each source
CREATE TABLE IF NOT EXISTS similarities (
    run_id              INT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    citation_id         INT NOT NULL REFERENCES citations(id) ON DELETE CASCADE,
    query_answer_sim    FLOAT,
    query_source_sim    FLOAT,
    answer_source_sim   FLOAT,
    PRIMARY KEY (run_id, citation_id)
);

-- ── Layer 2: Weekly source metrics ───────────────────────────────────────────
-- Aggregated per-domain stats for the 7-day tracker
CREATE TABLE IF NOT EXISTS source_metrics (
    id                  SERIAL PRIMARY KEY,
    domain              TEXT NOT NULL,
    query_id            TEXT REFERENCES queries(id),
    week_start          DATE NOT NULL,
    appearance_count    INT  NOT NULL DEFAULT 1,
    avg_rank            FLOAT,
    UNIQUE (domain, query_id, week_start)
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_runs_query_id     ON runs(query_id);
CREATE INDEX IF NOT EXISTS idx_runs_provider     ON runs(provider);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp    ON runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_citations_run_id  ON citations(run_id);
CREATE INDEX IF NOT EXISTS idx_citations_domain  ON citations(domain);
CREATE INDEX IF NOT EXISTS idx_source_metrics_week ON source_metrics(week_start);
