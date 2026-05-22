CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS queries (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    framing TEXT NOT NULL,
    text TEXT NOT NULL,
    query_embedding vector(384)
);

CREATE TABLE IF NOT EXISTS runs (
    id SERIAL PRIMARY KEY,
    query_id TEXT NOT NULL REFERENCES queries(id),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    attempt_no INT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    latency_ms INT,
    prompt_tokens INT,
    completion_tokens INT,
    total_cost DOUBLE PRECISION,
    error TEXT,
    UNIQUE (query_id, provider, attempt_no)
);

CREATE INDEX IF NOT EXISTS idx_runs_query_id ON runs(query_id);
CREATE INDEX IF NOT EXISTS idx_runs_provider ON runs(provider);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);

CREATE TABLE IF NOT EXISTS raw_responses (
    run_id INT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS answers (
    run_id INT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    answer TEXT,
    source_selection_justification TEXT,
    location TEXT,
    copyright_subject_matter TEXT,
    social_media_use TEXT,
    fair_dealing TEXT,
    licensing TEXT,
    answer_embedding vector(384)
);

CREATE TABLE IF NOT EXISTS reasoning_steps (
    id SERIAL PRIMARY KEY,
    run_id INT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_no INT NOT NULL,
    content TEXT NOT NULL,
    UNIQUE (run_id, step_no)
);

CREATE INDEX IF NOT EXISTS idx_reasoning_steps_run_id ON reasoning_steps(run_id);

CREATE TABLE IF NOT EXISTS citations (
    id SERIAL PRIMARY KEY,
    run_id INT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    url TEXT,
    resolved_url TEXT,
    title TEXT,
    snippet TEXT,
    published_date TEXT,
    rank INT,
    embedding vector(384)
);

CREATE INDEX IF NOT EXISTS idx_citations_run_id ON citations(run_id);
CREATE INDEX IF NOT EXISTS idx_citations_url ON citations(url);

CREATE TABLE IF NOT EXISTS source_metrics (
    id SERIAL PRIMARY KEY,
    domain TEXT NOT NULL,
    query_id TEXT NOT NULL REFERENCES queries(id),
    week_start DATE NOT NULL,
    appearance_count INT DEFAULT 1,
    avg_rank DOUBLE PRECISION,
    UNIQUE (domain, query_id, week_start)
);

CREATE INDEX IF NOT EXISTS idx_source_metrics_query_id ON source_metrics(query_id);
CREATE INDEX IF NOT EXISTS idx_source_metrics_week_start ON source_metrics(week_start);
