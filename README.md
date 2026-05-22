# LLM Audit Demo

Research tool for auditing citation and source-selection behavior across 4 agentic LLM systems with web search: OpenAI, Anthropic, Google Gemini, and Perplexity.

Each provider is queried with the same prompt and instructed to return structured JSON covering: answer, source selection justification, location bias, copyright considerations, social media use, fair dealing, and licensing. Results are stored as comparable JSON records and exportable to Excel for human review.

## Project structure

```
wrapper_demo/
├── .env                        # API keys (not committed)
├── .env.example                # Template for required keys
├── prompts.yaml                # 75 prompts: 15 topics × 5 framings
├── system_prompts.yaml         # Per-provider system prompts (JSON output format)
│
├── schema.sql                  # PostgreSQL schema for Layer 1 storage
├── src/
│   ├── db.py                   # PostgreSQL connection helpers + schema init
│   ├── schema.py               # Pydantic models: StandardResponse, Search_Result, Usage
│   ├── prompts.py              # load_prompts(), filter_prompts(), load_system_prompt()
│   ├── wrapper.py              # PROVIDERS registry + run_one() dispatcher
│   ├── storage.py              # save/load JSON runs, DB persistence, export_excel()
│   └── providers/
│       ├── perplexity_p.py     # Perplexity SDK (sonar / sonar-pro, streaming)
│       ├── openai_p.py         # OpenAI Responses API (web_search_preview tool)
│       ├── anthropic_p.py      # Anthropic SDK (web_search_20250305 tool)
│       └── gemini.py           # Google GenAI SDK (google_search grounding + thinking)
│
├── scripts/
│   ├── run.py                  # CLI entry point
│   └── view.py                 # Pretty-print a single saved JSON result
│
└── outputs/
    └── YYYY-MM-DD/
        └── <prompt_id>__<provider>__attempt<N>.json
```

## Setup

```bash
# 1. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env and fill in:
#   OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, PERPLEXITY_API_KEY
#
# 4. Optional: enable PostgreSQL Layer 1 storage
#   DATABASE_URL=postgresql://user:pass@host:5432/dbname
#   # or PGHOST / PGPORT / PGUSER / PGPASSWORD / PGDATABASE
#   ENABLE_DB_SAVE=both   # both (default when DB configured) | file | off
```

## Running

```bash
# Run a single prompt across all providers
python scripts/run.py --prompt uk-elections-2026-neutral

# Run a single prompt with specific providers
python scripts/run.py --prompt uk-elections-2026-neutral --provider perplexity gemini

# Run all prompts for a topic (5 framings × N providers)
python scripts/run.py --topic uk-elections-2026 --provider perplexity openai

# Sample each prompt 6 times and export results to Excel
python scripts/run.py --topic uk-elections-2026 --provider perplexity gemini --repeat 6 --export

# Run everything (75 prompts × 2 providers)
python scripts/run.py --all --provider perplexity gemini --repeat 6 --export

# Migrate existing JSON outputs into PostgreSQL
python scripts/migrate.py
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--prompt PROMPT_ID` | Run a single prompt by ID |
| `--topic TOPIC` | Run all 5 framings for a topic |
| `--all` | Run all 75 prompts |
| `--provider` | One or more providers (default: all 4) |
| `--repeat N` | Sample each prompt N times (default: 1); attempt numbers auto-increment, no overwriting |
| `--export` | Export this run's results to Excel after completion |

## Prompts

75 prompts across 15 UK politics/news topics, each in 5 framings:

| Framing | Description |
|---------|-------------|
| `neutral` | Balanced, factual framing |
| `left` | Left-leaning perspective |
| `right` | Right-leaning perspective |
| `outlets-national` | Ask for national outlet coverage |
| `outlets-local` | Ask for local/regional outlet coverage |

Topics include: `uk-elections-2026`, `spring-budget-2026`, `immigration-asylum-2026`, `nhs-pressures`, `housing-affordability-2026`, `teachers-strikes`, `climate-net-zero`, `bbc-censorship-festivals`, and more.

## Viewing a single result

```bash
python scripts/view.py outputs/2026-05-16/uk-elections-2026-neutral__perplexity__attempt1.json
```

Displays: metadata, prompt, answer, structured fields, search results with URLs, reasoning steps (Gemini), and raw response keys.

## Output format

Each JSON file is a `StandardResponse` (see `src/schema.py`):

| Field | Description |
|-------|-------------|
| `provider`, `model` | Which system answered |
| `prompt_id`, `framing`, `attempt_no` | Prompt identity |
| `answer` | Main response text |
| `source_selection_justification` | Model's explanation of why it chose these sources |
| `location` | Geographic bias or coverage focus noted by the model |
| `copyright_subject_matter` | Copyright nature of sources used |
| `social_media_use` | Whether/how social media was used as a source |
| `fair_dealing` | Model's fair dealing assessment |
| `licensing` | Licensing notes on cited material |
| `search_results` | List of `{url, resolved_url, title, snippet, published_date}` |
| `reasoning_steps` | Internal reasoning trace (Perplexity sonar-pro, Gemini) |
| `usage` | `{prompt_tokens, completion_tokens, total_tokens, total_cost}` |
| `latency_ms` | Wall-clock time for the API call |
| `raw` | Full original API response (excluded from Excel export) |

## Excel export

The `--export` flag (or `export_excel()` in `storage.py`) writes one row per run to `outputs/audit_export_YYYY-MM-DD.xlsx`, with all fields except `raw`. Columns are auto-sized and the header row is frozen for easy scrolling.

## PostgreSQL Layer 1

When PostgreSQL is configured (`DATABASE_URL` or `PG*` variables), `scripts/run.py` now writes every run to both flat files and the relational schema in `schema.sql`.

### Schema objects
- `queries`
- `runs`
- `raw_responses`
- `answers`
- `reasoning_steps`
- `citations`
- `source_metrics`

### Initialize + migrate existing demo files

```bash
python scripts/migrate.py
```

This creates the schema if needed, then ingests every `outputs/**/*.json` file (or a filtered subset via `--date` / `--provider`).

## Verify install

```bash
python -c "import openai, anthropic, google.genai, pydantic, perplexity; print('OK')"
```
