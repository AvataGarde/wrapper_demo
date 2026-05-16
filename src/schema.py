from __future__ import annotations
from pydantic import BaseModel
from typing import Any
from datetime import datetime, timezone


class Search_Result(BaseModel):
    url: str                          # original URL as returned by the API
    resolved_url: str | None = None   # final URL after following redirects (Gemini only)
    title: str | None = None
    snippet: str | None = None       # cited_text / annotation excerpt
    published_date: str | None = None  # Perplexity gives this directly

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float

class StandardResponse(BaseModel):
    # Metadata
    provider: str                    # "openai" | "anthropic" | "gemini" | "perplexity"
    model: str                       # specific model name
    prompt_id: str
    prompt_text: str
    framing: str | None = None       # "neutral" | "left" | "right" | "intersectional"
    timestamp: str = ""
    attempt_no: int = 1

    # Content (parsed by adapter from raw)
    answer: str = ""
    source_selection_justification: str | None = None
    location: str | None = None
    copyright_subject_matter: str | None = None
    social_media_use: str | None = None
    fair_dealing: str | None = None
    licensing: str | None = None
    search_results: list[Search_Result] = []

    # Optional
    reasoning_steps: str | None = None

    # Full raw API response — never omit
    raw: dict[str, Any] = {}

    # Call stats
    usage: Usage | None = None
    latency_ms: int | None = None
    error: str | None = None
    

    def model_post_init(self, __context: Any) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
