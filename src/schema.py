from __future__ import annotations
from pydantic import BaseModel
from typing import Any
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "ref_src", "ref_url",
    "source", "share", "_unique_id",
}

def normalize_url(url: str) -> str:
    try:
        p = urlparse(url)
    except Exception:
        return url
    
    # 1. lowercase scheme + netloc
    scheme = p.scheme.lower() or "https"
    netloc = p.netloc.lower()
    
    # 2. remove tracking query params
    params = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
              if k.lower() not in _TRACKING_PARAMS]
    query = urlencode(params)
    # 3. remove trailing slash (unless the whole path is "/")
    path = p.path.rstrip("/") if p.path != "/" else "/"
    # 4. fragment usually doesn't matter for content, drop it
    return urlunparse((scheme, netloc, path, p.params, query, ""))


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
    cited_sources: list[Search_Result] = [] # same as search_results, but only sources that were cited
    searched_sources: list[Search_Result] = [] # all sources that were searched
    

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
