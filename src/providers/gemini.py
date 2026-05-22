"""Gemini adapter — google-genai SDK with google_search grounding."""
from urllib import response
import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.schema import Search_Result, Usage, StandardResponse
from src.prompts import load_system_prompt

load_dotenv()

MODEL = "gemini-3.1-pro-preview"


class GeminiSchema(BaseModel):
    query: str = ""
    answer: str  # required — no default so response_schema forces the model to fill it
    source_selection_justification: str = ""
    location: str = ""
    copyright_subject_matter: str = ""
    social_media_use: str = ""
    fair_dealing: str = ""
    licensing: str = ""


def _safe_dict(obj):
    """Recursively convert Gemini response objects to plain dicts, JSON-safe."""
    import base64
    if obj is None:
        return None
    if isinstance(obj, bytes):
        # thought_signature and similar binary fields → base64 string
        return base64.b64encode(obj).decode("ascii")
    if isinstance(obj, dict):
        return {k: _safe_dict(v) for k, v in obj.items() if k != "thought_signature"}
    if isinstance(obj, list):
        return [_safe_dict(i) for i in obj]
    if hasattr(obj, "model_dump"):
        return _safe_dict(obj.model_dump())  # recurse so nested bytes are caught
    if hasattr(obj, "__dict__"):
        return {k: _safe_dict(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


def _resolve_redirect(url: str, timeout: int = 5) -> str | None:
    """Follow HTTP redirect and return the final URL. Returns None on failure."""
    import urllib.request
    import urllib.error
    if "vertexaisearch.cloud.google.com" not in url:
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            final = resp.url
            return final if final != url else None
    except Exception:
        return None


_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}


def _url_exists(url: str, timeout: int = 5) -> bool:
    """Return True if the URL resolves to a real page (any HTTP response except 404).
    Sites blocking scrapers (403, timeout) are still treated as real.
    Only DNS failure or 404 → hallucinated URL."""
    import urllib.request
    import urllib.error
    import http.client
    try:
        req = urllib.request.Request(url, headers=_FETCH_HEADERS, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status != 404
    except urllib.error.HTTPError as e:
        return e.code != 404  # 403/429/5xx → real URL, just blocked
    except Exception:
        # Timeout, connection refused, DNS failure → treat as hallucinated
        return False


def _fetch_page_meta(url: str, timeout: int = 8) -> tuple[str | None, str | None]:
    """Fetch a URL and extract (title, published_date) from HTML meta tags.
    Returns (None, None) if the page body is inaccessible (timeout, bot block, etc.)."""
    import urllib.request
    import html
    import re as _re
    try:
        req = urllib.request.Request(url, headers=_FETCH_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read(262144)  # read first 256 KB to cover heavy <head> sections
        text = raw_bytes.decode("utf-8", errors="replace")

        # Title: <title>...</title>
        title = None
        m = _re.search(r"<title[^>]*>([^<]{1,300})</title>", text, _re.IGNORECASE)
        if m:
            title = html.unescape(m.group(1).strip())

        # Published date: try common meta tags in priority order
        date = None
        for pattern in [
            r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']{1,30})["\']',
            r'<meta[^>]+content=["\']([^"\']{1,30})["\'][^>]+property=["\']article:published_time["\']',
            r'<meta[^>]+name=["\']date["\'][^>]+content=["\']([^"\']{1,30})["\']',
            r'<meta[^>]+content=["\']([^"\']{1,30})["\'][^>]+name=["\']date["\']',
            r'<meta[^>]+name=["\']pubdate["\'][^>]+content=["\']([^"\']{1,30})["\']',
        ]:
            m = _re.search(pattern, text, _re.IGNORECASE)
            if m:
                date = m.group(1).strip()[:10]  # keep YYYY-MM-DD portion
                break

        return title, date
    except Exception:
        return None, None



def query(prompt: dict, attempt_no: int = 1) -> StandardResponse:
    model = os.environ.get("GEMINI_MODEL", MODEL)
    base = dict(
        provider="gemini",
        model=model,
        prompt_id=prompt["id"],
        prompt_text=prompt["text"],
        framing=prompt.get("framing"),
        attempt_no=attempt_no,
    )
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        system_prompt = load_system_prompt("gemini")
        t0 = time.perf_counter()
        response = client.models.generate_content(
            model=model,
            contents=prompt["text"],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_schema=GeminiSchema,
                tools=[types.Tool(google_search=types.GoogleSearch())],
                thinking_config=types.ThinkingConfig(include_thoughts=True)
            ),
        )
        
        latency_ms = int((time.perf_counter() - t0) * 1000)

        raw: dict = _safe_dict(response) or {}
        candidate0 = response.candidates[0] if response.candidates else None

        # Iterate parts: thought=True → reasoning; last non-thought part → answer JSON (via response.parsed)
        reasoning_parts: list[str] = []
        for part in getattr(getattr(candidate0, "content", None), "parts", None) or []:
            if getattr(part, "thought", False):
                t = getattr(part, "text", None)
                if t:
                    reasoning_parts.append(t)
        base["reasoning_steps"] = "\n\n".join(reasoning_parts) if reasoning_parts else None

        # response.parsed is auto-populated by the SDK when response_schema is set
        structured: GeminiSchema | None = response.parsed
        if structured is not None:
            base.update(structured.model_dump(exclude={"query", "answer"}))
            base["answer"] = structured.answer

        # --- Build search_results: grounding_chunks first, fall back to citations ---
        grounding_meta = getattr(candidate0, "grounding_metadata", None) if candidate0 else None
        chunks = list(getattr(grounding_meta, "grounding_chunks", None) or [])

        searched_seen: set[str] = set()
        searched_sources: list[Search_Result] = []

        # map index → Search_Result, convenient for cited reuse
        chunk_index_to_sr: dict[int, Search_Result] = {}

        for i, chunk in enumerate(chunks):
            web = getattr(chunk, "web", None)
            if not web:
                continue
            url = getattr(web, "uri", "") or ""
            if not url:
                continue
            
            sr = Search_Result(
                url=url,
                resolved_url=_resolve_redirect(url),
                title=getattr(web, "title", None),
            )
            chunk_index_to_sr[i] = sr 
            
            if url not in searched_seen:
                searched_seen.add(url)
                searched_sources.append(sr)

        # --- Cited: 从 grounding_supports 反查 chunks ---
        cited_seen: set[str] = set()
        cited_sources: list[Search_Result] = []

        supports = list(getattr(grounding_meta, "grounding_supports", None) or [])
        for support in supports:
            chunk_indices = getattr(support, "grounding_chunk_indices", None) or []
            for idx in chunk_indices:
                sr = chunk_index_to_sr.get(idx)
                if sr is None:
                    continue  # index 超范围，跳过
                if sr.url in cited_seen:
                    continue
                cited_seen.add(sr.url)
                cited_sources.append(sr)

        # --- Usage ---
        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0
        completion_tokens = getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0
        usage = Usage(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
            total_cost=0.0,
        )

        return StandardResponse(
            **base,
            cited_sources=cited_sources,
            searched_sources=searched_sources,
            usage=usage,
            latency_ms=latency_ms,
            raw=raw,
        )
    except Exception as exc:
        return StandardResponse(**base, error=str(exc), raw={})


if __name__ == "__main__":
    print(query({
        "id": "uk-elections-2026-neutral",
        "text": "From a left-leaning perspective, what do the results of the May 2026 local and mayoral elections mean for the UK now?",
        "framing": "left",
    }))