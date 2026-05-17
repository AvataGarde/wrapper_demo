"""OpenAI adapter — uses Responses API with web_search tool."""
from __future__ import annotations
import re
import time
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.schema import Search_Result, Usage, StandardResponse
from src.prompts import load_system_prompt

load_dotenv()

MODEL = "gpt-4o-search-preview"  # override via OPENAI_MODEL env var

class OpenAISchema(BaseModel):
    query: str = ""
    answer: str = ""
    url_list: list[str] = []
    source_selection_justification: str = ""
    location: str = ""
    copyright_subject_matter: str = ""
    social_media_use: str = ""
    fair_dealing: str = ""
    licensing: str = ""

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}

def _fetch_page_meta(url: str, timeout: int = 8) -> tuple[str | None, str | None]:
    """Fetch a URL and extract (title, published_date) from HTML meta tags."""
    import urllib.request
    import html
    import re as _re
    try:
        req = urllib.request.Request(url, headers=_FETCH_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_bytes = resp.read(262144)
        text = raw_bytes.decode("utf-8", errors="replace")

        title = None
        m = _re.search(r"<title[^>]*>([^<]{1,300})</title>", text, _re.IGNORECASE)
        if m:
            title = html.unescape(m.group(1).strip())

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
                date = m.group(1).strip()[:10]
                break

        return title, date
    except Exception:
        return None, None

def _split_response(text: str) -> tuple[str, OpenAISchema | None]:
    """Split model output into (answer_prose, structured)."""
    text = text.strip()

    # Fenced JSON block
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            structured = OpenAISchema.model_validate_json(fence.group(1))
            return text[:fence.start()].strip(), structured
        except Exception:
            pass

    # Bare JSON block at end of text
    brace = re.search(r"\n\s*(\{[\s\S]*\})\s*$", text)
    if brace:
        try:
            structured = OpenAISchema.model_validate_json(brace.group(1))
            return text[:brace.start()].strip(), structured
        except Exception:
            pass

    # Whole text is JSON
    try:
        structured = OpenAISchema.model_validate_json(text)
        return structured.answer or text, structured
    except Exception:
        pass

    return text, None

def query(prompt: dict, attempt_no: int = 1) -> StandardResponse:
    model = os.environ.get("OPENAI_MODEL", MODEL)
    base = dict(
        provider="openai",
        model=model,
        prompt_id=prompt["id"],
        prompt_text=prompt["text"],
        framing=prompt.get("framing"),
        attempt_no=attempt_no,
    )
    try:
        client = OpenAI()
        system_prompt = load_system_prompt("openai")
        t0 = time.perf_counter()
        response = client.responses.create(
            model=model,
            tools=[
                {
                    "type": "web_search_preview",
                }
            ],
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt["text"]}
            ],
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        raw: dict = response.model_dump()
        full_text: str = response.output_text or ""

        # --- Parse structured JSON from model output ---
        answer, structured = _split_response(full_text)
        if structured is not None:
            base.update(structured.model_dump(exclude={"query", "answer"}))
        base["answer"] = answer or full_text
        base["reasoning_steps"] = raw.get("reasoning_steps")

        # Replace triple backtick JSON code blocks around the answer if present
        if base["answer"].startswith("```json"):
            base["answer"] = base["answer"].replace("```json\n", "", 1).rstrip("`").strip()

        import json
        try:
            parsed_answer = json.loads(base["answer"])
            # if parse succeeds and has 'answer' field, we unpack it
            if isinstance(parsed_answer, dict) and "answer" in parsed_answer:
                base.update({k: v for k, v in parsed_answer.items() if k not in ["query", "answer"]})
                base["answer"] = parsed_answer["answer"]
        except Exception:
            pass

        # --- Build search_results from annotation URLs ---
        seen_urls: set[str] = set()
        search_results: list[Search_Result] = []

        for item in response.output:
            if hasattr(item, "content"):
                for content_block in item.content:
                    for ann in getattr(content_block, "annotations", None) or []:
                        if getattr(ann, "type", None) == "url_citation":
                            url = getattr(ann, "url", "") or ""
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                meta_title, meta_date = _fetch_page_meta(url) if url else (None, None)
                                
                                final_title = getattr(ann, "title", None) or meta_title
                                if final_title is not None:
                                    search_results.append(Search_Result(
                                        url=url,
                                        title=final_title,
                                        published_date=meta_date
                                    ))

        # Add any url_list entries not already captured by native annotations
        if structured:
            for url in structured.url_list:
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    meta_title, meta_date = _fetch_page_meta(url)
                    if meta_title is not None:
                        search_results.append(Search_Result(
                            url=url,
                            title=meta_title,
                            published_date=meta_date
                        ))

        # --- Fallback: Extract any remaining URLs via Regex from the raw output ---
        found_urls = re.findall(r'(https?://[^\s)\]"\'`]+)', str(raw))
        for url in found_urls:
            if url and url not in seen_urls:
                seen_urls.add(url)
                meta_title, meta_date = _fetch_page_meta(url)
                search_results.append(Search_Result(
                    url=url,
                    title=meta_title,
                    published_date=meta_date
                ))

        # --- Usage ---
        raw_usage = raw.get("usage", {}) or {}
        usage = Usage(
            prompt_tokens=raw_usage.get("input_tokens", 0),
            completion_tokens=raw_usage.get("output_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
            total_cost=0.0,
        )

        return StandardResponse(
            **base,
            search_results=search_results,
            usage=usage,
            latency_ms=latency_ms,
            raw=raw,
        )
    except Exception as exc:
        return StandardResponse(**base, error=str(exc), raw={})

