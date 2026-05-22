"""OpenAI adapter — uses Responses API with web_search tool."""
from __future__ import annotations
import re
import time
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.schema import Search_Result, Usage, StandardResponse, normalize_url
from src.prompts import load_system_prompt

load_dotenv()

MODEL = "gpt-5.4"  # override via OPENAI_MODEL env var

class OpenAISchema(BaseModel):
    query: str = ""
    answer: str = ""
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
            tools=[{"type": "web_search","search_context_size": "low"}],
            include=["web_search_call.action.sources"],
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

        # remove ```json``` fence and unwrap nested answer
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
        searched_seen: set[str] = set()
        searched_sources: list[Search_Result] = []

        for item in response.output:
            if getattr(item, "type", None) != "web_search_call":
                continue
            action = getattr(item, "action", None)
            sources = getattr(action, "sources", None) or []
            for src in sources:
                url = getattr(src, "url", None) or (src.get("url") if isinstance(src, dict) else None)
                if not url:
                    continue
                norm = normalize_url(url)
                if norm in searched_seen:
                    continue
                searched_seen.add(norm)
                title = getattr(src, "title", None) or (src.get("title") if isinstance(src, dict) else None)
                searched_sources.append(Search_Result(url=url, title=title))

        # --- cited_sources: from message.content[*].annotations[url_citation] ---
        cited_seen: set[str] = set()
        cited_sources: list[Search_Result] = []

        for item in response.output:
            if getattr(item, "type", None) != "message":
                continue
            for content_block in getattr(item, "content", None) or []:
                for ann in getattr(content_block, "annotations", None) or []:
                    if getattr(ann, "type", None) != "url_citation":
                        continue
                    url = normalize_url(getattr(ann, "url", "") or "")
                    if not url or url in cited_seen:
                        continue
                    cited_seen.add(url)
                    cited_sources.append(Search_Result(
                        url=url,
                        title=getattr(ann, "title", None),
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
            cited_sources=cited_sources,
            searched_sources=searched_sources,
            usage=usage,
            latency_ms=latency_ms,
            raw=raw,
        )
    except Exception as exc:
        return StandardResponse(**base, error=str(exc), raw={})

