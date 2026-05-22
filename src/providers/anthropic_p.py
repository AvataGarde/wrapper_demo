"""Anthropic adapter — Messages API with web_search tool."""
import base64
import re
import time
import os
import re
import json
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel

from src.schema import Search_Result, Usage, StandardResponse, normalize_url
from src.prompts import load_system_prompt

load_dotenv()

MODEL = "claude-opus-4-5"  # override via ANTHROPIC_MODEL env var



class AnthropicSchema(BaseModel):
    query: str = ""
    answer: str = ""
    #url_list: list[str] = []
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


def _serialize_content_block(block) -> dict:
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return vars(block) if hasattr(block, "__dict__") else str(block)

def _build_search_index(response_content) -> list[list[dict]]:
    search_results_by_call: list[list[dict]] = []
    
    for block in response_content:
        btype = getattr(block, "type", None)
        
        if btype == "server_tool_use":
            search_results_by_call.append([])
        
        elif btype == "web_search_tool_result":

            items = getattr(block, "content", None) or []
            if search_results_by_call:
                for item in items:
                    url = getattr(item, "url", None)
                    title = getattr(item, "title", None)
                    page_age = getattr(item, "page_age", None)
                    if url:
                        search_results_by_call[-1].append({
                            "url": url,
                            "title": title,
                            "page_age": page_age,
                        })
    
    return search_results_by_call


_CITE_URL_PATTERN = re.compile(
    r'<cite\s+url\s*=\s*\\?"([^"\\]+)\\?"',
    re.IGNORECASE,
)

def _extract_cited_by_url(
    answer_text: str,
    searched_sources: list[Search_Result],
) -> tuple[list[Search_Result], list[str]]:
    url_to_sr = {sr.url: sr for sr in searched_sources}
    cited: list[Search_Result] = []
    hallucinated: list[str] = []
    seen: set[str] = set()
    
    for m in _CITE_URL_PATTERN.finditer(answer_text):
        url_str = m.group(1)
        for url in url_str.split(","):
            url = normalize_url(url.strip())
            if not url or url in seen:
                continue
            seen.add(url)
            if url in url_to_sr:
                cited.append(url_to_sr[url])
            else:
                hallucinated.append(url)
    
    return cited, hallucinated

    
def _split_response(text: str) -> tuple[str, AnthropicSchema | None]:
    """Split model output into (answer_prose, structured). Returns (full_text, None) on failure."""
    text = text.strip()

    # Fenced JSON block
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            structured = AnthropicSchema.model_validate_json(fence.group(1))
            return structured.answer or text[:fence.start()].strip(), structured
        except Exception:
            pass

    # Bare JSON block at end of text
    brace = re.search(r"\n\s*(\{[\s\S]*\})\s*$", text)
    if brace:
        try:
            structured = AnthropicSchema.model_validate_json(brace.group(1))
            return structured.answer or text[:brace.start()].strip(), structured
        except Exception:
            pass

    # Whole text is JSON
    try:
        structured = AnthropicSchema.model_validate_json(text)
        return structured.answer or text, structured
    except Exception:
        pass

    return text, None


def query(prompt: dict, attempt_no: int = 1) -> StandardResponse:
    model = os.environ.get("ANTHROPIC_MODEL", MODEL)
    base = dict(
        provider="anthropic",
        model=model,
        prompt_id=prompt["id"],
        prompt_text=prompt["text"],
        framing=prompt.get("framing"),
        attempt_no=attempt_no,
    )
    try:
        client = anthropic.Anthropic()
        system_prompt = load_system_prompt("anthropic")
        t0 = time.perf_counter()
        
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt["text"]}],
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": 2
                }
            ],
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        
        raw: dict = {
            "id": response.id,
            "type": response.type,
            "role": response.role,
            "model": response.model,
            "stop_reason": response.stop_reason,
            "stop_sequence": response.stop_sequence,
            "usage": response.usage.model_dump() if response.usage else None,
            "content": [_serialize_content_block(b) for b in response.content],
        }

        search_index = _build_search_index(response.content)
        
        # --- Collect text and native citations from content blocks ---
        text_parts: list[str] = []
        searched_sources: list[Search_Result] = []
        searched_seen: set[str] = set()
        for call_results in search_index:
            for item in call_results:
                url = item["url"]
                if not url:
                    continue
                norm = normalize_url(url)
                if norm in searched_seen:
                    continue
                searched_seen.add(norm)
                searched_sources.append(Search_Result(
                        url=url,
                        title=item["title"],
                        published_date=item["page_age"],
                    ))
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", "") or "")
        full_text = "\n".join(text_parts).strip()


        # --- Parse structured JSON from assembled text ---
        answer, structured = _split_response(full_text)
        if structured is not None:
            base.update(structured.model_dump(exclude={"query", "answer"}))
        base["answer"] = answer or full_text
        base["reasoning_steps"] = raw.get("reasoning_steps")
        
        # Replace triple backtick JSON code blocks around the answer if present
        fence = re.search(r"```json\s*(\{.*?\})\s*```", base["answer"], re.DOTALL)
        if fence:
            base["answer"] = fence.group(1).strip()

        
        try:
            parsed_answer = json.loads(base["answer"])
            # if parse succeeds and has 'answer' field, we unpack it
            if isinstance(parsed_answer, dict) and "answer" in parsed_answer:
                base.update({k: v for k, v in parsed_answer.items() if k not in ["query", "answer"]})
                base["answer"] = parsed_answer["answer"]
        except Exception:
            pass
        
        cited_sources, hallucinated_urls = _extract_cited_by_url(full_text, searched_sources)
        # --- Usage ---
        raw_usage = raw.get("usage", {}) or {}
        usage = Usage(
            prompt_tokens=raw_usage.get("input_tokens", 0),
            completion_tokens=raw_usage.get("output_tokens", 0),
            total_tokens=(raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0)),
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
