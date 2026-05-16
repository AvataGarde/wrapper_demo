"""Gemini adapter — google-genai SDK with google_search grounding."""
from __future__ import annotations
import re
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
    url_list: list[str] = []
    source_selection_justification: str = ""
    location: str = ""
    copyright_subject_matter: str = ""
    social_media_use: str = ""
    fair_dealing: str = ""
    licensing: str = ""


def _split_response(text: str) -> tuple[str, GeminiSchema | None]:
    """Split model output into (answer_prose, structured). Returns (full_text, None) on failure."""
    text = text.strip()

    # Fenced JSON block
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            structured = GeminiSchema.model_validate_json(fence.group(1))
            return text[:fence.start()].strip(), structured
        except Exception:
            pass

    # Bare JSON block at end of text
    brace = re.search(r"\n\s*(\{[\s\S]*\})\s*$", text)
    if brace:
        try:
            structured = GeminiSchema.model_validate_json(brace.group(1))
            return text[:brace.start()].strip(), structured
        except Exception:
            pass

    # Whole text is JSON
    try:
        structured = GeminiSchema.model_validate_json(text)
        return "", structured
    except Exception:
        pass

    return text, None


def _safe_dict(obj):
    """Recursively convert Gemini response objects to plain dicts, JSON-safe."""
    import base64
    if obj is None:
        return None
    if isinstance(obj, bytes):
        # thought_signature and similar binary fields → base64 string
        return base64.b64encode(obj).decode("ascii")
    if isinstance(obj, dict):
        return {k: _safe_dict(v) for k, v in obj.items()}
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


def _extract_reasoning(response) -> str | None:
    """Collect text from all thought=True parts across all candidates."""
    parts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", None) is True:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
    return "\n\n".join(parts) if parts else None


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
                tools=[types.Tool(google_search=types.GoogleSearch())],
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True
                ),
            ),
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        raw: dict = _safe_dict(response) or {}
        full_text: str = response.text or ""

        print(raw)
        print()
        # --- Parse structured JSON from model output ---
        answer, structured = _split_response(full_text)
        if structured is not None:
            base.update(structured)
        base["answer"] = answer or full_text
        base["reasoning_steps"] = _extract_reasoning(response)

        # --- Build search_results from grounding_chunks ---
        # Keep redirect URLs as-is (no resolution at demo stage)
        seen_urls: set[str] = set()
        search_results: list[Search_Result] = []

        candidate = response.candidates[0] if response.candidates else None
        grounding_meta = getattr(candidate, "grounding_metadata", None) if candidate else None

        if grounding_meta:
            for chunk in getattr(grounding_meta, "grounding_chunks", None) or []:
                web = getattr(chunk, "web", None)
                if web:
                    url = getattr(web, "uri", "") or ""
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        search_results.append(Search_Result(
                            url=url,
                            resolved_url=_resolve_redirect(url),
                            title=getattr(web, "title", None),
                        ))

        # Add any url_list entries not already captured by grounding
        if structured:
            for url in structured.url_list:
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    search_results.append(Search_Result(url=url))

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
            search_results=search_results,
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