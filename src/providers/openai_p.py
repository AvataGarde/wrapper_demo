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
    url_list: list[str] = []
    source_selection_justification: str = ""
    location: str = ""
    copyright_subject_matter: str = ""
    social_media_use: str = ""
    fair_dealing: str = ""
    licensing: str = ""


def _split_response(text: str) -> tuple[str, OpenAISchema | None]:
    """Split model output into (answer_prose, structured). Returns (full_text, None) on failure."""
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
        return "", structured
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
            tools=[{"type": "web_search_preview"}],
            instructions=system_prompt,
            input=prompt["text"],
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        raw: dict = response.model_dump()
        full_text: str = response.output_text or ""

        # --- Parse structured JSON from model output ---
        answer, structured = _split_response(full_text)
        if structured is not None:
            base.update(structured)
        base["answer"] = answer or full_text
        base["reasoning_steps"] = raw.get("reasoning_steps")

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
                                search_results.append(Search_Result(
                                    url=url,
                                    title=getattr(ann, "title", None),
                                ))

        # Add any url_list entries not already captured by native annotations
        if structured:
            for url in structured.url_list:
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    search_results.append(Search_Result(url=url))

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
