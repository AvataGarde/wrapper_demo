"""Anthropic adapter — Messages API with web_search tool."""
from __future__ import annotations
import re
import time
import os
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel

from src.schema import Search_Result, Usage, StandardResponse
from src.prompts import load_system_prompt

load_dotenv()

MODEL = "claude-opus-4-5"  # override via ANTHROPIC_MODEL env var


class AnthropicSchema(BaseModel):
    query: str = ""
    url_list: list[str] = []
    source_selection_justification: str = ""
    location: str = ""
    copyright_subject_matter: str = ""
    social_media_use: str = ""
    fair_dealing: str = ""
    licensing: str = ""


def _serialize_content_block(block) -> dict:
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return vars(block) if hasattr(block, "__dict__") else str(block)


def _split_response(text: str) -> tuple[str, AnthropicSchema | None]:
    """Split model output into (answer_prose, structured). Returns (full_text, None) on failure."""
    text = text.strip()

    # Fenced JSON block
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            structured = AnthropicSchema.model_validate_json(fence.group(1))
            return text[:fence.start()].strip(), structured
        except Exception:
            pass

    # Bare JSON block at end of text
    brace = re.search(r"\n\s*(\{[\s\S]*\})\s*$", text)
    if brace:
        try:
            structured = AnthropicSchema.model_validate_json(brace.group(1))
            return text[:brace.start()].strip(), structured
        except Exception:
            pass

    # Whole text is JSON
    try:
        structured = AnthropicSchema.model_validate_json(text)
        return "", structured
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
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt["text"]}],
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
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

        # --- Collect text and native citations from content blocks ---
        text_parts: list[str] = []
        seen_urls: set[str] = set()
        search_results: list[Search_Result] = []

        for block in response.content:
            block_type = getattr(block, "type", None)

            if block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")

                # cited_text is Anthropic's unique signal — store as snippet
                for cit in getattr(block, "citations", None) or []:
                    if getattr(cit, "type", None) == "web_search_result_location":
                        url = getattr(cit, "url", "") or ""
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            search_results.append(Search_Result(
                                url=url,
                                title=getattr(cit, "title", None),
                                snippet=getattr(cit, "cited_text", None),
                            ))

            elif block_type == "tool_result":
                for item in getattr(block, "content", None) or []:
                    if isinstance(item, dict):
                        url = item.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            search_results.append(Search_Result(
                                url=url,
                                title=item.get("title"),
                                snippet=item.get("snippet"),
                            ))

        full_text = "\n".join(text_parts).strip()

        # --- Parse structured JSON from assembled text ---
        answer, structured = _split_response(full_text)
        if structured is not None:
            base.update(structured)
        base["answer"] = answer or full_text
        base["reasoning_steps"] = raw.get("reasoning_steps")

        # Add any url_list entries not already captured natively
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
            total_tokens=(raw_usage.get("input_tokens", 0) + raw_usage.get("output_tokens", 0)),
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
