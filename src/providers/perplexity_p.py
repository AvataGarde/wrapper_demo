from __future__ import annotations
import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from perplexity import Perplexity
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.schema import Usage, Search_Result, StandardResponse
from src.prompts import load_system_prompt

load_dotenv()

MODEL = "sonar-pro"  # override via PERPLEXITY_MODEL env var

# Models that require streaming per Perplexity API spec
_STREAMING_MODELS = {"sonar-pro"}


class PerplexitySchema(BaseModel):
    query: str
    answer: str
    source_selection_justification: str
    location: str
    copyright_subject_matter: str
    social_media_use: str
    fair_dealing: str
    licensing: str


def query(prompt: dict, attempt_no: int = 1) -> StandardResponse:
    model = os.environ.get("PERPLEXITY_MODEL", MODEL)
    base = dict(
        provider="perplexity",
        model=model,
        prompt_id=prompt["id"],
        prompt_text=prompt["text"],
        framing=prompt.get("framing"),
        attempt_no=attempt_no,
    )
    try:
        client = Perplexity(api_key=os.environ["PERPLEXITY_API_KEY"])
        system_prompt = load_system_prompt("perplexity")
        needs_stream = model in _STREAMING_MODELS

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt["text"]},
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "audit_response",
                "schema": {
                    **PerplexitySchema.model_json_schema(),
                    "required": list(PerplexitySchema.model_fields.keys()),
                    "additionalProperties": False,
                },
            },
        }
        web_search_options = {"search_type": "pro", "max_results": 15}
        reasoning_effort = "medium"

        t0 = time.perf_counter()

        if needs_stream:
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                response_format=response_format,
                web_search_options=web_search_options,
                reasoning_effort=reasoning_effort,
            )

            content_parts: list[str] = []
            raw_steps: list[dict] = []
            search_results_raw: list[dict] = []
            usage_raw: dict = {}

            for chunk in stream:
                d = chunk.model_dump()
                choices = d.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}

                if delta.get("content"):
                    content_parts.append(delta["content"])

                steps = delta.get("reasoning_steps")
                if steps:
                    raw_steps.extend(steps)

                if d.get("search_results"):
                    search_results_raw = d["search_results"]
                if d.get("usage"):
                    usage_raw = d["usage"]

            content = "".join(content_parts)
            raw = {
                "steps": raw_steps,
                "search_results": search_results_raw,
                "usage": usage_raw,
            }

        else:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                web_search_options=web_search_options,
                reasoning_effort=reasoning_effort,
            )
            content = response.choices[0].message.content or ""
            raw = response.model_dump()
            search_results_raw = raw.get("search_results") or []
            usage_raw = raw.get("usage") or {}
            raw_steps = []

        latency_ms = int((time.perf_counter() - t0) * 1000)

        # --- Parse structured JSON ---
        structured: PerplexitySchema | None = None
        text = content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        structured = PerplexitySchema.model_validate_json(text.strip())

        if structured is not None:
            base.update(structured.model_dump(exclude={"query", "answer"}))
            base["answer"] = structured.answer
        else:
            base["answer"] = content

        # --- reasoning_steps: join thought strings from each step ---
        thoughts = [s["thought"] for s in raw_steps if s.get("thought")]
        reasoning_steps = "\n\n".join(thoughts) if thoughts else None

        # --- search_results ---
        all_results_raw: list[dict] = list(search_results_raw)
        for step in raw_steps:
            ws = step.get("web_search") or {}
            for sr in ws.get("search_results") or []:
                all_results_raw.append(sr)

        seen_urls: set[str] = set()
        search_results: list[Search_Result] = []
        for sr in all_results_raw:
            url = sr.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            search_results.append(Search_Result(
                url=url,
                title=sr.get("title"),
                snippet=sr.get("snippet"),
                published_date=sr.get("date"),
            ))

        # --- Usage ---
        cost_raw = (usage_raw.get("cost") or {}) if isinstance(usage_raw, dict) else {}
        usage = Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
            total_cost=float(cost_raw.get("total_cost", 0.0)),
        )

        return StandardResponse(
            **base,
            reasoning_steps=reasoning_steps,
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
        "text": "What do the results of the May 2026 local and mayoral elections mean for the UK now?",
        "framing": "left",
    }))
