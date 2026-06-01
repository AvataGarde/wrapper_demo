from src.schema import Search_Result, StandardResponse, Usage
from src.storage import build_db_records


def test_build_db_records_maps_standard_response_into_relational_rows():
    response = StandardResponse(
        provider="anthropic",
        model="claude-test",
        prompt_id="uk-elections-2026-neutral",
        prompt_text="What do the results mean?",
        framing="neutral",
        attempt_no=3,
        timestamp="2026-05-17T10:30:00+00:00",
        answer="Plain answer",
        source_selection_justification="Why these sources",
        location="UK",
        copyright_subject_matter="News reporting",
        social_media_use="None",
        fair_dealing="Possible",
        licensing="Unknown",
        reasoning_steps="step one\nstep two",
        searched_sources=[
            Search_Result(
                url="https://example.com/a",
                resolved_url="https://example.com/final-a",
                title="Example A",
                snippet="Snippet A",
                published_date="2026-05-10",
            ),
            Search_Result(
                url="https://example.com/b",
                title="Example B",
            ),
        ],
        cited_sources=[
            Search_Result(
                url="https://example.com/a",
                resolved_url="https://example.com/final-a",
                title="Example A",
                snippet="Snippet A",
                published_date="2026-05-10",
            ),
        ],
        usage=Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            total_cost=0.12,
        ),
        latency_ms=4321,
        raw={"id": "raw-1"},
    )

    records = build_db_records(response)

    assert records["query"] == {
        "id": "uk-elections-2026-neutral",
        "topic": "uk-elections-2026",
        "framing": "neutral",
        "text": "What do the results mean?",
    }
    assert records["run"] == {
        "query_id": "uk-elections-2026-neutral",
        "provider": "anthropic",
        "model": "claude-test",
        "attempt_no": 3,
        "timestamp": "2026-05-17T10:30:00+00:00",
        "latency_ms": 4321,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_cost": 0.12,
        "error": None,
    }
    assert records["answer"] == {
        "answer": "Plain answer",
        "source_selection_justification": "Why these sources",
        "location": "UK",
        "copyright_subject_matter": "News reporting",
        "social_media_use": "None",
        "fair_dealing": "Possible",
        "licensing": "Unknown",
    }
    assert records["reasoning_steps"] == [
        {"step_no": 1, "content": "step one"},
        {"step_no": 2, "content": "step two"},
    ]
    assert records["searched_sources"] == [
        {
            "rank": 1,
            "url": "https://example.com/a",
            "resolved_url": "https://example.com/final-a",
            "title": "Example A",
            "snippet": "Snippet A",
            "published_date": "2026-05-10",
        },
        {
            "rank": 2,
            "url": "https://example.com/b",
            "resolved_url": None,
            "title": "Example B",
            "snippet": None,
            "published_date": None,
        },
    ]
    assert records["cited_sources"] == [
        {
            "rank": 1,
            "url": "https://example.com/a",
            "resolved_url": "https://example.com/final-a",
            "title": "Example A",
            "snippet": "Snippet A",
            "published_date": "2026-05-10",
        },
    ]
    assert records["raw_response"] == {"id": "raw-1"}
