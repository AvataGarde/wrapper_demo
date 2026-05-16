from __future__ import annotations
import yaml
from pathlib import Path


_cache: list[dict] | None = None
_system_cache: dict | None = None


def _project_root() -> Path:
    return Path(__file__).parent.parent



def load_prompts(path: str = "prompts.yaml") -> list[dict]:
    """Load prompts from YAML file if not already loaded"""
    global _cache
    if _cache is None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = _project_root() / path
        with open(resolved, "r", encoding="utf-8") as f:
            _cache = yaml.safe_load(f)
    return _cache


def load_system_prompt(provider: str | None = None, path: str = "system_prompts.yaml") -> str:
    """Return the system prompt string for a given provider (falls back to default)."""
    global _system_cache
    if _system_cache is None:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = _project_root() / path
        with open(resolved, "r", encoding="utf-8") as f:
            _system_cache = yaml.safe_load(f)
    data = _system_cache
    if provider and "providers" in data and provider in data["providers"]:
        return data["providers"][provider].strip()
    return data.get("default", "").strip()


def get_prompt(prompt_id: str, path: str = "prompts.yaml") -> dict:
    """Get a specific prompt by ID, ID here is a topic for now, e.g. iran-uk-war-neutral"""
    prompts = load_prompts(path)
    for p in prompts:
        if p["id"] == prompt_id:
            return p
    raise KeyError(f"Prompt '{prompt_id}' not found")


def filter_prompts(
    topic: str | None = None,
    framing: str | None = None,
    path: str = "prompts.yaml",
) -> list[dict]:
    prompts = load_prompts(path)
    result = prompts
    if topic is not None:
        result = [p for p in result if p.get("topic") == topic]
    if framing is not None:
        result = [p for p in result if p.get("framing") == framing]
    return result

if __name__ == "__main__":
    # test 
    print(load_prompts())

    # test get_prompt
    print()
    print(get_prompt("iran-uk-war-neutral"))

    # test filter_prompts
    print()
    print(filter_prompts(topic="iran-israel-us-war", framing="left"))