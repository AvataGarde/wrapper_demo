"""Domain registry loader and URL categorizer.

Usage:
    from src.domains import categorize, get_metadata, extract_domain
    
    categorize("https://www.theguardian.com/politics/...")
    # → "mainstream-news-uk"
    
    get_metadata("https://www.theguardian.com/politics/...")
    # → {"category": "mainstream-news-uk", "country": "GB",
    #    "leaning": "centre-left", "outlet_type": "national"}

Static lookup only — no LLM call, no network access. Registry lives in
``domains.yaml`` at the project root and is hot-reloadable via
``reload_registry()``.

Used by ``scripts/analyze.py`` to produce per-provider source-category
distributions without re-querying any LLM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

_REGISTRY_PATH = Path(__file__).parent.parent / "domains.yaml"
_registry_cache: dict[str, Any] | None = None


def _load_registry() -> dict[str, Any]:
    global _registry_cache
    if _registry_cache is None:
        if not _REGISTRY_PATH.exists():
            # Fail soft: empty registry → everything classified as "other"
            _registry_cache = {"domains": {}, "suffix_rules": [], "categories": {}}
        else:
            with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
                _registry_cache = yaml.safe_load(f) or {}
    return _registry_cache


def extract_domain(url_or_domain: str) -> str:
    """Return bare domain (lowercase, no www, no scheme, no port, no path)."""
    if not url_or_domain:
        return ""
    s = url_or_domain.strip().lower()
    # If it doesn't contain :// or /, treat as bare domain
    if "://" not in s and "/" not in s and ":" not in s:
        return s[4:] if s.startswith("www.") else s
    try:
        parsed = urlparse(s if "://" in s else f"https://{s}")
        netloc = parsed.netloc
        if not netloc and parsed.path:
            netloc = parsed.path.split("/")[0]
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return s


def get_metadata(url_or_domain: str) -> dict[str, Any]:
    """Look up metadata for a URL or bare domain.
    
    Match order:
        1. Exact match against registry
        2. Parent-domain match (e.g. news.example.com → example.com)
        3. Suffix rules (.gov.uk, wikipedia.org, etc.)
        4. Fallback to {"category": "other"}
    """
    registry = _load_registry()
    domain = extract_domain(url_or_domain)
    
    domains = registry.get("domains", {}) or {}
    
    # 1. Exact match
    if domain in domains:
        return dict(domains[domain])
    
    # 2. Parent domain match (strip subdomains progressively)
    parts = domain.split(".")
    for i in range(1, len(parts) - 1):
        parent = ".".join(parts[i:])
        if parent in domains:
            meta = dict(domains[parent])
            meta["_matched_via"] = f"parent:{parent}"
            return meta
    
    # 3. Suffix rules
    for rule in registry.get("suffix_rules", []) or []:
        suffix = rule.get("suffix", "")
        if suffix and domain.endswith(suffix):
            meta = {k: v for k, v in rule.items() if k != "suffix"}
            meta["_matched_via"] = f"suffix:{suffix}"
            return meta
    
    # 4. Fallback
    return {"category": "other"}


def categorize(url_or_domain: str) -> str:
    """Return just the category string for a URL or bare domain."""
    return get_metadata(url_or_domain).get("category", "other")


def all_categories() -> dict[str, str]:
    """Return mapping of category name → description from the registry."""
    registry = _load_registry()
    return {
        name: (info or {}).get("description", "")
        for name, info in (registry.get("categories", {}) or {}).items()
    }


def list_unknown_domains(urls: list[str]) -> list[tuple[str, int]]:
    """For each input URL, classify it. Return (domain, count) pairs that
    fell through to 'other'. Sorted by frequency descending. Useful for
    triage when adding new entries to domains.yaml."""
    from collections import Counter
    unknown: Counter[str] = Counter()
    for url in urls:
        if not url:
            continue
        meta = get_metadata(url)
        if meta.get("category") == "other":
            dom = extract_domain(url)
            if dom:
                unknown[dom] += 1
    return unknown.most_common()


def reload_registry() -> None:
    """Force reload from disk. Call after editing domains.yaml in the same
    Python session (e.g. notebook)."""
    global _registry_cache
    _registry_cache = None


if __name__ == "__main__":
    # Quick smoke test
    test_urls = [
        "https://www.theguardian.com/politics/article",
        "https://lewisham.gov.uk/some/page",
        "https://en.wikipedia.org/wiki/Foo",
        "https://www.unknownsite-xyz.com/page",
        "bbc.co.uk",
    ]
    for u in test_urls:
        meta = get_metadata(u)
        print(f"{u}\n  → {meta}\n")