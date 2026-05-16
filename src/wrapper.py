from __future__ import annotations
from src.providers import perplexity_p, openai_p, anthropic_p, gemini
from src.schema import StandardResponse

PROVIDERS: dict = {
    "perplexity": perplexity_p.query,
    "openai": openai_p.query,
    "anthropic": anthropic_p.query,
    "gemini": gemini.query,
}


def run_one(prompt: dict, provider: str, attempt_no: int = 1) -> StandardResponse:
    fn = PROVIDERS[provider]
    return fn(prompt, attempt_no=attempt_no)


def run_all(prompt: dict, providers: list[str] | None = None) -> list[StandardResponse]:
    providers = providers or list(PROVIDERS.keys())
    return [run_one(prompt, p) for p in providers]

if __name__ == "__main__":
    query = {"id": "iran-uk-war-neutral", "text": "What is the UK's role in the current conflict between Iran, Israel, and the United States?"}
    print("Provider: perplexity")
    print(run_one(query, "perplexity"))
    print("Provider: gemini")
    print(run_one(query, "gemini"))
    
    
