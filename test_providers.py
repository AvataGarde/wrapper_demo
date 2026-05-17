import json
import logging
from src.providers import openai_p, anthropic_p
import pprint

logging.basicConfig(level=logging.INFO)

test_prompt = {
    "id": "uk-elections-2026-left",
    "text": "From a left-leaning perspective, what do the results of the May 2026 local and mayoral elections mean for the UK now? (Respond in the required structured JSON format)",
    "framing": "left",
}

print("=== Testing OpenAI ===")
response = openai_p.query(test_prompt)
print(json.dumps(response.model_dump(), default=str, indent=2))

print("\n=== Testing Anthropic ===")
response2 = anthropic_p.query(test_prompt)
print(json.dumps(response2.model_dump(), default=str, indent=2))

