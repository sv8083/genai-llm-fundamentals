import json

import httpx

from core.settings import settings


class OllamaLLMClient:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model

    def generate(self, prompt: str):
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )
        response.raise_for_status()

        payload = response.json()
        content = payload.get("response")
        if not content:
            raise ValueError("Ollama returned an empty response")

        try:
            return {"response": json.loads(content), "raw": content}
        except json.JSONDecodeError:
            return {"response": None, "raw": content}


def get_llm_client() -> OllamaLLMClient:
    return OllamaLLMClient()
