import json

import httpx

from core.settings import settings


class OllamaLLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model = model or settings.ollama_model
        self.temperature = temperature or 0.7
        self.top_p = top_p or 0.9

    def generate(self, prompt: str, temperature: float | None = None, top_p: float | None = None):
        temp = temperature if temperature is not None else self.temperature
        top_p_val = top_p if top_p is not None else self.top_p
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temp,
                "top_p": top_p_val,
            },
        }
        
        response = httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
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


def get_llm_client(base_url: str | None = None, model: str | None = None) -> OllamaLLMClient:
    if model.lower().startswith("openai"):
        return None
    return OllamaLLMClient(base_url=base_url, model=model)
