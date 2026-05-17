"""
pipeline/inference.py
Inference engine — communication with Jeeves's Ollama instance.
"""

import logging
import time
from typing import Optional

import httpx

from pipeline.models import OllamaGenerateResponse, OllamaChatResponse, OllamaOptions


class InferenceEngine:

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger

        jeeves_cfg = config["services"]["jeeves_ollama"]
        self.base_url = f"http://{jeeves_cfg['host']}:{jeeves_cfg['port']}"
        self.default_model = jeeves_cfg["model"]
        self.timeout = jeeves_cfg.get("timeout", 120)

    async def generate(self, prompt: str, model: Optional[str] = None, options: Optional[OllamaOptions] = None) -> OllamaGenerateResponse:
        model = model or self.default_model
        start = time.time()

        payload = {"model": model, "prompt": prompt, "stream": False}
        if options:
            payload["options"] = options.dict(exclude_none=True)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

        self.logger.debug(f"Generate complete | {(time.time() - start) * 1000:.0f}ms")

        return OllamaGenerateResponse(
            model=data.get("model", model),
            created_at=data.get("created_at", ""),
            response=data.get("response", ""),
            done=data.get("done", True),
            total_duration=data.get("total_duration"),
            eval_count=data.get("eval_count"),
            eval_duration=data.get("eval_duration"),
        )

    async def chat(self, messages: list[dict], model: Optional[str] = None, options: Optional[OllamaOptions] = None) -> OllamaChatResponse:
        model = model or self.default_model
        start = time.time()

        payload = {"model": model, "messages": messages, "stream": False}
        if options:
            payload["options"] = options.dict(exclude_none=True)

        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        self.logger.debug(f"Chat complete | {(time.time() - start) * 1000:.0f}ms")

        return OllamaChatResponse(
            model=data.get("model", model),
            created_at=data.get("created_at", ""),
            message=data.get("message", {"role": "assistant", "content": ""}),
            done=data.get("done", True),
            total_duration=data.get("total_duration"),
            eval_count=data.get("eval_count"),
            eval_duration=data.get("eval_duration"),
        )

    async def list_models(self) -> dict:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return response.json()

    async def ping(self) -> tuple[bool, float]:
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200, (time.time() - start) * 1000
        except Exception:
            return False, (time.time() - start) * 1000
