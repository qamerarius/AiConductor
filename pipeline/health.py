"""
pipeline/health.py
Health checking for all pipeline dependent services.
"""

import logging
import time

import httpx
from qdrant_client import QdrantClient

from pipeline.models import HealthResponse, ServiceHealth


class HealthChecker:

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger

    async def check_all(self) -> HealthResponse:
        checks = [
            await self._check_jeeves_ollama(),
            await self._check_embedding_ollama(),
            await self._check_qdrant(),
            await self._check_tts(),
        ]

        critical = [c for c in checks if c.name in ("jeeves_ollama", "qdrant", "embedding_ollama")]
        all_critical_healthy = all(c.status == "healthy" for c in critical)

        return HealthResponse(
            status="healthy" if all_critical_healthy else "degraded",
            services=checks
        )

    async def _check_jeeves_ollama(self) -> ServiceHealth:
        cfg = self.config["services"]["jeeves_ollama"]
        url = f"http://{cfg['host']}:{cfg['port']}/api/tags"
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(url)
                latency_ms = (time.time() - start) * 1000
                if response.status_code == 200:
                    model_count = len(response.json().get("models", []))
                    return ServiceHealth(name="jeeves_ollama", status="healthy",
                                        latency_ms=round(latency_ms, 2),
                                        detail=f"{model_count} model(s) available")
                return ServiceHealth(name="jeeves_ollama", status="unhealthy",
                                     latency_ms=round(latency_ms, 2),
                                     detail=f"HTTP {response.status_code}")
        except Exception as e:
            return ServiceHealth(name="jeeves_ollama", status="unhealthy",
                                 latency_ms=round((time.time() - start) * 1000, 2),
                                 detail=str(e))

    async def _check_embedding_ollama(self) -> ServiceHealth:
        cfg = self.config["services"]["embedding_ollama"]
        url = f"http://{cfg['host']}:{cfg['port']}/api/tags"
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(url)
                latency_ms = (time.time() - start) * 1000
                status = "healthy" if response.status_code == 200 else "unhealthy"
                return ServiceHealth(name="embedding_ollama", status=status,
                                     latency_ms=round(latency_ms, 2),
                                     detail=f"Model: {cfg['model']}")
        except Exception as e:
            return ServiceHealth(name="embedding_ollama", status="unhealthy",
                                 latency_ms=round((time.time() - start) * 1000, 2),
                                 detail=str(e))

    async def _check_qdrant(self) -> ServiceHealth:
        cfg = self.config["services"]["qdrant"]
        start = time.time()
        try:
            client = QdrantClient(host=cfg["host"], port=cfg["port"], timeout=5)
            collections = client.get_collections()
            latency_ms = (time.time() - start) * 1000
            collection_names = [c.name for c in collections.collections]
            target_exists = cfg["collection"] in collection_names
            return ServiceHealth(
                name="qdrant", status="healthy",
                latency_ms=round(latency_ms, 2),
                detail=f"Collection '{cfg['collection']}' {'ready' if target_exists else 'not yet created'}"
            )
        except Exception as e:
            return ServiceHealth(name="qdrant", status="unhealthy",
                                 latency_ms=round((time.time() - start) * 1000, 2),
                                 detail=str(e))

    async def _check_tts(self) -> ServiceHealth:
        cfg = self.config["services"].get("tts", {})
        if not cfg:
            return ServiceHealth(name="tts", status="not_configured",
                                 detail="TTS not configured")
        url = f"http://{cfg['host']}:{cfg['port']}/health"
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(url)
                latency_ms = (time.time() - start) * 1000
                return ServiceHealth(
                    name="tts",
                    status="healthy" if response.status_code == 200 else "degraded",
                    latency_ms=round(latency_ms, 2),
                    detail=f"Voice: {cfg.get('voice', 'default')}"
                )
        except Exception as e:
            return ServiceHealth(name="tts", status="degraded",
                                 latency_ms=round((time.time() - start) * 1000, 2),
                                 detail=f"Unavailable: {str(e)}")
