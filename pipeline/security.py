"""
pipeline/security.py
Security middleware — API key authentication and rate limiting.
"""

import logging
import os
import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

TRUSTED_IPS = {"192.168.2.5","192.168.0.100"}
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json"}


class SecurityMiddleware(BaseHTTPMiddleware):

    def __init__(self, app, config: dict, logger: logging.Logger):
        super().__init__(app)
        self.logger = logger
        self.config = config
        self.api_key = os.environ.get("PIPELINE_API_KEY", "")

        if not self.api_key:
            self.logger.warning("PIPELINE_API_KEY not set — authentication disabled")

        self.header_name = config.get("security", {}).get("api_key_header", "X-API-Key").lower()

    async def dispatch(self, request: Request, call_next: Callable):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        if not self.api_key:
            return await call_next(request)

        provided_key = request.headers.get(self.header_name, "")

        if not provided_key:
            self.logger.warning(f"Missing API key | IP: {request.client.host} | Path: {request.url.path}")
            return JSONResponse(status_code=401, content={"error": "API key required"})

        if provided_key != self.api_key:
            self.logger.warning(f"Invalid API key | IP: {request.client.host} | Path: {request.url.path}")
            return JSONResponse(status_code=403, content={"error": "Invalid API key"})

        return await call_next(request)


class RateLimiter:

    def __init__(self):
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def check(self, client_ip: str, config: dict):
        rate_limit_str = config.get("pipeline", {}).get("rate_limit", "30/minute")
        limit, window_seconds = self._parse_rate_limit(rate_limit_str)

        now = time.time()
        window_start = now - window_seconds

        self._requests[client_ip] = [
            ts for ts in self._requests[client_ip] if ts > window_start
        ]

        if len(self._requests[client_ip]) >= limit:
            raise HTTPException(
                status_code=429,
                detail={"error": "Rate limit exceeded", "limit": limit}
            )

        self._requests[client_ip].append(now)

    def _parse_rate_limit(self, rate_limit: str) -> tuple[int, int]:
        parts = rate_limit.split("/")
        if len(parts) != 2:
            return 30, 60

        count = int(parts[0].strip())
        windows = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
        window_seconds = windows.get(parts[1].strip().lower(), 60)
        return count, window_seconds
