"""
pipeline/search.py
Web search via SearXNG for the Jeeves pipeline.
"""

import logging
import httpx
from typing import Optional


class SearchEngine:

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        search_cfg = config.get("services", {}).get("searxng", {})
        self.base_url = f"http://{search_cfg.get('host', 'searxng.local')}:{search_cfg.get('port', 8080)}"
        self.enabled = search_cfg.get("enabled", False)
        self.result_count = search_cfg.get("results", 3)

    def should_search(self, message: str) -> bool:
        # Ignore very short messages — likely generated tasks or tags
        if len(message.strip()) < 20:
            return False

        # Ignore messages that look like generated titles or tags
        skip_patterns = [
            "generate", "summarize", "title", "tag",
            "follow up", "followup", "suggest"
        ]
        message_lower = message.lower()
        if any(pattern in message_lower for pattern in skip_patterns):
            return False

        # Simple heuristic to determine if a web search is warranted.
        # Checks for keywords suggesting current information is needed.

        triggers = [
            "current", "today", "latest", "recent", "now",
            "news", "weather", "price", "stock", "score",
            "online", "search", "look up", "find out",
            "what is happening", "right now", "this week",
            "this year", "2026", "update"
        ]
        return any(trigger in message_lower for trigger in triggers)

    async def refine_query(self, message: str, inference_url: str, model: str) -> str:

        # Use the LLM to convert a conversational message into
        # a concise effective search query.

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{inference_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": (
                            f"Convert this message into a concise web search query."
                            f"Return only the search query, "
                            f"nothing else. /no_think\n\nMessage: {message}"
                        ),
                        "stream": False,
                        "options": {"num_predict": 4096}
                    }
                )
                response.raise_for_status()
                data = response.json()
                refined = data.get("thinking").strip()

                # Fall back to original message if refinement returns empty
                if not refined:
                    self.logger.warning("Query refinement returned empty — using original")
                    return message[:256]

                self.logger.info(f"Refined query: '{refined}' from: '{message[:256]}'")
                return refined
        except Exception as e:
            self.logger.error(f"Query refinement error: {str(e)} — using original")
            return message

    async def search(self, query: str, inference_url: str, model: str) -> Optional[str]:
        self.logger.info(f"Search query: {query[:256]}")

        # Perform a web search and return formatted results.
        # Returns None if search fails or is disabled.

        if not self.enabled:
            return None

        refined_query = await self.refine_query(query, inference_url, model)
        search_query = refined_query[:200]

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.get(
                    f"{self.base_url}/search",
                    params={
                        "q": search_query,
                        "format": "json",
                        "categories": "general"
                    }
                )
                response.raise_for_status()
                data = response.json()

            results = data.get("results", [])[:self.result_count]

            if not results:
                return None

            formatted = "Current web search results '{refined_query}':\n"
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                content = r.get("content", "")[:300]
                formatted += f"{i}. {title}\n{content}\n\n"

            self.logger.info(f"Search returned {len(results)} results for: {query[:50]}")
            return formatted

        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return None
