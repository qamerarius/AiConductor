"""
pipeline/memory.py
Memory management for Jeeves — Qdrant and embedding coordination.
Written for qdrant-client 1.18
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

import httpx
from qdrant_client import QdrantClient, models
from datetime import datetime

class MemoryManager:

    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client: Optional[QdrantClient] = None

        qdrant_cfg = config["services"]["qdrant"]
        self.qdrant_host = qdrant_cfg["host"]
        self.qdrant_port = qdrant_cfg["port"]
        self.collection_name = qdrant_cfg["collection"]

        embedding_cfg = config["services"]["embedding_ollama"]
        self.embedding_host = embedding_cfg["host"]
        self.embedding_port = embedding_cfg["port"]
        self.embedding_model = embedding_cfg["model"]
        self.embedding_url = f"http://{self.embedding_host}:{self.embedding_port}"

        pipeline_cfg = config.get("pipeline", {})
        self.memory_results = pipeline_cfg.get("memory_results", 5)
        self.min_relevance_score = pipeline_cfg.get("min_relevance_score", 0.65)
        self.max_context_tokens = pipeline_cfg.get("max_context_tokens", 2000)

    async def initialise(self):
        self.logger.info(
            f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}"
        )

        self.client = QdrantClient(
            host=self.qdrant_host,
            port=self.qdrant_port,
            timeout=30
        )

        collections = self.client.get_collections()
        self.logger.info(
            f"Qdrant connected — {len(collections.collections)} collections found"
        )

        await self._ensure_collection()

    async def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in existing:
            self.logger.info(f"Creating collection: {self.collection_name}")
            test_vector = await self._embed("test")
            vector_size = len(test_vector)

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            self.logger.info(
                f"Collection created: {self.collection_name} "
                f"(vector size: {vector_size})"
            )
        else:
            self.logger.info(f"Collection exists: {self.collection_name}")

    async def _embed(self, text: str) -> list[float]:
        self.logger.info(
            f"Embedding request | Length: {len(text)} chars | "
            f"Preview: {text[:100]}"
        )
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"{self.embedding_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text}
            )
            if response.status_code != 200:
                self.logger.error(
                    f"Embedding error: HTTP {response.status_code} | "
                    f"Body: {response.text}"
                )
            response.raise_for_status()
            return response.json()["embedding"]

    async def retrieve(self, query: str) -> list[dict]:
        if not self.client:
            return []

        try:
            query_vector = await self._embed(query)

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=self.memory_results,
                score_threshold=self.min_relevance_score,
                with_payload=True
            ).points

            return [
                {
                    "text": r.payload.get("text", ""),
                    "timestamp": r.payload.get("timestamp", ""),
                    "score": r.score,
                    "type": r.payload.get("type", "episodic")
                }
                for r in results
            ]

        except Exception as e:
            self.logger.error(
                f"Memory retrieval error: {str(e)} — continuing without memory"
            )
            return []

    async def store(self, user_message: str, assistant_response: str):
        if not self.client:
            return

        try:
            user_truncated = user_message[:500]
            response_truncated = assistant_response[:3000]
            exchange_text = (
                f"User: {user_truncated}\nJeeves: {response_truncated}"
            )

            vector = await self._embed(exchange_text)

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": exchange_text,
                            "user_message": user_message[:500],
                            "assistant_response": assistant_response[:3000],
                            "timestamp": datetime.now().isoformat(),
                            "type": "episodic"
                        }
                    )
                ]
            )

        except Exception as e:
            self.logger.error(f"Memory storage error: {str(e)}")

    def inject_memory_context(
        self,
        messages: list[dict],
        memories: list[dict],
        search_results: str = None
    ) -> list[dict]:

        memory_text = self._format_memories(memories) if memories else ""
        current_time = datetime.now().strftime("%A, %B %d, %Y at %H:%M")
        content = (
            f"Current date and time: {current_time}\n\n"
            "Your name is Jeeves. You are a knowledgeable and thoughtful "
            "assistant with quiet dignity and dry wit. You have persistent "
            "memory of previous conversations.\n\n"
        )

        if memory_text:
            content += (
                f"Relevant memories from previous conversations:\n"
                f"{memory_text}\n\n"
            )

        if search_results:
            content += (
                f"{search_results}\n"
                "Use the above search results to inform your response where "
                "relevant. Reference them naturally without listing URLs "
                "unless specifically asked.\n\n"
            )

        if memory_text or search_results:
            content += (
                "When memories or search results are relevant, reference "
                "them naturally in your response."
            )

        system_message = {"role": "system", "content": content}

        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = (
                content + "\n\n" + messages[0]["content"]
            )
            return messages
        else:
            return [system_message] + messages

    def build_enriched_prompt(
        self,
        original_prompt: str,
        memories: list[dict]
    ) -> str:
        if not memories:
            return original_prompt

        memory_text = self._format_memories(memories)
        return (
            f"Relevant context from your memory of previous conversations:\n"
            f"{memory_text}\n\n"
            f"Current message:\n{original_prompt}"
        )

    def _format_memories(self, memories: list[dict]) -> str:
        if not memories:
            return ""

        formatted = []
        total_chars = 0
        char_limit = self.max_context_tokens * 4

        for memory in memories:
            text = memory.get("text", "")
            timestamp = memory.get("timestamp", "")

            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    entry = f"[{dt.strftime('%B %d, %Y')}] {text}"
                except ValueError:
                    entry = text
            else:
                entry = text

            if len(entry) > 500:
                entry = entry[:497] + "..."

            if total_chars + len(entry) > char_limit:
                break

            formatted.append(entry)
            total_chars += len(entry)

        return "\n---\n".join(formatted)

    async def store_document(
        self,
        text: str,
        doc_type: str = "knowledge",
        metadata: dict = None
    ):
        if not self.client:
            return

        chunks = self._chunk_text(text)
        self.logger.info(
            f"Storing document as {len(chunks)} chunks | Type: {doc_type}"
        )

        for i, chunk in enumerate(chunks):
            try:
                vector = await self._embed(chunk)

                payload = {
                    "text": chunk,
                    "timestamp": datetime.now().isoformat(),
                    "type": doc_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                if metadata:
                    payload.update(metadata)

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload=payload
                        )
                    ]
                )
                await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error storing chunk {i}: {str(e)}")

        self.logger.info(f"Document stored — {len(chunks)} chunks")

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> list[str]:
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            start = end - overlap

        return chunks

    async def get_collection_stats(self) -> dict:
        if not self.client:
            return {"error": "Client not initialised"}
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": str(info.status)
            }
        except Exception as e:
            return {"error": str(e)}
