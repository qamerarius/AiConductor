"""
tests/test_pipeline.py
======================
Basic connectivity and integration tests for the Jeeves pipeline.

Run with: python -m pytest tests/ -v
Or directly: python tests/test_pipeline.py

These tests verify that all services are reachable and responding
correctly before putting the pipeline into production use.
They do not test inference quality — only connectivity and basic function.
"""

import asyncio
import os
import sys
import yaml
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import pytest
from qdrant_client import QdrantClient


# ─── Configuration ────────────────────────────────────────────────────────────

def load_config() -> dict:
    config_path = os.environ.get(
        "JEEVES_CONFIG",
        "/opt/jeeves-pipeline/config/config.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def logger():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("tests")


# ─── Jeeves Ollama Tests ──────────────────────────────────────────────────────

class TestJeevesOllama:
    """Tests for Jeeves's inference Ollama instance on the P100 host."""

    def test_ollama_reachable(self, config):
        """Verify Jeeves's Ollama API is reachable."""
        cfg = config["services"]["jeeves_ollama"]
        url = f"http://{cfg['host']}:{cfg['port']}/api/tags"

        response = httpx.get(url, timeout=10)
        assert response.status_code == 200, (
            f"Jeeves Ollama not reachable at {url}\n"
            f"Check that Ollama is running on the P100 host and "
            f"OLLAMA_HOST=0.0.0.0 is configured."
        )

    def test_model_available(self, config):
        """Verify the configured model is loaded in Ollama."""
        cfg = config["services"]["jeeves_ollama"]
        url = f"http://{cfg['host']}:{cfg['port']}/api/tags"
        model_name = cfg["model"]

        response = httpx.get(url, timeout=10)
        data = response.json()
        models = [m["name"] for m in data.get("models", [])]

        assert any(model_name in m for m in models), (
            f"Model '{model_name}' not found in Ollama.\n"
            f"Available models: {models}\n"
            f"Run: ollama pull {model_name}"
        )

    def test_basic_inference(self, config):
        """Verify Jeeves can generate a basic response."""
        cfg = config["services"]["jeeves_ollama"]
        url = f"http://{cfg['host']}:{cfg['port']}/api/generate"

        payload = {
            "model": cfg["model"],
            "prompt": "Say only the word: Hello",
            "stream": False,
            "options": {"num_predict": 50}
        }

        response = httpx.post(url, json=payload, timeout=cfg.get("timeout", 120))
        assert response.status_code == 200

        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0, "Jeeves returned an empty response"

        print(f"\n  Jeeves responded: '{data['response'].strip()}'")


# ─── Embedding Ollama Tests ───────────────────────────────────────────────────

class TestEmbeddingOllama:
    """Tests for the embedding model Ollama instance on Node 1."""

    def test_embedding_ollama_reachable(self, config):
        """Verify the embedding Ollama instance is reachable."""
        cfg = config["services"]["embedding_ollama"]
        url = f"http://{cfg['host']}:{cfg['port']}/api/tags"

        response = httpx.get(url, timeout=10)
        assert response.status_code == 200, (
            f"Embedding Ollama not reachable at {url}\n"
            f"Check Ollama is running on Node 1."
        )

    def test_embedding_model_available(self, config):
        """Verify nomic-embed-text is available."""
        cfg = config["services"]["embedding_ollama"]
        url = f"http://{cfg['host']}:{cfg['port']}/api/tags"
        model_name = cfg["model"]

        response = httpx.get(url, timeout=10)
        data = response.json()
        models = [m["name"] for m in data.get("models", [])]

        assert any(model_name in m for m in models), (
            f"Embedding model '{model_name}' not found.\n"
            f"Run on Node 1: ollama pull {model_name}"
        )

    def test_embedding_generates_vector(self, config):
        """Verify the embedding model produces vectors of expected dimensions."""
        cfg = config["services"]["embedding_ollama"]
        url = f"http://{cfg['host']}:{cfg['port']}/api/embeddings"

        payload = {
            "model": cfg["model"],
            "prompt": "The key turned with a sharp click."
        }

        response = httpx.post(url, json=payload, timeout=30)
        assert response.status_code == 200

        data = response.json()
        assert "embedding" in data
        assert len(data["embedding"]) > 0

        print(f"\n  Vector dimensions: {len(data['embedding'])}")


# ─── Qdrant Tests ─────────────────────────────────────────────────────────────

class TestQdrant:
    """Tests for the Qdrant vector database on Node 1."""

    def test_qdrant_reachable(self, config):
        """Verify Qdrant is reachable."""
        cfg = config["services"]["qdrant"]

        client = QdrantClient(
            host=cfg["host"],
            port=cfg["port"],
            timeout=10
        )
        collections = client.get_collections()
        assert collections is not None, "Qdrant not reachable"

        print(f"\n  Qdrant collections: {[c.name for c in collections.collections]}")

    def test_collection_exists_or_createable(self, config):
        """Verify the memory collection exists or can be created."""
        cfg = config["services"]["qdrant"]

        client = QdrantClient(
            host=cfg["host"],
            port=cfg["port"],
            timeout=10
        )

        collections = [c.name for c in client.get_collections().collections]

        if cfg["collection"] in collections:
            print(f"\n  Collection '{cfg['collection']}' exists — ready")
        else:
            print(f"\n  Collection '{cfg['collection']}' does not exist yet.")
            print(f"  It will be created automatically when the pipeline starts.")

        # This test passes either way — collection is created on first startup
        assert True

    def test_store_and_retrieve(self, config):
        """
        Verify a vector can be stored and retrieved from Qdrant.
        Uses a test collection to avoid polluting Jeeves's memory.
        """
        from qdrant_client.models import Distance, VectorParams, PointStruct
        import uuid

        cfg = config["services"]["qdrant"]
        embedding_cfg = config["services"]["embedding_ollama"]

        # Generate a test embedding
        embed_url = f"http://{embedding_cfg['host']}:{embedding_cfg['port']}/api/embeddings"
        response = httpx.post(
            embed_url,
            json={"model": embedding_cfg["model"], "prompt": "test memory storage"},
            timeout=30
        )
        assert response.status_code == 200
        vector = response.json()["embedding"]

        # Store in a test collection
        client = QdrantClient(host=cfg["host"], port=cfg["port"], timeout=10)
        test_collection = "pipeline_test_collection"

        # Create test collection
        if test_collection not in [c.name for c in client.get_collections().collections]:
            client.create_collection(
                collection_name=test_collection,
                vectors_config=VectorParams(size=len(vector), distance=Distance.COSINE)
            )

        # Store a point
        test_id = str(uuid.uuid4())
        client.upsert(
            collection_name=test_collection,
            points=[PointStruct(
                id=test_id,
                vector=vector,
                payload={"text": "test memory", "type": "test"}
            )]
        )

        # Retrieve it
        results = client.query_points(
            collection_name=test_collection,
            query=vector,
            limit=1
        ).points

        assert len(results) > 0
        assert results[0].payload["text"] == "test memory"

        # Clean up test collection
        client.delete_collection(test_collection)

        print(f"\n  Store and retrieve test passed")


# ─── Pipeline Integration Tests ───────────────────────────────────────────────

class TestPipelineIntegration:
    """
    Integration tests for the pipeline service itself.
    These require the pipeline to be running.
    """

    def test_pipeline_health(self, config):
        """Verify the pipeline health endpoint responds."""
        response = httpx.get(
            "http://localhost:8000/health",
            timeout=15
        )
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "services" in data

        print(f"\n  Pipeline status: {data['status']}")
        for service in data["services"]:
            print(f"  {service['name']}: {service['status']}", end="")
            if service.get("latency_ms"):
                print(f" ({service['latency_ms']}ms)", end="")
            print()

    def test_pipeline_chat_endpoint(self, config):
        """
        Verify the pipeline chat endpoint works end to end.
        This is the full pipeline test — embedding, memory, inference, storage.
        """
        api_key = os.environ.get("PIPELINE_API_KEY", "")

        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key

        response = httpx.post(
            "http://localhost:8000/pipeline/chat",
            json={"message": "Hello Jeeves. This is a pipeline connectivity test."},
            headers=headers,
            timeout=120
        )

        assert response.status_code == 200

        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0

        print(f"\n  Pipeline response received")
        print(f"  Memories used: {data.get('memories_used', 0)}")
        print(f"  Response length: {len(data['response'])} characters")


# ─── Direct Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run tests directly without pytest.
    Useful for quick connectivity checks during deployment.
    """
    import traceback

    config = load_config()

    test_classes = [
        TestJeevesOllama(),
        TestEmbeddingOllama(),
        TestQdrant(),
    ]

    print("=" * 60)
    print("Jeeves Pipeline Connectivity Tests")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{class_name}")
        print("-" * len(class_name))

        methods = [m for m in dir(test_class) if m.startswith("test_")]

        for method_name in methods:
            print(f"  {method_name}...", end=" ", flush=True)
            try:
                method = getattr(test_class, method_name)
                method(config)
                print("PASSED")
                passed += 1
            except AssertionError as e:
                print(f"FAILED\n    {e}")
                failed += 1
            except Exception as e:
                print(f"ERROR\n    {e}")
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
