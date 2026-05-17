"""
Jeeves Memory Pipeline - Main Application
"""

import logging
import logging.handlers
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipeline.memory import MemoryManager
from pipeline.search import SearchEngine
from pipeline.inference import InferenceEngine
from pipeline.security import SecurityMiddleware, RateLimiter
from pipeline.health import HealthChecker
from pipeline.models import (
    ChatRequest, ChatResponse,
    OllamaGenerateRequest, OllamaGenerateResponse,
    OllamaChatRequest, OllamaChatResponse,
    HealthResponse,
)


def setup_logging(config: dict) -> logging.Logger:
    log_path = config.get("logging", {}).get("path", "/var/log/jeeves-pipeline/pipeline.log")
    log_level = config.get("logging", {}).get("level", "INFO")
    max_bytes = config.get("logging", {}).get("max_size_mb", 100) * 1024 * 1024
    backup_count = config.get("logging", {}).get("backup_count", 10)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("jeeves-pipeline")
    logger.setLevel(getattr(logging, log_level))

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_config() -> dict:
    config_path = os.environ.get(
        "JEEVES_CONFIG",
        "/opt/jeeves-pipeline/config/config.yaml"
    )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = app.state.logger
    config = app.state.config

    logger.info("=" * 60)
    logger.info("Jeeves Memory Pipeline starting up")
    logger.info("=" * 60)

    logger.info("Initialising memory manager...")
    app.state.memory = MemoryManager(config, logger)
    await app.state.memory.initialise()
    logger.info("Memory manager ready")

    logger.info("Initialising inference engine...")
    app.state.inference = InferenceEngine(config, logger)
    logger.info("Inference engine ready")

    app.state.health = HealthChecker(config, logger)

    logger.info("Pipeline ready — the conductor is on the podium")
    logger.info("=" * 60)

    yield

    logger.info("Pipeline shutting down gracefully")


def create_app() -> FastAPI:
    config = load_config()
    logger = setup_logging(config)

    app = FastAPI(
        title="Jeeves Memory Pipeline",
        description="Memory-enriched inference pipeline for Jeeves",
        version="1.0.0",
        lifespan=lifespan
    )

    app.state.config = config
    app.state.logger = logger

    allowed_origins = config.get("security", {}).get(
        "allowed_origins", ["http://jeeves-services.local"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    app.add_middleware(SecurityMiddleware, config=config, logger=logger)

    app.state.search = SearchEngine(config, logger)

    return app


app = create_app()
rate_limiter = RateLimiter()


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    return await request.app.state.health.check_all()


@app.get("/api/tags")
async def list_models(request: Request):
    return await request.app.state.inference.list_models()


@app.post("/api/generate", response_model=OllamaGenerateResponse)
async def generate(request: Request, body: OllamaGenerateRequest):
    logger = request.app.state.logger
    memory = request.app.state.memory
    inference = request.app.state.inference

    start_time = datetime.now()
    logger.info(f"Generate request | Model: {body.model} | Prompt length: {len(body.prompt)}")

    try:
        client_ip = request.client.host
        await rate_limiter.check(client_ip, request.app.state.config)

        memories = await memory.retrieve(body.prompt)
        logger.info(f"Retrieved {len(memories)} relevant memories")

        enriched_prompt = memory.build_enriched_prompt(body.prompt, memories)

        response = await inference.generate(
            model=body.model,
            prompt=enriched_prompt,
            options=body.options
        )

        await memory.store(
            user_message=body.prompt,
            assistant_response=response.response
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Generate complete | Elapsed: {elapsed:.2f}s")

        return response

    except Exception as e:
        logger.error(f"Generate error: {str(e)}", exc_info=True)
        raise


@app.post("/api/chat", response_model=OllamaChatResponse)
async def chat(request: Request, body: OllamaChatRequest):
    logger = request.app.state.logger
    memory = request.app.state.memory
    inference = request.app.state.inference

    start_time = datetime.now()

    user_messages = [m for m in body.messages if m.get("role") == "user"]
    latest_message = user_messages[-1]["content"] if user_messages else ""

    logger.info(f"Chat request | Model: {body.model} | Messages: {len(body.messages)}")

    try:
        client_ip = request.client.host
        await rate_limiter.check(client_ip, request.app.state.config)

        memories = await memory.retrieve(latest_message)
        logger.info(f"Retrieved {len(memories)} relevant memories")

        search_results = None
        if request.app.state.search.enabled:
            if request.app.state.search.should_search(latest_message):
                logger.info("Query triggers web search")
                jeeves_cfg = request.app.state.config["services"]["jeeves_ollama"]
                search_results = await request.app.state.search.search(
                    latest_message,
                    f"http://{jeeves_cfg['host']}:{jeeves_cfg['port']}",
                    jeeves_cfg["model"]
                )

        enriched_messages = memory.inject_memory_context(
            body.messages, memories, search_results
        )

        response = await inference.chat(
            model=body.model,
            messages=enriched_messages,
            options=body.options
        )

        assistant_content = response.message.get("content", "")
        await memory.store(
            user_message=latest_message,
            assistant_response=assistant_content
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Chat complete | Elapsed: {elapsed:.2f}s")

        return response

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise


@app.post("/pipeline/chat", response_model=ChatResponse)
async def pipeline_chat(request: Request, body: ChatRequest):
    logger = request.app.state.logger
    memory = request.app.state.memory
    inference = request.app.state.inference

    logger.info(f"Pipeline chat | Message: {body.message[:100]}...")

    try:
        memories = await memory.retrieve(body.message)
        enriched_messages = memory.inject_memory_context(
            [{"role": "user", "content": body.message}],
            memories
        )

        response = await inference.chat(
            model=request.app.state.config["services"]["jeeves_ollama"]["model"],
            messages=enriched_messages
        )

        assistant_content = response.message.get("content", "")
        await memory.store(body.message, assistant_content)

        return ChatResponse(
            response=assistant_content,
            memories_used=len(memories),
            memory_contexts=[m.get("text", "") for m in memories]
        )

    except Exception as e:
        logger.error(f"Pipeline chat error: {str(e)}", exc_info=True)
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = request.app.state.logger
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal pipeline error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    config = load_config()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=2, log_level="info")
