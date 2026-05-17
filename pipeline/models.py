"""
pipeline/models.py
Pydantic data models for the Jeeves Memory Pipeline.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, validator


class ChatRequest(BaseModel):
    message: str
    user_id: str = "human"

    @validator("message")
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Message cannot be empty")
        if len(v) > 32000:
            raise ValueError("Message exceeds maximum length")
        return v


class ChatResponse(BaseModel):
    response: str
    memories_used: int = 0
    memory_contexts: list[str] = []
    timestamp: str = ""

    def __init__(self, **data):
        if "timestamp" not in data or not data["timestamp"]:
            data["timestamp"] = datetime.now().isoformat()
        super().__init__(**data)


class OllamaOptions(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    num_predict: Optional[int] = None
    stop: Optional[list[str]] = None
    seed: Optional[int] = None

    class Config:
        extra = "allow"


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    options: Optional[OllamaOptions] = None
    system: Optional[str] = None

    @validator("prompt")
    def prompt_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        if len(v) > 32000:
            raise ValueError("Prompt exceeds maximum length")
        return v


class OllamaGenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool = True
    total_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

    def __init__(self, **data):
        if "created_at" not in data or not data["created_at"]:
            data["created_at"] = datetime.now().isoformat()
        super().__init__(**data)


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[dict]
    stream: bool = False
    options: Optional[OllamaOptions] = None

    @validator("messages")
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v


class OllamaChatResponse(BaseModel):
    model: str
    created_at: str
    message: dict
    done: bool = True
    total_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

    def __init__(self, **data):
        if "created_at" not in data or not data["created_at"]:
            data["created_at"] = datetime.now().isoformat()
        super().__init__(**data)


class ServiceHealth(BaseModel):
    name: str
    status: str
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    services: list[ServiceHealth] = []
    timestamp: str = ""
    version: str = "1.0.0"

    def __init__(self, **data):
        if "timestamp" not in data or not data["timestamp"]:
            data["timestamp"] = datetime.now().isoformat()
        super().__init__(**data)
