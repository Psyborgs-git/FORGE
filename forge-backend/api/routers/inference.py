"""Inference API router."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api.deps import get_inference_port
from core.entities import InferenceRequest as InferenceReq
from core.entities import InferenceResponse as InferenceResp
from core.entities import ModelInfo
from core.ports import InferencePort

router = APIRouter()


class InferenceRequest(BaseModel):
    """Inference request model."""

    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0
    top_k: int = 50
    stop: list[str] | None = None
    system: str | None = None


class InferenceResponse(BaseModel):
    """Inference response model."""

    model: str
    content: str
    finish_reason: str | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: dict = {}


class EmbedRequest(BaseModel):
    """Embedding request model."""

    texts: list[str]
    model: str | None = None


class EmbedResponse(BaseModel):
    """Embedding response model."""

    embeddings: list[list[float]]
    model: str


class ModelInfoResponse(BaseModel):
    """Model info response."""

    name: str
    size: int | None = None
    parameter_count: str | None = None
    quantization: str | None = None
    family: str | None = None
    modified_at: str | None = None
    metadata: dict = {}


@router.post("/generate")
async def generate(
    request: InferenceRequest,
    inference_port: InferencePort = Depends(get_inference_port),
) -> InferenceResponse:
    """Generate text from a model (non-streaming)."""
    req = InferenceReq(
        model=request.model,
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=request.stop,
        system=request.system,
        stream=False,
    )

    response = await inference_port.generate(req)

    return InferenceResponse(
        model=response.model,
        content=response.content,
        finish_reason=response.finish_reason,
        total_tokens=response.total_tokens,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        metadata=response.metadata,
    )


@router.post("/embed")
async def embed(
    request: EmbedRequest,
    inference_port: InferencePort = Depends(get_inference_port),
) -> EmbedResponse:
    """Generate embeddings for texts."""
    embeddings = await inference_port.embed(request.texts, request.model)

    return EmbedResponse(
        embeddings=embeddings,
        model=request.model or "nomic-embed-text",
    )


@router.get("/models")
async def list_models(
    inference_port: InferencePort = Depends(get_inference_port),
) -> list[ModelInfoResponse]:
    """List available models."""
    models = await inference_port.list_models()

    return [
        ModelInfoResponse(
            name=m.name,
            size=m.size,
            parameter_count=m.parameter_count,
            quantization=m.quantization,
            family=m.family,
            modified_at=m.modified_at.isoformat() if m.modified_at else None,
            metadata=m.metadata,
        )
        for m in models
    ]


@router.get("/health")
async def health_check(
    inference_port: InferencePort = Depends(get_inference_port),
) -> dict:
    """Check inference backend health."""
    is_healthy = await inference_port.health_check()

    return {
        "healthy": is_healthy,
        "backend": "ollama",
    }
