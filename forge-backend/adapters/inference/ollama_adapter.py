"""Ollama adapter implementing InferencePort."""
from __future__ import annotations

from typing import AsyncIterator

import httpx

from config import settings
from core.entities import InferenceRequest, InferenceResponse, ModelInfo
from core.ports import InferencePort


class OllamaAdapter(InferencePort):
    """Adapter for Ollama inference backend."""

    def __init__(self, base_url: str | None = None) -> None:
        """Initialize Ollama adapter."""
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.client = httpx.AsyncClient(timeout=300.0, base_url=self.base_url)

    async def generate(self, req: InferenceRequest) -> InferenceResponse:
        """Generate text from model (non-streaming)."""
        try:
            payload = {
                "model": req.model,
                "messages": req.messages,
                "stream": False,
                "options": {
                    "temperature": req.temperature,
                    "top_p": req.top_p,
                    "top_k": req.top_k,
                },
            }

            if req.system:
                payload["system"] = req.system

            if req.stop:
                payload["options"]["stop"] = req.stop

            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()

            data = response.json()

            return InferenceResponse(
                model=req.model,
                content=data.get("message", {}).get("content", ""),
                finish_reason=data.get("done_reason"),
                total_tokens=data.get("eval_count", 0) + data.get("prompt_eval_count", 0),
                prompt_tokens=data.get("prompt_eval_count"),
                completion_tokens=data.get("eval_count"),
                metadata={
                    "created_at": data.get("created_at"),
                    "total_duration": data.get("total_duration"),
                    "load_duration": data.get("load_duration"),
                    "prompt_eval_duration": data.get("prompt_eval_duration"),
                    "eval_duration": data.get("eval_duration"),
                },
            )
        except httpx.HTTPStatusError as e:
            raise InferenceError(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise InferenceError(f"Ollama connection error: {str(e)}")
        except Exception as e:
            raise InferenceError(f"Ollama generation error: {str(e)}")

    async def stream(self, req: InferenceRequest) -> AsyncIterator[str]:
        """Generate text from model (streaming)."""
        try:
            payload = {
                "model": req.model,
                "messages": req.messages,
                "stream": True,
                "options": {
                    "temperature": req.temperature,
                    "top_p": req.top_p,
                    "top_k": req.top_k,
                },
            }

            if req.system:
                payload["system"] = req.system

            if req.stop:
                payload["options"]["stop"] = req.stop

            async with self.client.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            import json

                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                content = data["message"]["content"]
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPStatusError as e:
            raise InferenceError(f"Ollama HTTP error: {e.response.status_code}")
        except httpx.RequestError as e:
            raise InferenceError(f"Ollama connection error: {str(e)}")
        except Exception as e:
            raise InferenceError(f"Ollama streaming error: {str(e)}")

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings for texts."""
        try:
            embedding_model = model or "nomic-embed-text"
            embeddings = []

            for text in texts:
                response = await self.client.post(
                    "/api/embed", json={"model": embedding_model, "input": text}
                )
                response.raise_for_status()
                data = response.json()
                embeddings.append(data.get("embeddings", [[]])[0])

            return embeddings
        except httpx.HTTPStatusError as e:
            raise InferenceError(f"Ollama embed HTTP error: {e.response.status_code}")
        except httpx.RequestError as e:
            raise InferenceError(f"Ollama connection error: {str(e)}")
        except Exception as e:
            raise InferenceError(f"Ollama embed error: {str(e)}")

    async def list_models(self) -> list[ModelInfo]:
        """List available models."""
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            data = response.json()

            models = []
            for model in data.get("models", []):
                from datetime import datetime

                modified_at = None
                if "modified_at" in model:
                    try:
                        modified_at = datetime.fromisoformat(
                            model["modified_at"].replace("Z", "+00:00")
                        )
                    except Exception:
                        pass

                models.append(
                    ModelInfo(
                        name=model.get("name", ""),
                        size=model.get("size"),
                        parameter_count=model.get("details", {}).get("parameter_size"),
                        quantization=model.get("details", {}).get("quantization_level"),
                        family=model.get("details", {}).get("family"),
                        modified_at=modified_at,
                        metadata=model.get("details", {}),
                    )
                )

            return models
        except httpx.HTTPStatusError as e:
            raise InferenceError(f"Ollama list models HTTP error: {e.response.status_code}")
        except httpx.RequestError as e:
            raise InferenceError(f"Ollama connection error: {str(e)}")
        except Exception as e:
            raise InferenceError(f"Ollama list models error: {str(e)}")

    async def health_check(self) -> bool:
        """Check if Ollama is healthy."""
        try:
            response = await self.client.get("/api/version")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


class InferenceError(Exception):
    """Custom exception for inference errors."""

    pass
