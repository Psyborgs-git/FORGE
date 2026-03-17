"""Port interface contracts for FORGE hexagonal architecture."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from core.entities import (
    Document,
    GraphContext,
    InferenceRequest,
    InferenceResponse,
    MetricEvent,
    ModelInfo,
    RunHandle,
    SearchResult,
    Skill,
    SkillManifest,
    SkillResult,
    TrainingConfig,
)


class InferencePort(ABC):
    """Port for model inference operations."""

    @abstractmethod
    async def generate(self, req: InferenceRequest) -> InferenceResponse:
        """Generate text from the model (non-streaming)."""
        ...

    @abstractmethod
    async def stream(self, req: InferenceRequest) -> AsyncIterator[str]:
        """Generate text from the model (streaming tokens)."""
        ...

    @abstractmethod
    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings for texts."""
        ...

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List available models."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the inference backend is healthy."""
        ...


class TrainingBackendPort(ABC):
    """Port for training backend operations."""

    @abstractmethod
    async def start_run(self, config: TrainingConfig) -> RunHandle:
        """Start a training run."""
        ...

    @abstractmethod
    async def stream_metrics(self, handle: RunHandle) -> AsyncIterator[MetricEvent]:
        """Stream real-time metrics from a training run."""
        ...

    @abstractmethod
    async def cancel(self, handle: RunHandle) -> None:
        """Cancel a running training job."""
        ...

    @abstractmethod
    async def get_supported_methods(self) -> list[str]:
        """Get list of supported training methods."""
        ...


class VectorStorePort(ABC):
    """Port for vector storage and semantic search."""

    @abstractmethod
    async def upsert(self, docs: list[Document]) -> None:
        """Insert or update documents in the vector store."""
        ...

    @abstractmethod
    async def semantic_search(
        self, query_vec: list[float], k: int, filter: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """Perform semantic search using vector similarity."""
        ...

    @abstractmethod
    async def hybrid_search(
        self, query: str, query_vec: list[float], k: int, alpha: float = 0.7
    ) -> list[SearchResult]:
        """Perform hybrid search (vector + full-text). alpha: 0=pure BM25, 1=pure vector."""
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete documents by IDs."""
        ...


class GraphStorePort(ABC):
    """Port for knowledge graph operations."""

    @abstractmethod
    async def add_node(self, label: str, props: dict[str, Any]) -> str:
        """Add a node to the graph."""
        ...

    @abstractmethod
    async def add_edge(
        self, src_id: str, dst_id: str, rel_type: str, props: dict[str, Any] | None = None
    ) -> None:
        """Add an edge between two nodes."""
        ...

    @abstractmethod
    async def cypher_query(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a Cypher query."""
        ...

    @abstractmethod
    async def graphrag_retrieve(self, query: str, hops: int = 2) -> GraphContext:
        """Retrieve context from graph for GraphRAG."""
        ...

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        """Delete a node from the graph."""
        ...


class SkillRegistryPort(ABC):
    """Port for skills/MCP registry operations."""

    @abstractmethod
    async def list_skills(self) -> list[SkillManifest]:
        """List all available skills."""
        ...

    @abstractmethod
    async def get_skill(self, name: str) -> Skill:
        """Get a skill by name."""
        ...

    @abstractmethod
    async def install_skill(self, path_or_url: str) -> SkillManifest:
        """Install a skill from a local path or URL."""
        ...

    @abstractmethod
    async def invoke_skill(self, name: str, context: dict[str, Any]) -> SkillResult:
        """Invoke a skill with given context."""
        ...

    @abstractmethod
    async def uninstall_skill(self, name: str) -> None:
        """Uninstall a skill."""
        ...
