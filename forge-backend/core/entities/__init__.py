"""Domain entities for FORGE."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass
class InferenceRequest:
    """Request to generate text from a model."""

    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0
    top_k: int = 50
    stop: list[str] | None = None
    stream: bool = False
    system: str | None = None


@dataclass
class InferenceResponse:
    """Response from text generation."""

    model: str
    content: str
    finish_reason: str | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Information about an available model."""

    name: str
    size: int | None = None
    parameter_count: str | None = None
    quantization: str | None = None
    family: str | None = None
    modified_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for a training run."""

    project_id: UUID
    dataset_id: UUID
    architecture_id: UUID | None
    backend: str  # 'torchtune' | 'llamafactory'
    method: str  # 'lora' | 'qlora' | 'full' | 'dpo' | 'orpo'
    base_model: str
    output_dir: str
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunHandle:
    """Handle to a running training process."""

    run_id: UUID
    process_id: int
    status: str
    started_at: datetime


@dataclass
class MetricEvent:
    """Real-time metric from a training run."""

    run_id: UUID
    time: datetime
    step: int
    train_loss: float | None = None
    eval_loss: float | None = None
    learning_rate: float | None = None
    gpu_mem_gb: float | None = None
    throughput: float | None = None


@dataclass
class Document:
    """Document for vector storage."""

    id: str
    content: str
    vector: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    project_id: UUID | None = None
    source_type: str | None = None
    source_id: UUID | None = None


@dataclass
class SearchResult:
    """Result from vector/hybrid search."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphContext:
    """Context retrieved from knowledge graph."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillManifest:
    """Manifest describing a skill."""

    name: str
    description: str
    version: str
    port: str | None = None
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    requires: dict[str, Any] = field(default_factory=dict)
    config_schema: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)


@dataclass
class Skill:
    """A skill with its manifest and content."""

    manifest: SkillManifest
    skill_path: str
    content: str
    is_active: bool = True


@dataclass
class SkillResult:
    """Result from invoking a skill."""

    skill_name: str
    success: bool
    content: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
