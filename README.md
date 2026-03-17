# FORGE — Local AI Research & Build Studio

> A local-first, full-stack AI development environment for visual model building, fine-tuning, dataset management, DSPy pipeline composition, RAG/KG engineering, and interactive model experimentation.

-----

## Table of Contents

1. [Vision & Goals](#1-vision--goals)
1. [What Makes FORGE Different](#2-what-makes-forge-different)
1. [Assumptions & Decisions](#3-assumptions--decisions)
1. [System Architecture — Layer Stack](#4-system-architecture--layer-stack)
1. [Hexagonal Architecture — Ports & Adapters](#5-hexagonal-architecture--ports--adapters)
1. [The 10 Core Modules](#6-the-10-core-modules)
1. [Database Architecture](#7-database-architecture)
1. [Port Interface Contracts](#8-port-interface-contracts)
1. [Frontend Architecture](#9-frontend-architecture)
1. [DSPy Engine Design](#10-dspy-engine-design)
1. [Training System](#11-training-system)
1. [RAG & Knowledge Graph](#12-rag--knowledge-graph)
1. [MCP & Skills Filesystem](#13-mcp--skills-filesystem)
1. [Complete Tech Stack](#14-complete-tech-stack)
1. [Monorepo Structure](#15-monorepo-structure)
1. [Phased Build Plan](#16-phased-build-plan)
1. [Risks & Mitigations](#17-risks--mitigations)
1. [Development Setup](#18-development-setup)

-----

## 1. Vision & Goals

**FORGE** is a local AI lab where engineers and researchers forge models — build them from components, fine-tune them on their own data, compose them into pipelines, and test them live. Everything runs on the user’s machine. Nothing leaves unless the user explicitly chooses to push it.

### Target Users

FORGE serves two personas equally, with a persistent mode toggle:

|Mode    |User                                       |Experience                                |
|--------|-------------------------------------------|------------------------------------------|
|`GUIDED`|Non-technical researchers, product builders|LLM-assisted, form-driven, plain language |
|`EXPERT`|ML engineers, researchers                  |Raw YAML configs, code views, full control|

All features exist in both modes. The mode changes how things are *surfaced*, not what is *available*.

### Core Principles

- **Local-first**: All compute, data, and models stay on the user’s hardware by default
- **Privacy-preserving**: No telemetry, no cloud dependency, no data exfiltration
- **Hexagonal architecture**: Every capability is pluggable via Port/Adapter pattern — swap backends without touching business logic
- **DSPy-native**: Pipelines are DSPy programs; visual builders compile to runnable Python
- **Dual-mode UX**: One codebase, two render modes — Guided and Expert share the same domain model

-----

## 2. What Makes FORGE Different

No existing tool combines all of the following in a single local-first studio:

|Tool         |What It Does                        |What It Misses                                     |
|-------------|------------------------------------|---------------------------------------------------|
|LM Studio    |Local inference                     |No training, no pipelines, no datasets             |
|Flowise      |Visual LLM pipelines                |No training, no fine-tuning, no dataset management |
|LLaMA-Factory|Fine-tuning UI                      |No pipeline builder, no RAG, no experiment registry|
|Coze / Dify  |Agent builder                       |Cloud-dependent, no training                       |
|**FORGE**    |**All of the above, local, unified**|—                                                  |

FORGE is the only tool that unifies:

- Visual drag-and-drop model architecture designer
- LoRA / QLoRA / full fine-tuning with Torchtune + LLaMA-Factory
- DSPy pipeline graph editor with MIPROv2 optimization
- Dataset management, versioning, cleaning, synthetic generation
- GraphRAG, Agentic RAG, Hybrid RAG, multi-modal RAG
- Experiment registry with full reproducibility
- Research aggregator (ArXiv, HuggingFace papers)
- MCP-style skills filesystem for plug-and-play extensibility

-----

## 3. Assumptions & Decisions

These were decided during architecture planning. All are revisable before coding begins.

|Decision            |Choice                                        |Rationale                                                    |
|--------------------|----------------------------------------------|-------------------------------------------------------------|
|Deployment target   |Tauri v2 desktop app + Docker Compose         |Covers both desktop users and headless/server installs       |
|User tier priority  |Equal weight, full dual-mode from day one     |Mode is a render variant, not a feature gate                 |
|Visual builder scope|All four capabilities (see Module 1)          |Validates complete feature set from Phase 3                  |
|Fine-tuning backends|Torchtune + LLaMA-Factory                     |PyTorch-native control + broadest model coverage             |
|LLM guidance layer  |User-configured: any LLM via Ollama or API key|No vendor lock-in; works fully offline                       |
|Database strategy   |PostgreSQL 16 + extensions (single instance)  |pgvector + Apache AGE + pg_trgm + TimescaleDB coexist on PG16|
|Vector DB           |pgvector (no separate Chroma/Qdrant)          |Confirmed coexistence, 471 QPS on 50M vectors at 99% recall  |
|Graph DB            |Apache AGE (no separate Neo4j)                |openCypher on top of SQL, PG16 1.5.0 stable                  |
|Adapter plugins     |JSON manifest (`forge-adapter.json`)          |Same pattern as Anthropic’s skills system                    |

-----

## 4. System Architecture — Layer Stack

Six layers, strict separation. No layer calls down more than one level. All cross-layer communication goes through defined port interfaces.

```
┌─────────────────────────────────────────────────────────────────────┐
│  L1 — PRESENTATION                                                  │
│  React 19 + TypeScript SPA. Dual-mode UI. @xyflow/react builders.  │
├─────────────────────────────────────────────────────────────────────┤
│  L2 — API GATEWAY                                                   │
│  FastAPI. WebSocket streams (training/inference). REST (CRUD).      │
│  Pydantic v2 contracts. Local JWT auth.                             │
├─────────────────────────────────────────────────────────────────────┤
│  L3 — DOMAIN CORE                                                   │
│  Pure Python. No I/O. No framework deps. Use cases, entities,       │
│  Port ABCs. This is the hexagonal architecture heart.               │
├─────────────────────────────────────────────────────────────────────┤
│  L4 — DSPY ENGINE                                                   │
│  Pipeline composition, MIPROv2 optimization, ReAct agents,          │
│  BootstrapFinetune, tool-use. Bridges domain to LLM APIs.           │
├─────────────────────────────────────────────────────────────────────┤
│  L5 — ADAPTERS                                                      │
│  Concrete implementations of all ports. Swappable without changing  │
│  domain logic. Registered via manifest. Runtime-selectable.         │
├─────────────────────────────────────────────────────────────────────┤
│  L6 — INFRASTRUCTURE                                                │
│  PostgreSQL 16 (pgvector, Apache AGE, pg_trgm, pgai, TimescaleDB)  │
│  Filesystem (skills/MCP), hardware detection, process management.   │
└─────────────────────────────────────────────────────────────────────┘
```

**Layer dependency rule**: Dependencies flow inward only. The domain core (L3) imports from nothing outside itself.

-----

## 5. Hexagonal Architecture — Ports & Adapters

### Design Principle

Every major capability is defined as an abstract **Port** (Python ABC) in the domain core. Concrete **Adapters** implement those ports. The domain core never imports from adapters.

### Driving Ports (Input — what the UI/API calls)

|Port                      |Responsibility                           |
|--------------------------|-----------------------------------------|
|`PipelineBuilderPort`     |Create, serialize, and run DSPy pipelines|
|`TrainingOrchestratorPort`|Trigger and monitor fine-tune runs       |
|`DatasetManagerPort`      |CRUD, version, and transform datasets    |
|`PlaygroundPort`          |Send inference requests to models        |
|`ExperimentRegistryPort`  |Log, query, and compare experiment runs  |
|`ResearchAggregatorPort`  |Search ArXiv, HuggingFace, GitHub        |

### Driven Ports (Output — what adapters implement)

|Port                 |V1 Adapters                          |Optional Adapters      |
|---------------------|-------------------------------------|-----------------------|
|`InferencePort`      |OllamaAdapter, LlamaCppAdapter       |VLLMAdapter (Phase 4)  |
|`TrainingBackendPort`|TorchtuneAdapter, LlamaFactoryAdapter|—                      |
|`VectorStorePort`    |PgVectorAdapter                      |(future: QdrantAdapter)|
|`GraphStorePort`     |ApacheAGEAdapter                     |Neo4jAdapter (optional)|
|`DocumentStorePort`  |PostgreSQLAdapter + FilesystemAdapter|—                      |
|`SkillRegistryPort`  |MCPFilesystemAdapter                 |—                      |

### Adapter Plugin Manifest

Third-party adapters register via a `forge-adapter.json` file. FORGE scans `/skills` and `/plugins` directories at startup and registers valid manifests.

```json
{
  "name": "my-custom-adapter",
  "version": "1.0.0",
  "port": "VectorStorePort",
  "entrypoint": "./adapter.py",
  "class": "MyCustomVectorAdapter",
  "config_schema": "./config.schema.json",
  "capabilities": ["semantic_search", "hybrid_search"],
  "requires": {
    "python": ">=3.11",
    "packages": ["my-db-client>=2.0"]
  }
}
```

-----

## 6. The 10 Core Modules

Each module owns its own data, UI surface, and business logic. Modules communicate through domain events, not direct calls.

### Module 1 — Model Forge

Visual drag-and-drop model architecture builder.

- Drag layer types onto canvas (Attention, FFN, Embedding, Norm, LoRA adapter, MoE router, etc.)
- Attach LoRA / QLoRA adapters to HuggingFace base models visually
- Architecture template library: Transformer, MoE, Encoder-only, Decoder-only, RAG, etc.
- Export graph IR to runnable PyTorch model code
- GUIDED mode: wizard that asks “what task?” and recommends architecture

### Module 2 — Pipeline Studio

DSPy pipeline graph editor using `@xyflow/react`.

- Nodes: `dspy.Predict`, `dspy.ChainOfThought`, `dspy.ReAct`, `dspy.ProgramOfThought`, `dspy.MultiChainComparison`
- Retrieval nodes: Vector Retriever, Graph Retriever, Hybrid Retriever, BM25, Web Search
- Control nodes: Input/Output, Router (conditional), Aggregator, Loop, Skill/MCP Tool
- Run MIPROv2 optimization directly from the graph editor
- Export graph to valid DSPy Python code (Monaco editor preview)
- Save/load pipeline versions with optimized state

### Module 3 — Data Lab

Full dataset lifecycle management.

- Upload raw data: CSV, JSONL, Parquet, PDF, images
- Version datasets with semantic versioning and SHA-256 content hash
- Visual cleaning: deduplication, outlier detection, length distribution, token stats
- Auto-format to training formats: SFT (instruction-response), DPO (chosen/rejected), RLHF, ORPO
- Synthetic data generation: define schema + LLM pipeline to generate N examples
- Web scraping tool: extract + structure data from URLs
- GUIDED mode: “Prepare this data for fine-tuning” auto-formats with preview

### Module 4 — Training Center

End-to-end fine-tuning workflow.

- Backend adapter selection: Torchtune or LLaMA-Factory
- Methods: LoRA, QLoRA, Full FT, DPO, ORPO, knowledge distillation
- GUIDED mode: form-driven config builder with human-readable fields
- EXPERT mode: raw YAML config editor (Monaco)
- Real-time training dashboard: loss curves, learning rate, GPU memory (TimescaleDB stream)
- Hardware Advisor gates unavailable options based on detected VRAM
- Cancel, pause, resume training runs
- GGUF / ONNX export wizard post-training

### Module 5 — Eval Engine

Model evaluation and comparison.

- Custom metric definitions (Python functions)
- LLM-as-judge evaluation pipelines (DSPy-powered)
- A/B comparison: run two model versions on same eval set, diff results
- Training curve visualization (loss, perplexity, eval metrics over time)
- Export results to W&B and MLflow (JSON + API)
- Benchmark presets: MMLU, HellaSwag, TruthfulQA (via lm-evaluation-harness)

### Module 6 — Playground

Interactive model experimentation.

- Chat interface: single-turn and multi-turn with history
- Batch inference runner: submit JSONL file, get JSONL responses
- Side-by-side comparison: two models, same prompt, compare outputs
- Custom system prompt / persona builder with template library
- Token probability visualizer: color-coded token likelihood on responses
- Attention visualizer: heatmap of attention patterns (where available)
- EXPERT mode: raw generation params (temperature, top-p, top-k, min-p, repetition penalty)

### Module 7 — RAG / KG Studio

Retrieval-augmented generation and knowledge graph pipelines.

- Ingestion pipeline: PDF, images, HTML, CSV, DOCX chunked and embedded
- Standard RAG, GraphRAG, Agentic RAG (multi-hop), Hybrid RAG, Multi-modal RAG
- Knowledge graph editor: visual entity/relation builder backed by Apache AGE
- Cypher query editor for direct graph queries
- Retrieval testing: input query to see exact chunks/graph nodes retrieved
- All RAG pipelines are DSPy programs: visual to code export works here too

### Module 8 — Experiment Registry

Versioned record of every run for full reproducibility.

- Every training run captured: config snapshot, dataset SHA-256, hardware profile, metrics
- Query and filter runs by model, dataset, metric range, date
- “Reproduce this run” button recreates exact environment + config
- Compare runs: diff configs and metrics side-by-side
- Link runs to pipeline versions and dataset versions
- Export run lineage as JSON or push model card to HuggingFace Hub

### Module 9 — Research Lab

DSPy-powered research aggregation agent.

- Search ArXiv by keyword, category, author, date range
- Search HuggingFace papers, models, and datasets
- Search GitHub repositories for relevant implementations
- Auto-summarize papers: abstract, key contributions, architecture decisions, datasets used
- “Suggest architecture” — given a task description, retrieve and synthesize relevant papers
- Save research to project knowledge base (embedded + stored in pgvector for RAG)

### Module 10 — Lab Assistant

The GUIDED mode AI brain — aware of your project.

- Implemented as a `dspy.ReAct` agent with FORGE’s internal tools
- Context-aware: knows current project, hardware profile, datasets, models, runs
- Tools available to the agent:
  - `get_hardware_profile()` — VRAM, RAM, disk
  - `list_available_base_models()` — HF Hub filtered by hardware constraints
  - `explain_architecture_choice()` — goal to recommended architecture
  - `check_dataset_quality()` — profile current project dataset
  - `estimate_training_cost()` — time + VRAM estimate before running
  - `search_arxiv()` — research tool
  - `search_huggingface()` — model/dataset discovery
  - `invoke_skill()` — run a skill from the registry
- Configurable LLM backend: any Ollama model or API key (Claude, OpenAI, Gemini, etc.)

-----

## 7. Database Architecture

### Single PostgreSQL 16 Instance — Extension Stack

|Extension         |Purpose                                        |Replaces                   |
|------------------|-----------------------------------------------|---------------------------|
|`pgvector`        |Embedding storage + ANN search (HNSW + IVFFlat)|Chroma, Qdrant, Weaviate   |
|`Apache AGE 1.5.0`|Property graph + openCypher queries            |Neo4j, Kuzu                |
|`pg_trgm`         |Fuzzy text + BM25-style full-text (GIN index)  |Elasticsearch              |
|`pgai`            |LLM embedding/generation calls from SQL        |Separate embedding services|
|`TimescaleDB`     |Training metrics time-series (hypertable)      |InfluxDB, Prometheus TSDB  |

**Confirmed**: pgvector + Apache AGE + pg_trgm coexist on PostgreSQL 16. All three extensions are built from source against the same PG16 base in the Docker image.

### Core Schema

```sql
-- Projects
CREATE TABLE projects (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name        TEXT NOT NULL,
  mode        TEXT DEFAULT 'guided',
  created_at  TIMESTAMPTZ DEFAULT now(),
  metadata    JSONB DEFAULT '{}'
);

-- Model Architectures (graph IR stored as JSONB)
CREATE TABLE model_architectures (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      UUID REFERENCES projects(id),
  name            TEXT NOT NULL,
  graph_ir        JSONB NOT NULL,
  base_model_id   TEXT,
  adapter_config  JSONB DEFAULT '{}',
  exported_code   TEXT,
  created_at      TIMESTAMPTZ DEFAULT now()
);

-- Datasets with versioning
CREATE TABLE datasets (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id    UUID REFERENCES projects(id),
  name          TEXT NOT NULL,
  version       TEXT NOT NULL,
  format        TEXT NOT NULL,        -- 'sft' | 'dpo' | 'rlhf' | 'orpo' | 'raw'
  row_count     INTEGER,
  hash_sha256   TEXT NOT NULL,
  storage_path  TEXT NOT NULL,
  stats         JSONB DEFAULT '{}',
  created_at    TIMESTAMPTZ DEFAULT now()
);

-- Training Runs
CREATE TABLE training_runs (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id        UUID REFERENCES projects(id),
  dataset_id        UUID REFERENCES datasets(id),
  architecture_id   UUID REFERENCES model_architectures(id),
  backend           TEXT NOT NULL,    -- 'torchtune' | 'llamafactory'
  method            TEXT NOT NULL,    -- 'lora' | 'qlora' | 'full' | 'dpo' | 'orpo'
  config_snapshot   JSONB NOT NULL,
  status            TEXT DEFAULT 'queued',
  hardware_snapshot JSONB,
  output_path       TEXT,
  started_at        TIMESTAMPTZ,
  completed_at      TIMESTAMPTZ,
  error_message     TEXT
);

-- Training Metrics (TimescaleDB hypertable)
CREATE TABLE training_metrics (
  run_id        UUID REFERENCES training_runs(id),
  time          TIMESTAMPTZ NOT NULL,
  step          INTEGER NOT NULL,
  train_loss    FLOAT8,
  eval_loss     FLOAT8,
  learning_rate FLOAT8,
  gpu_mem_gb    FLOAT4,
  throughput    FLOAT4
);
SELECT create_hypertable('training_metrics', 'time');

-- DSPy Pipelines
CREATE TABLE pipelines (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      UUID REFERENCES projects(id),
  name            TEXT NOT NULL,
  graph_json      JSONB NOT NULL,
  dspy_code       TEXT,
  optimized_state JSONB,
  version         INTEGER DEFAULT 1,
  created_at      TIMESTAMPTZ DEFAULT now()
);

-- Vector Embeddings (pgvector)
CREATE TABLE embeddings (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id  UUID REFERENCES projects(id),
  content     TEXT NOT NULL,
  vector      VECTOR(1536),
  source_type TEXT,
  source_id   UUID,
  metadata    JSONB DEFAULT '{}'
);
CREATE INDEX ON embeddings USING hnsw (vector vector_cosine_ops);
CREATE INDEX ON embeddings USING gin (content gin_trgm_ops);

-- Skills / MCP Registry
CREATE TABLE skills (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name          TEXT UNIQUE NOT NULL,
  skill_path    TEXT NOT NULL,
  manifest      JSONB NOT NULL,
  port_binding  TEXT,
  is_active     BOOLEAN DEFAULT true,
  installed_at  TIMESTAMPTZ DEFAULT now()
);

-- Experiments
CREATE TABLE experiments (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id    UUID REFERENCES projects(id),
  name          TEXT NOT NULL,
  description   TEXT,
  run_ids       UUID[],
  pipeline_ids  UUID[],
  dataset_ids   UUID[],
  final_metrics JSONB DEFAULT '{}',
  tags          TEXT[],
  created_at    TIMESTAMPTZ DEFAULT now()
);
```

### Knowledge Graph (Apache AGE)

Each project gets its own AGE graph:

```sql
SELECT * FROM ag_catalog.create_graph('project_kg_<project_id>');

-- Example: find all models fine-tuned from llama3.2
SELECT * FROM ag_catalog.cypher('project_kg_<id>', $$
  MATCH (m:Model)-[:FINE_TUNED_FROM]->(base:Model)
  WHERE base.name = 'llama3.2'
  RETURN m, base
$$) AS (m ag_catalog.agtype, base ag_catalog.agtype);
```

-----

## 8. Port Interface Contracts

All ports live in `forge-backend/core/ports/`. They are pure Python ABCs with no I/O.

### InferencePort

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
from core.entities import InferenceRequest, InferenceResponse, ModelInfo

class InferencePort(ABC):
    @abstractmethod
    async def generate(self, req: InferenceRequest) -> InferenceResponse: ...
    @abstractmethod
    async def stream(self, req: InferenceRequest) -> AsyncIterator[str]: ...
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    @abstractmethod
    async def list_models(self) -> list[ModelInfo]: ...
    @abstractmethod
    async def health_check(self) -> bool: ...
# Adapters: OllamaAdapter, LlamaCppAdapter, VLLMAdapter
```

### TrainingBackendPort

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
from core.entities import TrainingConfig, RunHandle, MetricEvent

class TrainingBackendPort(ABC):
    @abstractmethod
    async def start_run(self, config: TrainingConfig) -> RunHandle: ...
    @abstractmethod
    async def stream_metrics(self, handle: RunHandle) -> AsyncIterator[MetricEvent]: ...
    @abstractmethod
    async def cancel(self, handle: RunHandle) -> None: ...
    @abstractmethod
    async def get_supported_methods(self) -> list[str]: ...
    # Returns: ['lora', 'qlora', 'full', 'dpo', 'orpo', 'kd']
# Adapters: TorchtuneAdapter, LlamaFactoryAdapter
```

### VectorStorePort

```python
from abc import ABC, abstractmethod
from core.entities import Document, SearchResult

class VectorStorePort(ABC):
    @abstractmethod
    async def upsert(self, docs: list[Document]) -> None: ...
    @abstractmethod
    async def semantic_search(
        self, query_vec: list[float], k: int, filter: dict | None = None
    ) -> list[SearchResult]: ...
    @abstractmethod
    async def hybrid_search(
        self, query: str, query_vec: list[float], k: int, alpha: float = 0.7
    ) -> list[SearchResult]: ...
    # alpha: 0 = pure BM25, 1 = pure vector
    @abstractmethod
    async def delete(self, ids: list[str]) -> None: ...
# Adapter: PgVectorAdapter (pgvector HNSW + pg_trgm GIN)
```

### GraphStorePort

```python
from abc import ABC, abstractmethod
from core.entities import GraphContext

class GraphStorePort(ABC):
    @abstractmethod
    async def add_node(self, label: str, props: dict) -> str: ...
    @abstractmethod
    async def add_edge(
        self, src_id: str, dst_id: str, rel_type: str, props: dict = {}
    ) -> None: ...
    @abstractmethod
    async def cypher_query(self, query: str, params: dict = {}) -> list[dict]: ...
    @abstractmethod
    async def graphrag_retrieve(self, query: str, hops: int = 2) -> GraphContext: ...
    @abstractmethod
    async def delete_node(self, node_id: str) -> None: ...
# Adapters: ApacheAGEAdapter, Neo4jAdapter (optional)
```

### SkillRegistryPort

```python
from abc import ABC, abstractmethod
from core.entities import SkillManifest, Skill, SkillResult

class SkillRegistryPort(ABC):
    @abstractmethod
    async def list_skills(self) -> list[SkillManifest]: ...
    @abstractmethod
    async def get_skill(self, name: str) -> Skill: ...
    @abstractmethod
    async def install_skill(self, path_or_url: str) -> SkillManifest: ...
    @abstractmethod
    async def invoke_skill(self, name: str, context: dict) -> SkillResult: ...
    @abstractmethod
    async def uninstall_skill(self, name: str) -> None: ...
# Adapter: MCPFilesystemAdapter
# Scans: ~/.forge/skills/public, ~/.forge/skills/user, ~/.forge/skills/private
```

-----

## 9. Frontend Architecture

### Stack

|Library       |Version|Usage                                          |
|--------------|-------|-----------------------------------------------|
|React         |19     |Core framework                                 |
|TypeScript    |5.x    |All source code                                |
|@xyflow/react |12.x   |Model builder, Pipeline Studio, KG graph editor|
|Zustand       |5      |Global UI state (mode, project, config)        |
|TanStack Query|v5     |Server state, caching, WebSocket subscriptions |
|Monaco Editor |latest |Code editor (YAML, Python export)              |
|Recharts      |2.x    |Training curves, eval charts                   |
|D3            |7.x    |Attention heatmaps, custom visualizations      |
|Tailwind CSS  |v4     |Styling                                        |
|shadcn/ui     |latest |Base component system (unstyled + accessible)  |
|Vite          |6      |Build tool                                     |
|Tauri         |v2     |Desktop packaging                              |

### Application Shell

```
┌─────────────────────────────────────────────────────────────────┐
│  TITLEBAR  [FORGE]   [Project: my-project]   [GUIDED ⟷ EXPERT] │
├──────────┬──────────────────────────────────┬───────────────────┤
│          │                                  │                   │
│  SIDEBAR │     WORKSPACE CANVAS             │   CONTEXT PANEL   │
│          │                                  │                   │
│  🔬 Forge│  Module-specific content:         │  Mode-aware:      │
│  🔗 Pipe │  - @xyflow graph editors          │  - Lab Assistant  │
│  📦 Data │  - Training dashboard             │  - Config form    │
│  ⚡ Train│  - Playground chat                │  - Docs / props   │
│  🧪 Eval │  - Dataset tables                 │  - Code export    │
│  💬 Play │  - Research results               │                   │
│  🕸️ RAG   │                                  │                   │
│  📜 Reg  │                                  │                   │
│  🔍 Res  │                                  │                   │
│          │                                  │                   │
├──────────┴──────────────────────────────────┴───────────────────┤
│  STATUS BAR  GPU: 18.2/24GB  │  Active model: llama3.2  │  Runs │
└─────────────────────────────────────────────────────────────────┘
```

### Dual-Mode Pattern

The mode toggle is global Zustand state (`mode: 'guided' | 'expert'`). Components receive mode as a prop and render different variants. There is no duplicated business logic — only different renders of the same data.

```typescript
// Mode is a render variant, not a feature gate
function TrainingConfigForm({ mode }: { mode: 'guided' | 'expert' }) {
  if (mode === 'guided') return <GuidedTrainingWizard />;
  return <ExpertYAMLEditor />;
}
```

### @xyflow Custom Node Types

**Pipeline Studio nodes:** `LLMNode` (Predict/CoT/ReAct), `RetrieverNode` (Vector/Graph/Hybrid/BM25), `ToolNode` (web search, MCP skill), `RouterNode` (conditional), `AggregatorNode`, `InputNode`, `OutputNode`

**Model Builder nodes:** `LayerNode` (attention, FFN, norm, embedding), `AdapterNode` (LoRA, QLoRA, prefix tuning), `BaseModelNode` (HuggingFace model reference), `TemplateNode` (preset architectures)

-----

## 10. DSPy Engine Design

### Core Usage

FORGE uses DSPy 2.x for all LLM pipeline logic.

```python
import dspy

# Configure LM (user-selected at setup)
lm = dspy.LM("ollama_chat/llama3.2", api_base="http://localhost:11434")
dspy.configure(lm=lm)

# Pipeline Studio exports generate code like this:
class GeneratedRAGPipeline(dspy.Module):
    def __init__(self):
        self.retrieve = HybridRetriever(k=5)
        self.reason = dspy.ChainOfThought("context, question -> answer")
        self.judge = dspy.Predict("answer -> score: float, reasoning: str")

    def forward(self, question: str):
        context = self.retrieve(question)
        result = self.reason(context=context, question=question)
        judgment = self.judge(answer=result.answer)
        return dspy.Prediction(
            answer=result.answer,
            score=judgment.score,
            reasoning=judgment.reasoning
        )
```

### MIPROv2 Optimization Flow

1. User selects evaluation dataset + metric function in UI
1. FORGE configures `dspy.MIPROv2(auto="medium")` with user’s trainset
1. Bootstrap stage: collect traces from training examples
1. Proposal stage: LLM generates grounded candidate instructions
1. Bayesian search: evaluate instruction/demonstration combinations on mini-batches
1. Best program state serialized → `pipelines.optimized_state` (JSONB)
1. Code export updated with optimized prompts baked in

```python
optimizer = dspy.MIPROv2(metric=user_metric, auto="medium")
optimized_program = optimizer.compile(
    pipeline,
    trainset=trainset,
    num_trials=20,
    max_bootstrapped_demos=4
)
state = optimized_program.dump_state()  # → persisted to DB
```

### Lab Assistant Agent

```python
forge_tools = [
    get_hardware_profile,
    list_available_base_models,
    explain_architecture_choice,
    check_dataset_quality,
    estimate_training_cost,
    search_arxiv,
    search_huggingface,
    invoke_skill,
]

lab_assistant = dspy.ReAct(
    "user_goal, project_context, hardware_profile -> guidance, suggested_actions",
    tools=forge_tools,
    max_iters=5
)
```

### Pipeline → Code Compiler

The graph IR compiler walks the `@xyflow` node graph JSON, maps node types to DSPy constructs, resolves edge connections as data dependencies, and emits a valid `dspy.Module` subclass. The IR format is FORGE-defined JSON — independent of `@xyflow`’s internal format. `@xyflow` is the renderer only; the IR is the source of truth.

-----

## 11. Training System

### Hardware-Aware Advisor

At startup and before every training run, FORGE detects hardware using `py-nvml`, `psutil`, and `torch.cuda`. Options are gated based on available VRAM:

|VRAM   |Available                          |Gated (warning shown)      |
|-------|-----------------------------------|---------------------------|
|< 4GB  |CPU inference only, evaluation     |All training, GPU inference|
|4–8GB  |QLoRA 3B–7B, GGUF inference, Ollama|Full FT, LoRA 13B+, vLLM   |
|8–16GB |LoRA 7B–13B, QLoRA 30B, vLLM 7B    |Full FT 13B+, QLoRA 70B    |
|16–24GB|LoRA 13B–34B, QLoRA 70B, full FT 7B|Full FT 13B+               |
|24GB+  |Full FT 13B, LoRA 70B, vLLM 34B+   |Nothing blocked            |

### Torchtune Adapter

PyTorch-native. FORGE generates YAML configs programmatically then calls `tune run` as a managed subprocess. Supported methods: `lora_finetune_single_device`, `qlora_finetune_single_device`, `full_finetune_single_device`, `full_finetune_distributed`, `dpo_full_finetune_single_device`, `knowledge_distillation_single_device`.

### LLaMA-Factory Adapter

Broadest model coverage (100+ models including Llama 4, Qwen3, InternVL3, multimodal). FORGE calls its Python API or CLI. Extra methods: OFT/OFTv2, APOLLO/Muon optimizers, multi-modal training, SGLang inference.

### Training Run Lifecycle

```
User configures run in UI
    → Hardware Advisor validates feasibility
    → Config snapshot written to training_runs table
    → Adapter spawns subprocess (torchtune / llamafactory)
    → Metrics streamed: stdout parser → WebSocket → React UI
    → TimescaleDB hypertable receives metric rows in real time
    → On completion: output_path recorded, status = 'completed'
    → Experiment Registry auto-links run
    → Export wizard offered (GGUF / ONNX)
```

### GGUF / ONNX Export Wizard

- **GGUF**: `llama.cpp`’s `convert_hf_to_gguf.py` + quantization (Q4_K_M recommended for most users). One-click load into Ollama after export.
- **ONNX**: HuggingFace `optimum` library. Wizard selects quantization level based on target deployment VRAM.

-----

## 12. RAG & Knowledge Graph

### Retrieval Modes

All backed by PostgreSQL 16 extensions. All pipelines are DSPy programs under the hood.

**Hybrid RAG (Dense + Sparse):** pgvector HNSW for ANN + pg_trgm GIN for BM25. Results merged with Reciprocal Rank Fusion (RRF). `alpha` parameter (0–1) controls dense vs sparse weight.

**GraphRAG (KG-Augmented):** Apache AGE graph. Entity extraction → nodes, relation extraction → edges. Multi-hop Cypher traversal at query time (default 2 hops).

**Agentic RAG (Multi-hop, Self-correcting):** `dspy.ReAct` agent with retrieval tools. Issues follow-up queries if first retrieval insufficient. 2–5 hops, configurable.

**Multi-modal RAG:** PDF via `pdfplumber`, images via CLIP embeddings, tables extracted to structured JSON. Returns mixed-type context.

### Ingestion Pipeline

```
Input (PDF / HTML / CSV / DOCX / images)
    → unstructured (layout-aware parsing)
    → chunking (sliding window, sentence-aware)
    → pgai → local Ollama embedding model → pgvector upsert
    → pg_trgm GIN index update
    → DSPy NER pipeline → Apache AGE nodes
    → DSPy RE pipeline → Apache AGE edges
```

-----

## 13. MCP & Skills Filesystem

### Directory Layout

```
~/.forge/
├── skills/
│   ├── public/          # Built-in FORGE skills (read-only)
│   │   ├── rag-basic/
│   │   │   ├── SKILL.md
│   │   │   └── forge-adapter.json
│   │   ├── lora-guide/
│   │   ├── dataset-cleaner/
│   │   └── eval-harness/
│   ├── user/            # User-installed skills
│   └── private/         # Project-scoped private skills
├── plugins/             # Third-party adapter plugins
├── models/              # Downloaded model weights
├── datasets/            # Versioned dataset files
│   └── <dataset-id>/
│       ├── v1.0.0.jsonl
│       └── v1.0.0.meta.json
├── runs/                # Training run artifacts
│   └── <run-id>/
│       ├── config.yaml
│       ├── checkpoints/
│       └── final/
└── exports/             # GGUF / ONNX exported models
```

### SKILL.md Format

```markdown
---
name: my-skill
description: What this skill does and when to use it
port: VectorStorePort
version: 1.0.0
---

# Skill documentation
Full description of the skill's capabilities, usage patterns, and configuration.
```

The `MCPFilesystemAdapter` scans all three skill directories at startup, parses `SKILL.md` + `forge-adapter.json`, and registers valid skills in the `skills` PostgreSQL table.

-----

## 14. Complete Tech Stack

### Frontend

- React 19 + TypeScript 5.x
- @xyflow/react 12.x (visual graph editors)
- Zustand 5 (global state) + TanStack Query v5 (server state + WebSockets)
- Monaco Editor (YAML / Python code editing)
- Recharts 2.x + D3 7.x (charts + custom viz)
- Tailwind CSS v4 + shadcn/ui (components)
- Vite 6 (build) + Tauri v2 (desktop)

### Backend

- FastAPI + Pydantic v2 + Uvicorn
- WebSockets (real-time training metrics + inference streaming)
- Alembic (migrations)
- py-nvml + psutil + torch.cuda (hardware detection)
- OmegaConf (YAML config management)

### AI / ML

- DSPy 2.x (MIPROv2, ReAct, ChainOfThought, BootstrapFinetune)
- Torchtune (PyTorch-native fine-tuning)
- LLaMA-Factory (broad model coverage)
- Ollama (primary local inference) + llama.cpp (CPU + GGUF export)
- vLLM (high-throughput, Phase 4)
- huggingface_hub SDK (model downloads) + HuggingFace Optimum (ONNX export)
- lm-evaluation-harness (benchmarks)
- unstructured + pdfplumber (document parsing)
- pgai (embedding generation from PostgreSQL)

### Data / Infrastructure

- PostgreSQL 16 with pgvector, Apache AGE 1.5.0, pg_trgm, pgai, TimescaleDB
- Docker Compose (container orchestration)

-----

## 15. Monorepo Structure

```
forge/
├── forge-backend/
│   ├── core/
│   │   ├── entities/              # Domain entities (pure Python dataclasses)
│   │   ├── ports/                 # Abstract Port ABCs
│   │   └── use_cases/             # Business logic
│   ├── adapters/
│   │   ├── inference/             # OllamaAdapter, LlamaCppAdapter, VLLMAdapter
│   │   ├── training/              # TorchtuneAdapter, LlamaFactoryAdapter
│   │   ├── storage/               # PgVectorAdapter, ApacheAGEAdapter, PostgreSQLAdapter
│   │   └── skills/                # MCPFilesystemAdapter
│   ├── api/
│   │   ├── routers/               # FastAPI route handlers per module
│   │   ├── websockets/            # WebSocket handlers
│   │   └── deps.py                # Dependency injection
│   ├── dspy_engine/
│   │   ├── pipeline_compiler.py   # Graph IR → DSPy code
│   │   ├── optimization.py        # MIPROv2 orchestration
│   │   ├── lab_assistant.py       # ReAct agent
│   │   └── rag/                   # RAG pipeline implementations
│   ├── services/
│   │   ├── hardware_advisor.py
│   │   ├── experiment_registry.py
│   │   ├── model_card_generator.py
│   │   └── research_aggregator.py
│   ├── db/
│   │   ├── migrations/            # Alembic
│   │   └── models.py              # SQLAlchemy models
│   └── main.py
│
├── forge-frontend/
│   ├── src/
│   │   ├── modules/
│   │   │   ├── model-forge/
│   │   │   ├── pipeline-studio/
│   │   │   ├── data-lab/
│   │   │   ├── training-center/
│   │   │   ├── eval-engine/
│   │   │   ├── playground/
│   │   │   ├── rag-studio/
│   │   │   ├── experiment-registry/
│   │   │   ├── research-lab/
│   │   │   └── lab-assistant/
│   │   ├── components/
│   │   │   ├── nodes/             # @xyflow custom node types
│   │   │   ├── charts/            # Recharts + D3 components
│   │   │   └── ui/                # shadcn/ui components
│   │   ├── store/                 # Zustand stores
│   │   ├── api/                   # TanStack Query hooks + API client
│   │   └── types/
│   └── vite.config.ts
│
├── forge-docker/
│   ├── docker-compose.yml
│   ├── Dockerfile.backend
│   ├── Dockerfile.postgres        # PG16 + all extensions built from source
│   └── init.sql
│
├── forge-skills/                  # Built-in public skills
│   ├── rag-basic/
│   ├── lora-guide/
│   ├── dataset-cleaner/
│   └── eval-harness/
│
├── forge-desktop/                 # Tauri v2
│   ├── src-tauri/
│   └── tauri.conf.json
│
└── tests/
    ├── test_adapters/
    ├── test_use_cases/
    └── test_api/
```

-----

## 16. Phased Build Plan

### Phase 1 — Foundation (Months 1–3)

**Deliverable: Working AI playground + model manager**

- FastAPI skeleton + WebSocket infrastructure
- PostgreSQL 16 + all extensions setup (Alembic migrations)
- `InferencePort` → OllamaAdapter + LlamaCppAdapter
- Hardware detection service
- `SkillRegistryPort` → MCPFilesystemAdapter
- Project / session management
- React app shell (sidebar, workspace, context panel)
- GUIDED / EXPERT mode toggle
- Playground module (chat, batch, comparison)
- Hardware Advisor status bar widget
- Model Manager (HF Hub download + local list)
- Lab Assistant (DSPy ReAct, basic tools)

### Phase 2 — Data + Training (Months 3–6)

**Deliverable: End-to-end fine-tuning workflow**

- `DatasetManagerPort` + ingestion pipeline
- Visual data cleaning (dedup, filter, stats)
- SFT/DPO/RLHF/ORPO auto-formatter
- `TrainingBackendPort` → TorchtuneAdapter + LlamaFactoryAdapter
- TimescaleDB metrics streaming
- Experiment Registry
- GGUF/ONNX Export Wizard
- Data Lab module UI
- Synthetic data generator
- Training Center (guided wizard + expert YAML)
- Live training dashboard
- Eval Engine (custom metrics, A/B, W&B export)

### Phase 3 — Visual Builders + DSPy (Months 6–9)

**Deliverable: Visual pipeline builder + RAG studio**

- Graph IR → PyTorch/DSPy code compiler
- `VectorStorePort` → PgVectorAdapter (hybrid)
- `GraphStorePort` → ApacheAGEAdapter
- DSPy pipeline executor + MIPROv2 orchestration
- GraphRAG + Agentic RAG backends
- Multi-modal ingestion pipeline
- Model Forge UI (@xyflow)
- Pipeline Studio UI (@xyflow + Monaco export)
- RAG/KG Studio (ingestion, testing, KG visualizer)

### Phase 4 — Research + Polish (Months 9–12)

**Deliverable: Full V1 — Tauri app + Docker release**

- Research Aggregator (ArXiv + HF DSPy agent)
- Model Card generator + HF Hub push
- LLM-as-judge eval pipelines
- VLLMAdapter
- Tauri v2 desktop build
- Docker Compose production hardening
- Research Lab module
- Token probability + attention visualizer
- Model Card wizard
- Plugin Manager UI
- Onboarding flow

-----

## 17. Risks & Mitigations

|Risk                             |Severity|Mitigation                                                                                                   |
|---------------------------------|--------|-------------------------------------------------------------------------------------------------------------|
|Apache AGE stability on PG16     |Medium  |Pin AGE 1.5.0. Integration tests in CI. Neo4jAdapter ready as fallback.                                      |
|DSPy API churn                   |Medium  |Pin `dspy==2.x.y`. Wrap in `DSPyExecutor` service class.                                                     |
|Training subprocess crashes      |Medium  |Isolated subprocesses, heartbeat monitoring, FAILED status + error capture.                                  |
|@xyflow graph IR versioning      |Low     |IR is FORGE-defined JSON, not xyflow-native. Migrations independent of xyflow upgrades.                      |
|Large model downloads blocking UX|Low     |Background worker + progress streaming + resumable via huggingface_hub.                                      |
|Dual-mode UX divergence          |High    |Mode is a render prop, not two feature trees. One data model, two view variants. User-test both from Phase 1.|
|PG extension build conflicts     |Low     |Pre-built Docker image with all extensions compiled from source against PG 16.x.                             |
|GPU OOM during training          |Medium  |Hardware Advisor gates configs before run. Automatic gradient checkpointing suggestion.                      |

-----

## 18. Development Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker + Docker Compose
- CUDA-capable GPU (optional — CPU mode supported via llama.cpp)
- Rust toolchain (for Tauri desktop build only)

### Quick Start (Docker)

```bash
git clone https://github.com/your-org/forge
cd forge
docker compose up -d
cd forge-frontend && npm install && npm run dev
```

### Backend Dev

```bash
cd forge-backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
alembic upgrade head
uvicorn main:app --reload --port 8000
```

### Environment Variables

```env
DATABASE_URL=postgresql://forge:forge@localhost:5432/forge
OLLAMA_BASE_URL=http://localhost:11434
LLAMACPP_SERVER_URL=http://localhost:8080
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
HF_TOKEN=
FORGE_SKILLS_DIR=~/.forge/skills
FORGE_MODELS_DIR=~/.forge/models
HARDWARE_CHECK_INTERVAL_SECONDS=30
```

### Adding a New Adapter

1. Create a Python class implementing the relevant Port ABC in `forge-backend/adapters/`
1. Add a `forge-adapter.json` manifest
1. Write unit tests against the port contract in `tests/test_adapters/`
1. Add a `SKILL.md` describing capabilities
1. Register in the adapter registry at startup

-----

*FORGE Architecture Document — Version 1.0*
