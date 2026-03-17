-- FORGE Database Initialization Script
-- PostgreSQL 16 with pgvector, Apache AGE, pg_trgm, pgai, TimescaleDB

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgvector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS ai CASCADE;

-- Set search path to include AGE
SET search_path = ag_catalog, "$user", public;

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
  project_id      UUID REFERENCES projects(id) ON DELETE CASCADE,
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
  project_id    UUID REFERENCES projects(id) ON DELETE CASCADE,
  name          TEXT NOT NULL,
  version       TEXT NOT NULL,
  format        TEXT NOT NULL,
  row_count     INTEGER,
  hash_sha256   TEXT NOT NULL,
  storage_path  TEXT NOT NULL,
  stats         JSONB DEFAULT '{}',
  created_at    TIMESTAMPTZ DEFAULT now(),
  UNIQUE(project_id, name, version)
);

-- Training Runs
CREATE TABLE training_runs (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id        UUID REFERENCES projects(id) ON DELETE CASCADE,
  dataset_id        UUID REFERENCES datasets(id) ON DELETE SET NULL,
  architecture_id   UUID REFERENCES model_architectures(id) ON DELETE SET NULL,
  backend           TEXT NOT NULL,
  method            TEXT NOT NULL,
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
  run_id        UUID REFERENCES training_runs(id) ON DELETE CASCADE,
  time          TIMESTAMPTZ NOT NULL,
  step          INTEGER NOT NULL,
  train_loss    FLOAT8,
  eval_loss     FLOAT8,
  learning_rate FLOAT8,
  gpu_mem_gb    FLOAT4,
  throughput    FLOAT4
);

-- Convert training_metrics to hypertable
SELECT create_hypertable('training_metrics', 'time', if_not_exists => TRUE);

-- DSPy Pipelines
CREATE TABLE pipelines (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id      UUID REFERENCES projects(id) ON DELETE CASCADE,
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
  project_id  UUID REFERENCES projects(id) ON DELETE CASCADE,
  content     TEXT NOT NULL,
  vector      VECTOR(1536),
  source_type TEXT,
  source_id   UUID,
  metadata    JSONB DEFAULT '{}'
);

-- Create HNSW index on embeddings vector
CREATE INDEX embeddings_vector_idx ON embeddings USING hnsw (vector vector_cosine_ops);

-- Create GIN index on embeddings content for full-text search
CREATE INDEX embeddings_content_idx ON embeddings USING gin (content gin_trgm_ops);

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
  project_id    UUID REFERENCES projects(id) ON DELETE CASCADE,
  name          TEXT NOT NULL,
  description   TEXT,
  run_ids       UUID[],
  pipeline_ids  UUID[],
  dataset_ids   UUID[],
  final_metrics JSONB DEFAULT '{}',
  tags          TEXT[],
  created_at    TIMESTAMPTZ DEFAULT now()
);

-- Create indexes for common queries
CREATE INDEX idx_projects_created_at ON projects(created_at DESC);
CREATE INDEX idx_datasets_project_id ON datasets(project_id);
CREATE INDEX idx_training_runs_project_id ON training_runs(project_id);
CREATE INDEX idx_training_runs_status ON training_runs(status);
CREATE INDEX idx_pipelines_project_id ON pipelines(project_id);
CREATE INDEX idx_embeddings_project_id ON embeddings(project_id);
CREATE INDEX idx_experiments_project_id ON experiments(project_id);

-- Grant permissions to forge user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO forge;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO forge;
