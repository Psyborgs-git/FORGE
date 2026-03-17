/**
 * FORGE frontend types
 */

export interface HardwareProfile {
  gpu_count: number;
  gpu_names: string[];
  gpu_vram_total_gb: number[];
  gpu_vram_free_gb: number[];
  cuda_available: boolean;
  cuda_version: string | null;
  cpu_count: number;
  cpu_model: string;
  ram_total_gb: number;
  ram_available_gb: number;
  detected_at: number;
}

export interface FeasibleOps {
  can_train_full_ft: boolean;
  can_train_lora: boolean;
  can_train_qlora: boolean;
  max_trainable_params_b: number;
  can_use_gpu_inference: boolean;
  can_use_vllm: boolean;
  recommended_inference: string;
  warnings: string[];
}

export interface Project {
  id: string;
  name: string;
  mode: 'guided' | 'expert';
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface ModelInfo {
  name: string;
  size?: number;
  parameter_count?: string;
  quantization?: string;
  family?: string;
  modified_at?: string;
  metadata: Record<string, unknown>;
}

export interface InferenceMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface InferenceRequest {
  model: string;
  messages: InferenceMessage[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  stop?: string[];
  system?: string;
}

export interface InferenceResponse {
  model: string;
  content: string;
  finish_reason?: string;
  total_tokens?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  metadata: Record<string, unknown>;
}
