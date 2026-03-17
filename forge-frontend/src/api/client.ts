import axios from 'axios';
import type {
  Project,
  HardwareProfile,
  FeasibleOps,
  ModelInfo,
  InferenceRequest,
  InferenceResponse,
} from '@/types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Projects API
export const projectsApi = {
  list: () => client.get<Project[]>('/api/projects'),
  get: (id: string) => client.get<Project>(`/api/projects/${id}`),
  create: (data: { name: string; mode?: string; metadata?: Record<string, unknown> }) =>
    client.post<Project>('/api/projects', data),
  update: (id: string, data: Partial<Project>) =>
    client.put<Project>(`/api/projects/${id}`, data),
  delete: (id: string) => client.delete(`/api/projects/${id}`),
};

// Hardware API
export const hardwareApi = {
  getProfile: () => client.get<HardwareProfile>('/api/hardware/profile'),
  getFeasibleOps: () => client.get<FeasibleOps>('/api/hardware/feasible-ops'),
};

// Inference API
export const inferenceApi = {
  generate: (request: InferenceRequest) =>
    client.post<InferenceResponse>('/api/inference/generate', request),
  embed: (texts: string[], model?: string) =>
    client.post<{ embeddings: number[][]; model: string }>('/api/inference/embed', {
      texts,
      model,
    }),
  listModels: () => client.get<ModelInfo[]>('/api/inference/models'),
  healthCheck: () => client.get<{ healthy: boolean; backend: string }>('/api/inference/health'),
};

// Skills API
export const skillsApi = {
  list: () => client.get('/api/skills/'),
  get: (name: string) => client.get(`/api/skills/${name}`),
  install: (path: string) => client.post('/api/skills/install', { path_or_url: path }),
  invoke: (name: string, context: Record<string, unknown>) =>
    client.post(`/api/skills/${name}/invoke`, { context }),
  uninstall: (name: string) => client.delete(`/api/skills/${name}`),
};

export default client;
