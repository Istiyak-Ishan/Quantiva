import axios from 'axios';
import {
  getBackendConnectionInfo,
  createDesktopSSE,
  fetchEventSource,
  type DesktopEventSourceLike,
} from './api-client';

// ─── AXIOS INSTANCE ──────────────────────────────────────────────────────────
// Single Axios instance with base URL. In desktop or dev mode, the base URL
// and X-Quantiva-Token are injected dynamically by the request interceptor.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? '/api/v1';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30_000,
  withCredentials: true,
});

// Dynamic Desktop & Dev Token Request Interceptor
api.interceptors.request.use(async (config) => {
  const info = await getBackendConnectionInfo();
  if (!config.baseURL || config.baseURL === '/api/v1' || config.baseURL.endsWith('/api/v1')) {
    config.baseURL = `${info.baseUrl}/api/v1`;
  }
  if (info.ipcToken && !config.headers.get('X-Quantiva-Token')) {
    config.headers.set('X-Quantiva-Token', info.ipcToken);
  }
  return config;
});

// Browser credentials live only in backend-issued HttpOnly cookies. Serialize
// refreshes so parallel 401s do not replay a rotating refresh credential.
let refreshing: Promise<unknown> | null = null;
api.interceptors.response.use((response) => response, async (error) => {
  const config = error.config as (typeof error.config & { _retried?: boolean }) | undefined;
  if (error.response?.status === 401 && config && !config._retried && !config.url?.startsWith('/auth/')) {
    config._retried = true;
    refreshing ??= api.post('/auth/refresh').finally(() => { refreshing = null; });
    await refreshing;
    return api.request(config);
  }
  throw error;
});

export interface CurrentUser { id: string; email: string; tier: string; }
export const getCurrentUser = async (): Promise<CurrentUser> => {
  try {
    return (await api.get<CurrentUser>('/auth/me')).data;
  } catch (error) {
    if (!axios.isAxiosError(error) || error.response?.status !== 401) throw error;
    refreshing ??= api.post('/auth/refresh').finally(() => { refreshing = null; });
    await refreshing;
    return (await api.get<CurrentUser>('/auth/me')).data;
  }
};
export const authenticate = async (mode: 'login' | 'register', email: string, password: string) => {
  await api.post(`/auth/${mode}`, { email, password });
};
export const logout = async () => { await api.post('/auth/logout'); };
export const errorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map((item: { msg?: string }) => item.msg ?? 'Invalid input').join('; ');
  }
  return 'Unable to contact Quantiva. Please retry.';
};

// ─── TYPES ───────────────────────────────────────────────────────────────────
export interface DocumentUploadResponse {
  id: string;
  filename: string;
  status: 'processing' | 'ready' | 'failed';
}

export interface DocumentStatusResponse {
  id: string;
  status: 'processing' | 'ready' | 'failed';
  chunk_count: number | null;
  failure_reason?: string | null;
}

export interface SessionResponse {
  session_id: string;
}

export interface Citation {
  doc_id: string;
  filename: string;
  page_or_row: number;
  text_excerpt: string;
}

export interface BackendMessage {
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  confidence?: number;
  chart_url?: string;
  created_at?: string;
  guardrail_flags?: string[];
  cache_hit?: boolean;
}

export interface StreamEventData {
  token: string;
  done: boolean;
  citations?: Citation[];
  confidence?: number;
  chart_url?: string;
  tokens_used?: number;
  answer?: string;
  cost_usd?: number | null;
  guardrail_flags?: string[];
  cache_hit?: boolean;
}

// ─── DOCUMENT API ─────────────────────────────────────────────────────────────

export interface DocumentItem {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  status: 'pending' | 'uploading' | 'processing' | 'ready' | 'failed';
  chunk_count: number | null;
  failure_reason?: string | null;
  created_at: string;
}

/**
 * Fetch all documents owned by the user.
 */
export const getDocuments = async (): Promise<DocumentItem[]> => {
  const response = await api.get<DocumentItem[]>('/documents');
  return response.data;
};

/**
 * Delete a document and its chunks from backend.
 */
export const deleteDocument = async (docId: string): Promise<void> => {
  await api.delete(`/documents/${docId}`);
};

/**
 * Re-queue a failed (or previously ready) document for indexing.
 */
export const reindexDocument = async (docId: string): Promise<DocumentUploadResponse> => {
  const response = await api.post<DocumentUploadResponse>(`/documents/${docId}/reindex`);
  return response.data;
};

export const openDocument = async (docId: string) => {
  const tab = window.open('about:blank', '_blank');
  if (tab) tab.opener = null;
  try {
    const response = await api.get(`/documents/${docId}/download`, { responseType: 'blob' });
    const url = URL.createObjectURL(response.data);
    if (tab) tab.location.href = url;
    else { const link = document.createElement('a'); link.href = url; link.download = 'document'; link.click(); }
    window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
  } catch (error) {
    tab?.close();
    throw error;
  }
};

/**
 * Upload a document file to the backend.
 * Returns immediately with document ID and 'processing' status.
 * Use `watchDocumentStatus` to get notified when indexing finishes.
 */
export const uploadDocument = async (
  file: File,
  onProgress?: (percent: number) => void,
): Promise<DocumentUploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<DocumentUploadResponse>(
    '/documents/upload',
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (evt) => {
        if (onProgress && evt.total) {
          onProgress(Math.round((evt.loaded / evt.total) * 100));
        }
      },
    },
  );
  return response.data;
};

/**
 * Subscribe to SSE status updates for a document while it indexes.
 *
 * Usage:
 *   const es = watchDocumentStatus(docId, (status) => {
 *     if (status.status === 'ready') doSomething()
 *   })
 *   // cleanup: es.close()
 */
export const watchDocumentStatus = (
  docId: string,
  onStatus: (data: DocumentStatusResponse) => void,
  onError?: () => void,
): EventSource | DesktopEventSourceLike => {
  const sse = createDesktopSSE(`/documents/${docId}/status`);

  sse.onmessage = (e) => {
    try {
      const data: DocumentStatusResponse = JSON.parse(e.data);
      onStatus(data);
      if (data.status === 'ready' || data.status === 'failed') {
        sse.close();
      }
    } catch {
      // ignore malformed events
    }
  };

  sse.onerror = () => {
    onError?.();
    sse.close();
  };

  return sse;
};

// ─── SESSION API ──────────────────────────────────────────────────────────────

/**
 * Create a new chat session, optionally pre-seeded with document IDs.
 * The backend returns a session_id (UUID) for all future messages.
 */
export const createSession = async (
  documentIds: string[] = [],
): Promise<SessionResponse> => {
  const response = await api.post<SessionResponse>('/chat/sessions', {
    document_ids: documentIds,
  });
  return response.data;
};

/**
 * Fetch all sessions for the current user.
 */
export const getSessions = async () => {
  const response = await api.get('/chat/sessions');
  return response.data as Array<{
    id: string;
    title: string | null;
    created_at: string;
  }>;
};

/**
 * Delete a session and clear its Redis memory.
 */
export const deleteSession = async (sessionId: string) => {
  await api.delete(`/chat/sessions/${sessionId}`);
};

// ─── CHAT API ─────────────────────────────────────────────────────────────────

/**
 * Fetch message history for a session from Redis hot storage.
 * Returns an array of BackendMessage objects.
 */
export const getHistory = async (
  sessionId: string,
): Promise<BackendMessage[]> => {
  const response = await api.get<BackendMessage[]>(
    `/chat/sessions/${sessionId}/history`,
  );
  return response.data;
};

/**
 * Open a streaming SSE connection to get a token-by-token LLM response.
 *
 * The backend endpoint uses HttpOnly cookie authentication, never URL credentials.
 *
 * Each SSE event contains a StreamEventData JSON payload:
 *   { token: "word ", done: false }          — intermediate
 *   { token: "", done: true, citations: [] } — final
 *
 * Usage:
 *   const es = await streamMessage(sessionId, query)
 *   es.onmessage = (e) => { ... }
 *   // cleanup: es.close()
 */
export const streamMessage = async (
  sessionId: string,
  query: string,
): Promise<EventSource | DesktopEventSourceLike> => {
  try {
    await getCurrentUser(); // Refresh if necessary before opening a non-retrying SSE request.
  } catch {
    // In desktop mode without user auth, continue
  }

  const queryParams = new URLSearchParams({ session_id: sessionId, query });
  return createDesktopSSE(`/chat/message/stream?${queryParams.toString()}`);
};

// ─── DESKTOP SYSTEM, HARDWARE & MODELS API ────────────────────────────────────

export interface SystemHealth {
  status: string;
  app_version: string;
  startup_status: string;
  sqlite_status: string;
  vector_store_status: string;
  data_directory: string;
  database_path: string;
}

export interface GPUInfo {
  name: string;
  vram_total_mb: number;
  vram_total_gb: number;
  vram_free_mb?: number | null;
}

export interface HardwareProfile {
  cpu_model: string;
  physical_cores: number;
  logical_cores: number;
  ram_total_gb: number;
  ram_available_gb: number;
  disk_free_gb: number;
  gpus: GPUInfo[];
  detection_errors: string[];
}

export interface ModelTierRecommendation {
  tier: string;
  description: string;
  llm_models: string[];
  embedding_models: string[];
  reranker_models: string[];
  rationale: string;
}

export interface SystemHardware {
  hardware: HardwareProfile;
  recommendation: ModelTierRecommendation;
}

export interface OllamaModelInfo {
  name: string;
  size_bytes: number;
  family: string;
  parameter_size: string;
  quantization_level: string;
}

export interface OllamaStatus {
  is_installed: boolean;
  is_running: boolean;
  binary_path?: string | null;
  base_url: string;
  version?: string | null;
  models: OllamaModelInfo[];
}

export interface LLMConnectionTestResult {
  success: boolean;
  latency_ms: number;
  provider: string;
  model: string;
  capabilities?: {
    supports_streaming: boolean;
    supports_function_calling: boolean;
    supports_structured_output: boolean;
    supports_vision: boolean;
    supports_system_prompt: boolean;
    max_context_window: number;
    is_local: boolean;
  } | null;
  error_code?: string | null;
  error_message?: string | null;
}

export interface EmbeddingTestResult {
  success: boolean;
  latency_ms: number;
  provider: string;
  model: string;
  detected_dimensions?: number | null;
  error_code?: string | null;
  error_message?: string | null;
}

export interface LLMConfig {
  id: string;
  name: string;
  provider_type: string;
  model_name: string;
  base_url?: string | null;
  has_api_key: boolean;
  temperature: number;
  max_tokens: number;
  context_window?: number | null;
  capabilities_json?: Record<string, unknown>;
  is_default: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface EmbeddingConfig {
  id: string;
  name: string;
  provider_type: string;
  model_name: string;
  base_url?: string | null;
  has_api_key: boolean;
  dimensions: number;
  batch_size: number;
  is_default: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface RerankerTestResult {
  success: boolean;
  latency_ms: number;
  provider: string;
  model: string;
  error_code?: string | null;
  error_message?: string | null;
}

export interface RetrievalConfig {
  id: string;
  name: string;
  search_mode: string;
  dense_top_k: number;
  sparse_top_k: number;
  rrf_k: number;
  dense_weight: number;
  sparse_weight: number;
  enable_reranking: boolean;
  reranker_provider?: string | null;
  reranker_model?: string | null;
  final_top_k: number;
  enable_agentic_mode: boolean;
  max_agent_iterations: number;
  is_default: boolean;
}

export interface KnowledgeBaseItem {
  id: string;
  name: string;
  description?: string | null;
  embedding_config_id?: string | null;
  retrieval_config_id?: string | null;
  created_at: string;
  updated_at: string;
  document_count?: number;
  chunk_count?: number;
  vector_collection_name?: string;
  embedding_dimensions?: number;
  is_healthy?: boolean;
}

export interface StorageStatus {
  current_data_dir: string;
  database_path: string;
  disk_total_bytes: number;
  disk_free_bytes: number;
  documents_count: number;
  database_size_bytes: number;
  vector_store_size_bytes: number;
}

export const getSystemHealth = async (): Promise<SystemHealth> => {
  return (await api.get<SystemHealth>('/system/health')).data;
};

export const getSystemHardware = async (): Promise<SystemHardware> => {
  return (await api.get<SystemHardware>('/system/hardware')).data;
};

export const getOllamaStatus = async (): Promise<OllamaStatus> => {
  return (await api.get<OllamaStatus>('/models/ollama/status')).data;
};

export const testLLMConnection = async (payload: {
  provider_type: string;
  model_name: string;
  base_url?: string | null;
  api_key?: string | null;
}): Promise<LLMConnectionTestResult> => {
  return (await api.post<LLMConnectionTestResult>('/models/llm/test', payload)).data;
};

export const testEmbeddingConnection = async (payload: {
  provider_type: string;
  model_name: string;
  base_url?: string | null;
  api_key?: string | null;
}): Promise<EmbeddingTestResult> => {
  return (await api.post<EmbeddingTestResult>('/models/embeddings/test', payload)).data;
};

export const createLLMConfig = async (payload: {
  name: string;
  provider_type: string;
  model_name: string;
  base_url?: string | null;
  api_key?: string | null;
  is_default?: boolean;
}): Promise<LLMConfig> => {
  return (await api.post<LLMConfig>('/models/llm', payload)).data;
};

export const getLLMConfigs = async (): Promise<LLMConfig[]> => {
  return (await api.get<LLMConfig[]>('/models/llm')).data;
};

export const getKnowledgeBases = async (): Promise<KnowledgeBaseItem[]> => {
  return (await api.get<KnowledgeBaseItem[]>('/knowledge-bases')).data;
};

export const createKnowledgeBase = async (payload: {
  name: string;
  description?: string | null;
  embedding_config_id?: string | null;
  retrieval_config_id?: string | null;
}): Promise<KnowledgeBaseItem> => {
  return (await api.post<KnowledgeBaseItem>('/knowledge-bases', payload)).data;
};

export const getStorageStatus = async (): Promise<StorageStatus> => {
  return (await api.get<StorageStatus>('/settings/storage')).data;
};

export const updateLLMConfig = async (
  id: string,
  payload: Partial<{
    name: string;
    provider_type: string;
    model_name: string;
    base_url: string | null;
    api_key: string | null;
    temperature: number;
    max_tokens: number;
    is_default: boolean;
  }>
): Promise<LLMConfig> => {
  return (await api.put<LLMConfig>(`/models/llm/${id}`, payload)).data;
};

export const deleteLLMConfig = async (id: string): Promise<void> => {
  await api.delete(`/models/llm/${id}`);
};

export const getEmbeddingConfigs = async (): Promise<EmbeddingConfig[]> => {
  return (await api.get<EmbeddingConfig[]>('/models/embeddings')).data;
};

export const createEmbeddingConfig = async (payload: {
  name: string;
  provider_type: string;
  model_name: string;
  dimensions: number;
  base_url?: string | null;
  api_key?: string | null;
  batch_size?: number;
  is_default?: boolean;
}): Promise<EmbeddingConfig> => {
  return (await api.post<EmbeddingConfig>('/models/embeddings', payload)).data;
};

export const updateEmbeddingConfig = async (
  id: string,
  payload: Partial<{
    name: string;
    provider_type: string;
    model_name: string;
    dimensions: number;
    base_url: string | null;
    api_key: string | null;
    batch_size: number;
    is_default: boolean;
  }>
): Promise<EmbeddingConfig> => {
  return (await api.put<EmbeddingConfig>(`/models/embeddings/${id}`, payload)).data;
};

export const deleteEmbeddingConfig = async (id: string): Promise<void> => {
  await api.delete(`/models/embeddings/${id}`);
};

export const testRerankerConnection = async (payload: {
  reranker_provider?: string | null;
  reranker_model?: string | null;
  base_url?: string | null;
  api_key?: string | null;
}): Promise<RerankerTestResult> => {
  return (await api.post<RerankerTestResult>('/models/rerankers/test', payload)).data;
};

export const getRetrievalSettings = async (): Promise<RetrievalConfig> => {
  return (await api.get<RetrievalConfig>('/settings/retrieval')).data;
};

export const updateRetrievalSettings = async (
  payload: Partial<RetrievalConfig>
): Promise<RetrievalConfig> => {
  return (await api.put<RetrievalConfig>('/settings/retrieval', payload)).data;
};

// ─── JOBS & CHUNKS TYPES & METHODS ──────────────────────────────────────────

export interface ProcessingJob {
  id: string;
  job_type: string;
  knowledge_base_id?: string | null;
  document_id?: string | null;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  stage?: string | null;
  progress_percent?: number | null;
  message?: string | null;
  error_details?: string | null;
  payload_json?: Record<string, unknown>;
  started_at?: string | null;
  completed_at?: string | null;
  created_at: string;
}

export interface DocumentChunk {
  id: string;
  document_id: string;
  knowledge_base_id?: string | null;
  chunk_index: number;
  page_number?: number | null;
  section_title?: string | null;
  sheet_name?: string | null;
  is_table?: boolean;
  content: string;
  char_count: number;
  metadata_json?: Record<string, unknown>;
  created_at?: string;
}

export interface DocumentChunksResponse {
  document_id: string;
  knowledge_base_id?: string;
  total_chunks: number;
  offset: number;
  limit: number;
  chunks: DocumentChunk[];
}

export interface KnowledgeBaseDetailResponse {
  id: string;
  name: string;
  description?: string | null;
  embedding_config_id?: string | null;
  embedding_provider?: string | null;
  embedding_model_name?: string | null;
  embedding_dimensions?: number | null;
  retrieval_config_id?: string | null;
  document_count: number;
  chunk_count: number;
  vector_collection_name?: string;
  is_healthy?: boolean;
  reindex_required?: boolean;
  active_jobs?: ProcessingJob[];
  created_at: string;
  updated_at: string;
}

export const getKnowledgeBase = async (kbId: string): Promise<KnowledgeBaseDetailResponse> => {
  return (await api.get<KnowledgeBaseDetailResponse>(`/knowledge-bases/${kbId}`)).data;
};

export const deleteKnowledgeBase = async (kbId: string): Promise<void> => {
  await api.delete(`/knowledge-bases/${kbId}`);
};

export const reindexKnowledgeBase = async (
  kbId: string,
  newEmbeddingConfigId?: string
): Promise<{ job_id: string; status: string }> => {
  return (
    await api.post<{ job_id: string; status: string }>(`/knowledge-bases/${kbId}/reindex`, {
      new_embedding_config_id: newEmbeddingConfigId || null,
    })
  ).data;
};

export const exportKnowledgeBase = async (kbId: string, filename?: string): Promise<void> => {
  const response = await api.post(`/knowledge-bases/${kbId}/export`, {}, { responseType: 'blob' });
  const url = window.URL.createObjectURL(new Blob([response.data], { type: 'application/zip' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || `quantiva_kb_${kbId}.qkb`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
};

export const importKnowledgeBase = async (file: File): Promise<KnowledgeBaseDetailResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  return (
    await api.post<KnowledgeBaseDetailResponse>('/knowledge-bases/import', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  ).data;
};

export const getKBDocuments = async (
  kbId: string,
  statusFilter?: string
): Promise<DocumentItem[]> => {
  const params: Record<string, string> = {};
  if (statusFilter) params.status = statusFilter;
  return (await api.get<DocumentItem[]>(`/knowledge-bases/${kbId}/documents`, { params })).data;
};

export const uploadKBDocuments = async (
  kbId: string,
  files: File[]
): Promise<{ total_uploaded: number; total_failed: number }> => {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  return (
    await api.post<{ total_uploaded: number; total_failed: number }>(
      `/knowledge-bases/${kbId}/documents/upload`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } }
    )
  ).data;
};

export const getDocumentChunks = async (
  docId: string,
  limit = 100,
  offset = 0
): Promise<DocumentChunksResponse> => {
  return (
    await api.get<DocumentChunksResponse>(`/documents/${docId}/chunks`, {
      params: { limit, offset },
    })
  ).data;
};

export const getProcessingJobs = async (params?: {
  knowledge_base_id?: string;
  document_id?: string;
  status?: string;
  limit?: number;
  offset?: number;
}): Promise<ProcessingJob[]> => {
  return (await api.get<ProcessingJob[]>('/jobs', { params })).data;
};

export const cancelJob = async (jobId: string): Promise<void> => {
  await api.post(`/jobs/${jobId}/cancel`);
};

export const retryJob = async (jobId: string): Promise<{ new_job_id: string }> => {
  return (await api.post<{ new_job_id: string }>(`/jobs/${jobId}/retry`)).data;
};

// ─── CONVERSATIONS, CITATIONS & BACKUP TYPES & METHODS ─────────────────────────

export interface RetrievedChunk {
  chunk_id: string;
  content: string;
  score: number;
  metadata: {
    source_file?: string;
    filename?: string;
    page_number?: number;
    sheet_name?: string;
    is_table?: boolean;
    [key: string]: unknown;
  };
  dense_rank?: number | null;
  sparse_rank?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;
  final_rank?: number | null;
}

export interface StructuredCitation {
  marker: string;
  index: number;
  chunk_id: string;
  document_id: string;
  source_file: string;
  page_number?: number | null;
  quote_snippet: string;
  lexical_overlap?: number;
  quote_verified?: boolean;
}

export interface ConversationItem {
  id: string;
  knowledge_base_id: string;
  title: string;
  model_config_id?: string | null;
  is_pinned?: boolean;
  message_count?: number;
  created_at: string;
  updated_at: string;
}

export interface ConversationMessage {
  id: string;
  conversation_id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: StructuredCitation[];
  reasoning_trace?: {
    reasoning?: string;
    thoughts?: string[];
    tool_calls?: Array<{
      tool: string;
      input: Record<string, unknown> | string;
      output?: string;
    }>;
    [key: string]: unknown;
  } | null;
  retrieved_chunks?: RetrievedChunk[];
  token_usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  latency_ms?: number;
  provider_used?: string;
  model_used?: string;
  created_at: string;
}

export interface BackupItem {
  filename: string;
  path: string;
  size_bytes: number;
  created_at: string;
  schema_version?: number | null;
  app_version?: string | null;
}

export const listConversations = async (kb_id?: string): Promise<ConversationItem[]> => {
  const params: Record<string, string> = {};
  if (kb_id) params.kb_id = kb_id;
  return (await api.get<ConversationItem[]>('/conversations', { params })).data;
};

export const createConversation = async (payload: {
  knowledge_base_id: string;
  title?: string;
  model_config_id?: string | null;
  is_pinned?: boolean;
}): Promise<ConversationItem> => {
  return (await api.post<ConversationItem>('/conversations', payload)).data;
};

export const getConversationMessages = async (
  conv_id: string
): Promise<ConversationMessage[]> => {
  return (await api.get<ConversationMessage[]>(`/conversations/${conv_id}/messages`)).data;
};

export const deleteConversation = async (conv_id: string): Promise<void> => {
  await api.delete(`/conversations/${conv_id}`);
};

export const listWorkspaceBackups = async (): Promise<BackupItem[]> => {
  return (await api.get<BackupItem[]>('/settings/backups')).data;
};

export const createWorkspaceBackup = async (
  include_models = false
): Promise<{ status: string; archive_path: string; filename: string; size_bytes: number }> => {
  return (
    await api.post<{ status: string; archive_path: string; filename: string; size_bytes: number }>(
      '/settings/backups/create',
      { include_models }
    )
  ).data;
};

export const restoreWorkspaceBackup = async (
  archive_path: string,
  create_safety_snapshot = true
): Promise<{ status: string; archive_path: string }> => {
  return (
    await api.post<{ status: string; archive_path: string }>('/settings/backups/restore', {
      archive_path,
      create_safety_snapshot,
    })
  ).data;
};

export const migrateStorageLocation = async (
  new_path: string,
  verify_space = true
): Promise<{ status: string; new_data_dir: string }> => {
  return (
    await api.post<{ status: string; new_data_dir: string }>('/settings/storage/migrate', {
      new_path,
      verify_space,
    })
  ).data;
};

export const streamConversationAsk = (
  convId: string,
  payload: {
    query: string;
    mode?: 'direct' | 'agentic';
    model_config_id?: string | null;
    filters?: Record<string, unknown>;
  },
  callbacks: {
    onStatus?: (stage: string) => void;
    onSources?: (sources: RetrievedChunk[]) => void;
    onCitations?: (citations: StructuredCitation[]) => void;
    onTokenDelta?: (text: string) => void;
    onReasoningDelta?: (text: string) => void;
    onDone?: (data: { usage?: Record<string, unknown>; latency_ms?: number }) => void;
    onError?: (err: unknown) => void;
  }
): (() => void) => {
  const controller = new AbortController();
  void fetchEventSource(`/conversations/${convId}/ask`, {
    method: 'POST',
    signal: controller.signal,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: payload.query,
      stream: true,
      mode: payload.mode || 'direct',
      model_config_id: payload.model_config_id || null,
      filters: payload.filters || null,
    }),
    onMessage: (msg) => {
      try {
        const parsed = JSON.parse(msg.data);
        if (msg.event === 'status') {
          callbacks.onStatus?.(parsed.stage || parsed.text || '');
        } else if (msg.event === 'sources') {
          callbacks.onSources?.(Array.isArray(parsed) ? parsed : []);
        } else if (msg.event === 'citations') {
          callbacks.onCitations?.(Array.isArray(parsed) ? parsed : []);
        } else if (msg.event === 'token_delta') {
          callbacks.onTokenDelta?.(parsed.text || '');
        } else if (msg.event === 'reasoning_delta') {
          callbacks.onReasoningDelta?.(parsed.text || '');
        } else if (msg.event === 'done') {
          callbacks.onDone?.(parsed);
        } else if (msg.event === 'error') {
          callbacks.onError?.(new Error(parsed.message || 'Stream error'));
        }
      } catch {
        if (msg.event === 'token_delta') {
          callbacks.onTokenDelta?.(msg.data);
        }
      }
    },
    onError: (err) => {
      callbacks.onError?.(err);
    },
  });

  return () => {
    controller.abort();
  };
};

// ─── EVALUATION & RAG BENCHMARKING TYPES & METHODS ────────────────────────────

export interface BenchmarkQueryItem {
  question: string;
  expected_document_names?: string[];
  expected_page_numbers?: number[];
  expected_keywords_or_snippets?: string[];
}

export interface EvaluationScorecardRow {
  strategy: string;
  display_name: string;
  'recall@5'?: number;
  'precision@5'?: number;
  mrr?: number;
  'ndcg@5'?: number;
  latency_ms?: number;
  [key: string]: unknown;
}

export interface EvaluationRunItem {
  id: string;
  knowledge_base_id: string;
  name: string;
  status: string;
  strategies_evaluated: string[];
  strategy_results: {
    scorecard: EvaluationScorecardRow[];
    best_strategy?: string | null;
    strategies?: Record<string, unknown>;
    k_values?: number[];
    total_benchmark_queries?: number;
  };
  generation_metrics?: {
    sample_question?: string;
    answer_excerpt?: string;
    citation_precision?: number;
    citation_groundedness?: number;
    faithfulness_score?: number;
    lexical_overlap_score?: number;
    llm_judge_score?: number | null;
    judge_reasoning?: string | null;
    ungrounded_claims?: string[];
    generation_latency_ms?: number;
    error?: string;
  };
  telemetry?: {
    retrieval_latency_ms?: number;
    reranking_latency_ms?: number;
    ttft_ms?: number | null;
    total_generation_latency_ms?: number;
    total_pipeline_latency_ms?: number;
    peak_rss_mb_delta?: number;
    start_rss_mb?: number;
    end_rss_mb?: number;
  };
  benchmark_dataset_count: number;
  created_at: string;
  completed_at?: string | null;
}

export const runEvaluation = async (payload: {
  knowledge_base_id: string;
  name?: string;
  strategies?: string[];
  benchmark_dataset?: BenchmarkQueryItem[];
  evaluate_generation?: boolean;
  k_values?: number[];
}): Promise<EvaluationRunItem> => {
  return (await api.post<EvaluationRunItem>('/evaluation/run', payload)).data;
};

export const listEvaluationRuns = async (
  knowledge_base_id?: string
): Promise<EvaluationRunItem[]> => {
  const params: Record<string, string> = {};
  if (knowledge_base_id) params.knowledge_base_id = knowledge_base_id;
  return (await api.get<EvaluationRunItem[]>('/evaluation/runs', { params })).data;
};

export const getEvaluationRun = async (runId: string): Promise<EvaluationRunItem> => {
  return (await api.get<EvaluationRunItem>(`/evaluation/runs/${runId}`)).data;
};

export const getDefaultBenchmarks = async (
  knowledge_base_id?: string
): Promise<BenchmarkQueryItem[]> => {
  const params: Record<string, string> = {};
  if (knowledge_base_id) params.knowledge_base_id = knowledge_base_id;
  return (await api.get<BenchmarkQueryItem[]>('/evaluation/benchmarks/default', { params })).data;
};






