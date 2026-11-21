"use client";

import React, { useState } from 'react';
import {
  Layers,
  Server,
  Cloud,
  CheckCircle2,
  AlertTriangle,
  Play,
  Loader2,
  Trash2,
  AlertCircle,
  Star,
  Key,
  Eye,
  EyeOff,
  Check,
  ShieldAlert,
} from 'lucide-react';
import {
  EmbeddingConfig,
  EmbeddingTestResult,
  testEmbeddingConnection,
  createEmbeddingConfig,
  updateEmbeddingConfig,
  deleteEmbeddingConfig,
} from '@/lib/api';

interface EmbeddingConfigTabProps {
  configs: EmbeddingConfig[];
  onRefresh: () => void;
}

export const EMBEDDING_PROVIDERS = [
  { id: 'local_sentence_transformers', name: 'Local Sentence Transformers', isLocal: true, defaultUrl: '' },
  { id: 'ollama', name: 'Ollama Embeddings', isLocal: true, defaultUrl: 'http://127.0.0.1:11434' },
  { id: 'gemini', name: 'Gemini Embeddings', isLocal: false, defaultUrl: '' },
  { id: 'openai', name: 'OpenAI / Compatible API', isLocal: false, defaultUrl: '' },
];

export const EMBEDDING_PRESETS = [
  { name: 'BAAI/bge-small-en-v1.5', provider: 'local_sentence_transformers', dims: 384, label: 'BGE Small (384 dims - Fast & Accurate)' },
  { name: 'all-MiniLM-L6-v2', provider: 'local_sentence_transformers', dims: 384, label: 'MiniLM-L6-v2 (384 dims - Lightweight Standard)' },
  { name: 'intfloat/multilingual-e5-base', provider: 'local_sentence_transformers', dims: 768, label: 'Multilingual E5 Base (768 dims)' },
  { name: 'nomic-embed-text', provider: 'ollama', dims: 768, label: 'Nomic Embed Text (Ollama 768 dims)' },
  { name: 'text-embedding-004', provider: 'gemini', dims: 768, label: 'Google text-embedding-004 (768 dims)' },
  { name: 'text-embedding-3-small', provider: 'openai', dims: 1536, label: 'OpenAI text-embedding-3-small (1536 dims)' },
];

export function EmbeddingConfigTab({ configs, onRefresh }: EmbeddingConfigTabProps) {
  // Form State
  const [name, setName] = useState('Local BGE Small');
  const [providerType, setProviderType] = useState('local_sentence_transformers');
  const [modelName, setModelName] = useState('BAAI/bge-small-en-v1.5');
  const [dimensions, setDimensions] = useState(384);
  const [batchSize, setBatchSize] = useState(32);
  const [baseUrl, setBaseUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [isDefault, setIsDefault] = useState(false);

  // Testing & Saving States
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<EmbeddingTestResult | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  // List testing
  const [listTestingId, setListTestingId] = useState<string | null>(null);
  const [listTestResult, setListTestResult] = useState<{ id: string; res: EmbeddingTestResult } | null>(null);

  const handleProviderChange = (p: string) => {
    setProviderType(p);
    const prov = EMBEDDING_PROVIDERS.find((item) => item.id === p);
    setBaseUrl(prov?.defaultUrl || '');

    // Select first matching preset
    const preset = EMBEDDING_PRESETS.find((pre) => pre.provider === p);
    if (preset) {
      setModelName(preset.name);
      setDimensions(preset.dims);
      setName(`${prov?.name || p} (${preset.name})`);
    } else {
      setModelName('');
      setName(`${prov?.name || p} Custom`);
    }
    setTestResult(null);
  };

  const handlePresetSelect = (preset: typeof EMBEDDING_PRESETS[0]) => {
    setProviderType(preset.provider);
    const prov = EMBEDDING_PROVIDERS.find((item) => item.id === preset.provider);
    setBaseUrl(prov?.defaultUrl || '');
    setModelName(preset.name);
    setDimensions(preset.dims);
    setName(preset.label.split('(')[0].trim());
    setTestResult(null);
  };

  // Test embedding connection & detect dimensions
  const handleTest = async () => {
    setIsTesting(true);
    setTestResult(null);
    try {
      const res = await testEmbeddingConnection({
        provider_type: providerType,
        model_name: modelName.trim(),
        base_url: baseUrl.trim() || undefined,
        api_key: apiKey.trim() || undefined,
      });
      setTestResult(res);
      if (res.success && res.detected_dimensions) {
        setDimensions(res.detected_dimensions);
      }
    } catch (err) {
      setTestResult({
        success: false,
        latency_ms: 0,
        provider: providerType,
        model: modelName,
        error_message: err instanceof Error ? err.message : 'Embedding connection failed',
      });
    } finally {
      setIsTesting(false);
    }
  };

  // Save embedding config
  const handleSave = async (setDefault = false) => {
    setIsSaving(true);
    setSaveSuccess(null);
    setSaveError(null);
    try {
      await createEmbeddingConfig({
        name: name.trim() || `${providerType} (${modelName})`,
        provider_type: providerType,
        model_name: modelName.trim(),
        dimensions: dimensions > 0 ? dimensions : 384,
        batch_size: batchSize,
        base_url: baseUrl.trim() || undefined,
        api_key: apiKey.trim() || undefined,
        is_default: setDefault || isDefault,
      });
      setSaveSuccess(`Embedding configuration "${name}" saved!`);
      setTimeout(() => setSaveSuccess(null), 3000);
      onRefresh();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save embedding configuration');
    } finally {
      setIsSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this embedding configuration?')) return;
    try {
      await deleteEmbeddingConfig(id);
      onRefresh();
    } catch (err) {
      alert('Failed to delete embedding configuration');
    }
  };

  const handleTestListItem = async (cfg: EmbeddingConfig) => {
    setListTestingId(cfg.id);
    setListTestResult(null);
    try {
      const res = await testEmbeddingConnection({
        provider_type: cfg.provider_type,
        model_name: cfg.model_name,
        base_url: cfg.base_url,
      });
      setListTestResult({ id: cfg.id, res });
    } catch (err) {
      setListTestResult({
        id: cfg.id,
        res: {
          success: false,
          latency_ms: 0,
          provider: cfg.provider_type,
          model: cfg.model_name,
          error_message: err instanceof Error ? err.message : 'Embedding failed',
        },
      });
    } finally {
      setListTestingId(null);
    }
  };

  return (
    <div className="space-y-8">
      {/* ── Warning Banner: Dimension Lock & Reindexing Requirement ──────── */}
      <div className="rounded-2xl border border-amber-500/20 bg-amber-950/15 p-5">
        <div className="flex items-start gap-3">
          <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
          <div className="space-y-1">
            <h3 className="text-xs font-bold text-amber-200">
              Vector Dimension Space Lock Invariant
            </h3>
            <p className="text-xs text-amber-300/80 leading-relaxed">
              Every Knowledge Base strictly locks its vector embedding dimension (e.g. 384, 768, 1536) in its index metadata.
              Changing the embedding model on an existing Knowledge Base will invalidate previously indexed vectors and requires
              triggering a full document reindex.
            </p>
          </div>
        </div>
      </div>

      {/* ── Active Editor Form ───────────────────────────────────────────── */}
      <div className="rounded-2xl border border-white/10 bg-black/40 p-6 md:p-8 space-y-6">
        <div>
          <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
            <Layers className="h-4 w-4 text-teal-400" />
            Configure Embedding Model
          </h2>
          <p className="text-xs text-neutral-400 mt-1">
            Dense semantic vectors are generated locally or in the cloud to populate Quantiva vector stores.
          </p>
        </div>

        {/* Provider Selector Chips */}
        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
            Provider Type
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            {EMBEDDING_PROVIDERS.map((prov) => {
              const selected = providerType === prov.id;
              return (
                <button
                  key={prov.id}
                  type="button"
                  onClick={() => handleProviderChange(prov.id)}
                  className={`rounded-xl border p-3 text-left transition-all flex flex-col justify-between ${
                    selected
                      ? 'border-teal-500 bg-teal-500/10 text-white shadow-sm shadow-teal-500/10'
                      : 'border-white/10 bg-white/[0.02] text-neutral-400 hover:border-white/20 hover:bg-white/[0.04]'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    {prov.isLocal ? (
                      <Server className={`h-4 w-4 ${selected ? 'text-teal-400' : 'text-neutral-500'}`} />
                    ) : (
                      <Cloud className={`h-4 w-4 ${selected ? 'text-blue-400' : 'text-neutral-500'}`} />
                    )}
                    <span
                      className={`rounded px-1.5 py-0.5 text-[9px] font-semibold ${
                        prov.isLocal
                          ? 'bg-teal-500/10 text-teal-400 border border-teal-500/20'
                          : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                      }`}
                    >
                      {prov.isLocal ? 'Offline' : 'Cloud'}
                    </span>
                  </div>
                  <span className="text-xs font-semibold leading-tight">{prov.name}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Popular Presets Bar */}
        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
            Presets & Tested Models
          </label>
          <div className="flex flex-wrap gap-2">
            {EMBEDDING_PRESETS.map((preset) => {
              const isSelected = modelName === preset.name && providerType === preset.provider;
              return (
                <button
                  key={preset.name}
                  type="button"
                  onClick={() => handlePresetSelect(preset)}
                  className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                    isSelected
                      ? 'border-teal-500 bg-teal-500/20 text-teal-300'
                      : 'border-white/10 bg-white/5 text-neutral-400 hover:border-white/20 hover:text-white'
                  }`}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Form Inputs Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 pt-2">
          {/* Friendly Name */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-neutral-300">Configuration Name</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Local BGE Small Default"
              className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3.5 py-2 text-xs text-white focus:border-teal-500 focus:outline-none"
            />
          </div>

          {/* Model Name */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-neutral-300">Model Name / HuggingFace ID</label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="e.g. BAAI/bge-small-en-v1.5"
              className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3.5 py-2 text-xs text-white focus:border-teal-500 focus:outline-none font-mono"
            />
          </div>

          {/* Auto-Detected Dimensions (Read-Only Badge & Input) */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-neutral-300">Vector Dimensions</label>
              <span className="rounded bg-teal-500/10 px-2 py-0.5 text-[10px] font-mono text-teal-400 border border-teal-500/20">
                {dimensions > 0 ? `${dimensions} dimensions` : 'Auto-detected'}
              </span>
            </div>
            <input
              type="number"
              value={dimensions}
              readOnly
              className="w-full rounded-xl border border-white/10 bg-neutral-900/60 px-3.5 py-2 text-xs text-neutral-300 font-mono cursor-not-allowed"
            />
            <span className="text-[10px] text-neutral-500">
              Populated automatically after clicking <strong>Test Embedding</strong>.
            </span>
          </div>

          {/* Batch Size */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-neutral-300">Batch Size</label>
              <span className="font-mono text-xs text-teal-400">{batchSize}</span>
            </div>
            <input
              type="range"
              min={1}
              max={128}
              step={1}
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value, 10))}
              className="w-full accent-teal-500"
            />
            <span className="text-[10px] text-neutral-500 block">
              Number of document chunks vectorized concurrently per batch.
            </span>
          </div>

          {/* Base URL (if custom or local) */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-neutral-300">Base URL (Optional)</label>
            <input
              type="text"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="e.g. http://127.0.0.1:11434"
              className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3.5 py-2 text-xs text-white focus:border-teal-500 focus:outline-none font-mono"
            />
          </div>

          {/* API Key (if cloud) */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-neutral-300">API Key</label>
              <span className="text-[10px] text-neutral-500">Encrypted in Windows Keyring</span>
            </div>
            <div className="relative">
              <input
                type={showApiKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={providerType.startsWith('local') ? 'Not required for local models' : 'Enter API Key'}
                className="w-full rounded-xl border border-white/10 bg-neutral-900 pl-3.5 pr-10 py-2 text-xs text-white focus:border-teal-500 focus:outline-none font-mono"
              />
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-400 hover:text-white"
              >
                {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>
        </div>

        {/* Action Controls & Testing */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-white/5">
          <button
            type="button"
            onClick={handleTest}
            disabled={isTesting || !modelName.trim()}
            className="inline-flex items-center gap-1.5 rounded-xl border border-teal-500/30 bg-teal-500/10 px-4 py-2 text-xs font-semibold text-teal-300 hover:bg-teal-500/20 disabled:opacity-50 transition-colors"
          >
            {isTesting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            Test Embedding
          </button>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => handleSave(false)}
              disabled={isSaving || !modelName.trim()}
              className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold text-neutral-200 hover:bg-white/10 disabled:opacity-50 transition-colors"
            >
              Save Configuration
            </button>

            <button
              type="button"
              onClick={() => handleSave(true)}
              disabled={isSaving || !modelName.trim()}
              className="inline-flex items-center gap-1.5 rounded-xl bg-teal-600 px-4 py-2 text-xs font-semibold text-white shadow-md shadow-teal-600/30 hover:bg-teal-500 disabled:opacity-50 transition-colors"
            >
              <Star className="h-3.5 w-3.5" />
              Save & Set Default
            </button>
          </div>
        </div>

        {/* Test Result Display */}
        {testResult && (
          <div
            className={`rounded-xl border p-4 text-xs space-y-2 ${
              testResult.success
                ? 'border-teal-500/20 bg-teal-950/20 text-teal-300'
                : 'border-red-500/20 bg-red-950/20 text-red-300'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 font-medium">
                {testResult.success ? (
                  <CheckCircle2 className="h-4 w-4 text-teal-400 shrink-0" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />
                )}
                {testResult.success
                  ? `Embedding Validated for ${testResult.provider} (${testResult.model})`
                  : testResult.error_message || 'Embedding test failed'}
              </span>
              <span className="font-mono text-[11px] opacity-80">{testResult.latency_ms.toFixed(0)}ms latency</span>
            </div>

            {testResult.success && testResult.detected_dimensions && (
              <p className="text-[11px] text-teal-200/90 font-mono">
                Detected vector output: {testResult.detected_dimensions} dimensions.
              </p>
            )}
          </div>
        )}

        {saveSuccess && (
          <div className="rounded-xl border border-green-500/20 bg-green-950/20 p-3 text-xs text-green-300 flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-green-400" />
            {saveSuccess}
          </div>
        )}

        {saveError && (
          <div className="rounded-xl border border-red-500/20 bg-red-950/20 p-3 text-xs text-red-300 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-red-400" />
            {saveError}
          </div>
        )}
      </div>

      {/* ── Saved Embedding Configurations List ──────────────────────────── */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white">Saved Embedding Configurations</h2>
          <span className="text-xs text-neutral-500 font-mono">{configs.length} configured</span>
        </div>

        <div className="overflow-hidden rounded-xl border border-white/10 bg-black/30">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-white/5 bg-white/[0.02] text-neutral-400">
              <tr>
                <th className="px-4 py-3 font-medium">Name</th>
                <th className="px-4 py-3 font-medium">Provider</th>
                <th className="px-4 py-3 font-medium">Model</th>
                <th className="px-4 py-3 font-medium">Dimensions</th>
                <th className="px-4 py-3 font-medium">Batch Size</th>
                <th className="px-4 py-3 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-neutral-300">
              {configs.map((cfg) => (
                <tr key={cfg.id} className="hover:bg-white/[0.02] transition-colors">
                  <td className="px-4 py-3 font-medium text-white flex items-center gap-2">
                    <span>{cfg.name}</span>
                    {cfg.is_default && (
                      <span className="rounded bg-teal-500/10 px-1.5 py-0.5 text-[9px] font-semibold text-teal-400 border border-teal-500/20">
                        Default
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono uppercase text-neutral-400">{cfg.provider_type}</td>
                  <td className="px-4 py-3 font-mono text-neutral-300 truncate max-w-xs">{cfg.model_name}</td>
                  <td className="px-4 py-3 font-mono text-teal-400 font-semibold">{cfg.dimensions}-d</td>
                  <td className="px-4 py-3 font-mono text-neutral-400">{cfg.batch_size}</td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <button
                        type="button"
                        onClick={() => handleTestListItem(cfg)}
                        disabled={listTestingId === cfg.id}
                        className="p-1 text-teal-400 hover:text-teal-300 transition-colors"
                        title="Test Embedding"
                      >
                        {listTestingId === cfg.id ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <Play className="h-3.5 w-3.5" />
                        )}
                      </button>

                      <button
                        type="button"
                        onClick={() => handleDelete(cfg.id)}
                        className="p-1 text-red-400/80 hover:text-red-400 transition-colors"
                        title="Delete"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {listTestResult && (
          <div
            className={`rounded-xl border p-3 text-xs ${
              listTestResult.res.success
                ? 'border-teal-500/20 bg-teal-950/20 text-teal-300'
                : 'border-red-500/20 bg-red-950/20 text-red-300'
            }`}
          >
            {listTestResult.res.success
              ? `Embedding ${listTestResult.res.model} verified (${listTestResult.res.detected_dimensions}-d, ${listTestResult.res.latency_ms.toFixed(0)}ms latency).`
              : listTestResult.res.error_message || 'Embedding test failed.'}
          </div>
        )}
      </div>
    </div>
  );
}
