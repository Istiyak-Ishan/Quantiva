"use client";

import React, { useState, useEffect } from 'react';
import {
  SlidersHorizontal,
  Server,
  Cloud,
  CheckCircle2,
  AlertTriangle,
  Play,
  Loader2,
  Check,
  ShieldCheck,
  Key,
  Eye,
  EyeOff,
  Sparkles,
  ToggleLeft,
  ToggleRight,
} from 'lucide-react';
import {
  RetrievalConfig,
  RerankerTestResult,
  getRetrievalSettings,
  updateRetrievalSettings,
  testRerankerConnection,
} from '@/lib/api';

export const RERANKER_PRESETS = [
  {
    name: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    provider: 'local_cross_encoder',
    label: 'MS-Marco MiniLM-L-6-v2 (Fastest, CPU-Optimized)',
  },
  {
    name: 'BAAI/bge-reranker-base',
    provider: 'local_cross_encoder',
    label: 'BGE Reranker Base (High Accuracy, Balanced)',
  },
  {
    name: 'BAAI/bge-reranker-large',
    provider: 'local_cross_encoder',
    label: 'BGE Reranker Large (State-of-the-Art, High Precision)',
  },
  {
    name: 'rerank-english-v3.0',
    provider: 'custom_rerank_api',
    label: 'Cohere Rerank English v3 (Cloud API)',
  },
];

export function RerankerConfigTab() {
  const [retrievalConfig, setRetrievalConfig] = useState<RetrievalConfig | null>(null);
  const [loadingConfig, setLoadingConfig] = useState(true);

  // Form State
  const [enableReranker, setEnableReranker] = useState(true);
  const [providerType, setProviderType] = useState('local_cross_encoder');
  const [modelName, setModelName] = useState('cross-encoder/ms-marco-MiniLM-L-6-v2');
  const [finalTopK, setFinalTopK] = useState(8);
  const [baseUrl, setBaseUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);

  // Testing & Saving States
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<RerankerTestResult | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Fetch active retrieval configuration
  const loadConfig = async () => {
    setLoadingConfig(true);
    try {
      const cfg = await getRetrievalSettings();
      setRetrievalConfig(cfg);
      setEnableReranker(cfg.enable_reranking);
      if (cfg.reranker_provider) setProviderType(cfg.reranker_provider);
      if (cfg.reranker_model) setModelName(cfg.reranker_model);
      if (cfg.final_top_k) setFinalTopK(cfg.final_top_k);
    } catch (err) {
      console.warn('Failed to fetch retrieval settings:', err);
    } finally {
      setLoadingConfig(false);
    }
  };

  useEffect(() => {
    void loadConfig();
  }, []);

  const handlePresetSelect = (preset: typeof RERANKER_PRESETS[0]) => {
    setProviderType(preset.provider);
    setModelName(preset.name);
    setTestResult(null);
  };

  // Test Reranker
  const handleTest = async () => {
    setIsTesting(true);
    setTestResult(null);
    try {
      const res = await testRerankerConnection({
        reranker_provider: providerType,
        reranker_model: modelName.trim(),
        base_url: baseUrl.trim() || undefined,
        api_key: apiKey.trim() || undefined,
      });
      setTestResult(res);
    } catch (err) {
      setTestResult({
        success: false,
        latency_ms: 0,
        provider: providerType,
        model: modelName,
        error_message: err instanceof Error ? err.message : 'Reranker test failed',
      });
    } finally {
      setIsTesting(false);
    }
  };

  // Save Reranker Settings
  const handleSave = async () => {
    setIsSaving(true);
    setSaveSuccess(null);
    setSaveError(null);
    try {
      const updated = await updateRetrievalSettings({
        enable_reranking: enableReranker,
        reranker_provider: providerType,
        reranker_model: modelName.trim(),
        final_top_k: finalTopK,
      });
      setRetrievalConfig(updated);
      setSaveSuccess('Reranker configuration updated and active for all queries!');
      setTimeout(() => setSaveSuccess(null), 3000);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to update reranker settings');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* ── Active Reranker Form ─────────────────────────────────────────── */}
      <div className="rounded-2xl border border-white/10 bg-black/40 p-6 md:p-8 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
              <SlidersHorizontal className="h-4 w-4 text-purple-400" />
              Configure Cross-Encoder Reranker
            </h2>
            <p className="text-xs text-neutral-400 mt-1">
              Rerankers score document chunks against user questions jointly, eliminating false positives from vector similarity.
            </p>
          </div>

          {/* Master Enable/Disable Toggle */}
          <button
            type="button"
            onClick={() => setEnableReranker(!enableReranker)}
            className={`inline-flex items-center gap-2 rounded-xl border px-4 py-2 text-xs font-semibold transition-all ${
              enableReranker
                ? 'border-purple-500/40 bg-purple-500/15 text-purple-300 shadow-sm shadow-purple-500/20'
                : 'border-white/10 bg-white/5 text-neutral-400 hover:text-white'
            }`}
          >
            {enableReranker ? (
              <>
                <ToggleRight className="h-4 w-4 text-purple-400" />
                Reranking Enabled
              </>
            ) : (
              <>
                <ToggleLeft className="h-4 w-4 text-neutral-500" />
                Reranking Disabled
              </>
            )}
          </button>
        </div>

        {/* Provider Selector Chips */}
        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
            Reranker Provider
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {/* Local Cross-Encoder */}
            <button
              type="button"
              onClick={() => {
                setProviderType('local_cross_encoder');
                setModelName('cross-encoder/ms-marco-MiniLM-L-6-v2');
                setTestResult(null);
              }}
              className={`rounded-xl border p-4 text-left transition-all flex items-center justify-between ${
                providerType === 'local_cross_encoder'
                  ? 'border-purple-500 bg-purple-500/10 text-white shadow-sm shadow-purple-500/10'
                  : 'border-white/10 bg-white/[0.02] text-neutral-400 hover:border-white/20'
              }`}
            >
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-teal-500/10 text-teal-400">
                  <Server className="h-4 w-4" />
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-white">Local Cross-Encoder</h4>
                  <p className="text-[11px] text-neutral-400 mt-0.5">
                    HuggingFace sentence-transformers execution (100% offline on CPU/DirectML)
                  </p>
                </div>
              </div>
              <span className="rounded bg-teal-500/10 px-2 py-0.5 text-[10px] font-semibold text-teal-300 border border-teal-500/20">
                Offline
              </span>
            </button>

            {/* Custom / Cloud Reranker API */}
            <button
              type="button"
              onClick={() => {
                setProviderType('custom_rerank_api');
                setModelName('rerank-english-v3.0');
                setTestResult(null);
              }}
              className={`rounded-xl border p-4 text-left transition-all flex items-center justify-between ${
                providerType === 'custom_rerank_api'
                  ? 'border-purple-500 bg-purple-500/10 text-white shadow-sm shadow-purple-500/10'
                  : 'border-white/10 bg-white/[0.02] text-neutral-400 hover:border-white/20'
              }`}
            >
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-500/10 text-blue-400">
                  <Cloud className="h-4 w-4" />
                </div>
                <div>
                  <h4 className="text-xs font-semibold text-white">Custom / Cohere API</h4>
                  <p className="text-[11px] text-neutral-400 mt-0.5">
                    External Cohere or OpenAI-compatible reranking endpoint
                  </p>
                </div>
              </div>
              <span className="rounded bg-blue-500/10 px-2 py-0.5 text-[10px] font-semibold text-blue-300 border border-blue-500/20">
                Cloud API
              </span>
            </button>
          </div>
        </div>

        {/* Model Presets */}
        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
            Tested Model Presets
          </label>
          <div className="flex flex-wrap gap-2">
            {RERANKER_PRESETS.map((preset) => {
              const isSelected = modelName === preset.name && providerType === preset.provider;
              return (
                <button
                  key={preset.name}
                  type="button"
                  onClick={() => handlePresetSelect(preset)}
                  className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                    isSelected
                      ? 'border-purple-500 bg-purple-500/20 text-purple-300'
                      : 'border-white/10 bg-white/5 text-neutral-400 hover:border-white/20 hover:text-white'
                  }`}
                >
                  {preset.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Form Inputs */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 pt-2">
          {/* Model Name */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-neutral-300">Model Name / HuggingFace ID</label>
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="e.g. cross-encoder/ms-marco-MiniLM-L-6-v2"
              className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3.5 py-2 text-xs text-white focus:border-purple-500 focus:outline-none font-mono"
            />
          </div>

          {/* Final Top-K Slider */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-neutral-300">Final Top-K Chunks to Context</label>
              <span className="font-mono text-xs text-purple-400">{finalTopK}</span>
            </div>
            <input
              type="range"
              min={2}
              max={25}
              step={1}
              value={finalTopK}
              onChange={(e) => setFinalTopK(parseInt(e.target.value, 10))}
              className="w-full accent-purple-500"
            />
            <span className="text-[10px] text-neutral-500 block">
              Highest-scoring chunks passed to the LLM context window after reranking.
            </span>
          </div>

          {/* Custom API Base URL (if custom API) */}
          {providerType === 'custom_rerank_api' && (
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-neutral-300">Endpoint Base URL</label>
              <input
                type="text"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="e.g. https://api.cohere.com/v1"
                className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3.5 py-2 text-xs text-white focus:border-purple-500 focus:outline-none font-mono"
              />
            </div>
          )}

          {/* Custom API Key (if custom API) */}
          {providerType === 'custom_rerank_api' && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-neutral-300">API Key</label>
                <span className="text-[10px] text-neutral-500">Stored via Windows Credential Manager</span>
              </div>
              <div className="relative">
                <input
                  type={showApiKey ? 'text' : 'password'}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="Enter API Key"
                  className="w-full rounded-xl border border-white/10 bg-neutral-900 pl-3.5 pr-10 py-2 text-xs text-white focus:border-purple-500 focus:outline-none font-mono"
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
          )}
        </div>

        {/* Action Controls & Testing */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-white/5">
          <button
            type="button"
            onClick={handleTest}
            disabled={isTesting || !modelName.trim()}
            className="inline-flex items-center gap-1.5 rounded-xl border border-purple-500/30 bg-purple-500/10 px-4 py-2 text-xs font-semibold text-purple-300 hover:bg-purple-500/20 disabled:opacity-50 transition-colors"
          >
            {isTesting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            Test Reranker
          </button>

          <button
            type="button"
            onClick={handleSave}
            disabled={isSaving}
            className="inline-flex items-center gap-1.5 rounded-xl bg-purple-600 px-5 py-2 text-xs font-semibold text-white shadow-md shadow-purple-600/30 hover:bg-purple-500 disabled:opacity-50 transition-colors"
          >
            <Check className="h-3.5 w-3.5" />
            Save Reranker Settings
          </button>
        </div>

        {/* Test Result Display */}
        {testResult && (
          <div
            className={`rounded-xl border p-4 text-xs space-y-2 ${
              testResult.success
                ? 'border-purple-500/20 bg-purple-950/20 text-purple-300'
                : 'border-red-500/20 bg-red-950/20 text-red-300'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 font-medium">
                {testResult.success ? (
                  <CheckCircle2 className="h-4 w-4 text-purple-400 shrink-0" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />
                )}
                {testResult.success
                  ? `Reranker Verified for ${testResult.provider} (${testResult.model})`
                  : testResult.error_message || 'Reranker test failed'}
              </span>
              <span className="font-mono text-[11px] opacity-80">{testResult.latency_ms.toFixed(0)}ms latency</span>
            </div>

            {testResult.success && (
              <p className="text-[11px] text-purple-200/90 font-mono">
                Candidate scoring and relevance ordering successfully calibrated.
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
    </div>
  );
}
