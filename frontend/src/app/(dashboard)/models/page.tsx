"use client";

import React, { useEffect, useState, useCallback } from 'react';
import {
  Sparkles,
  Layers,
  SlidersHorizontal,
  RefreshCw,
} from 'lucide-react';
import {
  getSystemHardware,
  getOllamaStatus,
  getLLMConfigs,
  getEmbeddingConfigs,
  SystemHardware,
  OllamaStatus,
  LLMConfig,
  EmbeddingConfig,
} from '@/lib/api';
import { HardwareRuntimeBanner } from '@/components/models/HardwareRuntimeBanner';
import { LLMConfigTab } from '@/components/models/LLMConfigTab';
import { EmbeddingConfigTab } from '@/components/models/EmbeddingConfigTab';
import { RerankerConfigTab } from '@/components/models/RerankerConfigTab';

export default function ModelStudioPage() {
  const [activeTab, setActiveTab] = useState<'llm' | 'embeddings' | 'rerankers'>('llm');

  // Shared Data States
  const [hardware, setHardware] = useState<SystemHardware | null>(null);
  const [ollama, setOllama] = useState<OllamaStatus | null>(null);
  const [llmConfigs, setLlmConfigs] = useState<LLMConfig[]>([]);
  const [embeddingConfigs, setEmbeddingConfigs] = useState<EmbeddingConfig[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchAllData = useCallback(async () => {
    setLoading(true);
    try {
      const [hwRes, ollamaRes, llmRes, embRes] = await Promise.allSettled([
        getSystemHardware(),
        getOllamaStatus(),
        getLLMConfigs(),
        getEmbeddingConfigs(),
      ]);

      if (hwRes.status === 'fulfilled') setHardware(hwRes.value);
      if (ollamaRes.status === 'fulfilled') setOllama(ollamaRes.value);
      if (llmRes.status === 'fulfilled') setLlmConfigs(llmRes.value);
      if (embRes.status === 'fulfilled') setEmbeddingConfigs(embRes.value);
    } catch (err) {
      console.warn('Error loading Model Studio data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchAllData();
  }, [fetchAllData]);

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-8">
      {/* ── Page Header ─────────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="rounded-lg bg-blue-500/10 p-1.5 text-blue-400 border border-blue-500/20">
              <Sparkles className="h-4 w-4" />
            </span>
            <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">Model Studio</h1>
          </div>
          <p className="text-xs text-neutral-400 mt-1">
            Independent configuration of LLM reasoning models, dense vector embeddings, and cross-encoder rerankers.
          </p>
        </div>

        <button
          type="button"
          onClick={fetchAllData}
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3.5 py-2 text-xs font-semibold text-neutral-300 hover:bg-white/10 hover:text-white transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
          Refresh Models
        </button>
      </div>

      {/* ── 1. Hardware & Runtime Overview Banner ────────────────────────── */}
      <HardwareRuntimeBanner
        hardware={hardware}
        ollama={ollama}
        loading={loading}
        onRefresh={fetchAllData}
      />

      {/* ── Navigation Tabs ──────────────────────────────────────────────── */}
      <div className="border-b border-white/10">
        <nav className="flex space-x-6" aria-label="Model Studio Tabs">
          {/* Tab 1: LLM */}
          <button
            type="button"
            onClick={() => setActiveTab('llm')}
            className={`group inline-flex items-center gap-2 pb-4 text-xs font-semibold border-b-2 transition-all ${
              activeTab === 'llm'
                ? 'border-blue-500 text-blue-400'
                : 'border-transparent text-neutral-400 hover:text-neutral-200'
            }`}
          >
            <Sparkles
              className={`h-4 w-4 ${
                activeTab === 'llm' ? 'text-blue-400' : 'text-neutral-500 group-hover:text-neutral-300'
              }`}
            />
            <span>LLM Reasoning Engines</span>
            <span
              className={`rounded-full px-2 py-0.5 text-[10px] font-mono ${
                activeTab === 'llm' ? 'bg-blue-500/20 text-blue-300' : 'bg-white/5 text-neutral-500'
              }`}
            >
              {llmConfigs.length}
            </span>
          </button>

          {/* Tab 2: Embeddings */}
          <button
            type="button"
            onClick={() => setActiveTab('embeddings')}
            className={`group inline-flex items-center gap-2 pb-4 text-xs font-semibold border-b-2 transition-all ${
              activeTab === 'embeddings'
                ? 'border-teal-500 text-teal-400'
                : 'border-transparent text-neutral-400 hover:text-neutral-200'
            }`}
          >
            <Layers
              className={`h-4 w-4 ${
                activeTab === 'embeddings' ? 'text-teal-400' : 'text-neutral-500 group-hover:text-neutral-300'
              }`}
            />
            <span>Vector Embeddings</span>
            <span
              className={`rounded-full px-2 py-0.5 text-[10px] font-mono ${
                activeTab === 'embeddings' ? 'bg-teal-500/20 text-teal-300' : 'bg-white/5 text-neutral-500'
              }`}
            >
              {embeddingConfigs.length}
            </span>
          </button>

          {/* Tab 3: Reranker */}
          <button
            type="button"
            onClick={() => setActiveTab('rerankers')}
            className={`group inline-flex items-center gap-2 pb-4 text-xs font-semibold border-b-2 transition-all ${
              activeTab === 'rerankers'
                ? 'border-purple-500 text-purple-400'
                : 'border-transparent text-neutral-400 hover:text-neutral-200'
            }`}
          >
            <SlidersHorizontal
              className={`h-4 w-4 ${
                activeTab === 'rerankers' ? 'text-purple-400' : 'text-neutral-500 group-hover:text-neutral-300'
              }`}
            />
            <span>Cross-Encoder Reranker</span>
          </button>
        </nav>
      </div>

      {/* ── Active Tab View ──────────────────────────────────────────────── */}
      <div>
        {activeTab === 'llm' && (
          <LLMConfigTab
            configs={llmConfigs}
            ollama={ollama}
            onRefresh={fetchAllData}
          />
        )}

        {activeTab === 'embeddings' && (
          <EmbeddingConfigTab
            configs={embeddingConfigs}
            onRefresh={fetchAllData}
          />
        )}

        {activeTab === 'rerankers' && <RerankerConfigTab />}
      </div>
    </div>
  );
}
