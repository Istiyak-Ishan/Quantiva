"use client";

import React, { useEffect, useState } from 'react';
import {
  Sliders,
  CheckCircle2,
  RotateCcw,
  Loader2,
  Layers,
  Search,
  Sparkles,
} from 'lucide-react';
import {
  getRetrievalSettings,
  updateRetrievalSettings,
  RetrievalConfig,
} from '@/lib/api';

export default function RetrievalSettingsPage() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Config fields
  const [searchMode, setSearchMode] = useState<string>('hybrid');
  const [denseTopK, setDenseTopK] = useState(20);
  const [sparseTopK, setSparseTopK] = useState(20);
  const [rrfK, setRrfK] = useState(60);
  const [denseWeight, setDenseWeight] = useState(1.0);
  const [sparseWeight, setSparseWeight] = useState(1.0);
  const [enableReranking, setEnableReranking] = useState(true);
  const [finalTopK, setFinalTopK] = useState(8);

  useEffect(() => {
    async function loadSettings() {
      try {
        setLoading(true);
        const cfg = await getRetrievalSettings();
        setSearchMode(cfg.search_mode || 'hybrid');
        setDenseTopK(cfg.dense_top_k ?? 20);
        setSparseTopK(cfg.sparse_top_k ?? 20);
        setRrfK(cfg.rrf_k ?? 60);
        setDenseWeight(cfg.dense_weight ?? 1.0);
        setSparseWeight(cfg.sparse_weight ?? 1.0);
        setEnableReranking(cfg.enable_reranking ?? true);
        setFinalTopK(cfg.final_top_k ?? 8);
      } catch (err) {
        console.error('Failed to load retrieval settings:', err);
        setError('Could not load current retrieval configuration from backend.');
      } finally {
        setLoading(false);
      }
    }
    void loadSettings();
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await updateRetrievalSettings({
        search_mode: searchMode,
        dense_top_k: denseTopK,
        sparse_top_k: sparseTopK,
        rrf_k: rrfK,
        dense_weight: denseWeight,
        sparse_weight: sparseWeight,
        enable_reranking: enableReranking,
        final_top_k: finalTopK,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      console.error('Failed to update retrieval settings:', err);
      setError('Failed to update retrieval settings. Check backend connection.');
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    setSearchMode('hybrid');
    setDenseTopK(20);
    setSparseTopK(20);
    setRrfK(60);
    setDenseWeight(1.0);
    setSparseWeight(1.0);
    setEnableReranking(true);
    setFinalTopK(8);
  };

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">
            Retrieval Settings
          </h1>
          <p className="text-xs text-neutral-400 mt-1">
            Tune hybrid search fusion, Reciprocal Rank Fusion constants, and cross-encoder reranking.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={handleReset}
            className="inline-flex items-center gap-1.5 rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-neutral-400 hover:text-white transition-colors"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset Defaults
          </button>
        </div>
      </div>

      {error && (
        <div className="rounded-xl border border-red-500/20 bg-red-950/20 p-3 text-xs text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex items-center gap-2 text-xs text-neutral-500 py-12 justify-center">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>Loading retrieval configuration…</span>
        </div>
      ) : (
        <form onSubmit={handleSave} className="space-y-6 max-w-2xl">
          {/* Search Strategy */}
          <div className="rounded-xl border border-white/10 bg-black/30 p-5 space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
              Search Strategy Mode
            </h3>

            <div className="grid grid-cols-3 gap-3">
              {[
                {
                  id: 'hybrid',
                  title: 'Hybrid Fusion',
                  desc: 'Dense + BM25 FTS5 with RRF',
                },
                {
                  id: 'dense',
                  title: 'Dense Only',
                  desc: 'Vector cosine similarity',
                },
                {
                  id: 'sparse',
                  title: 'Sparse Only',
                  desc: 'BM25 lexical keyword index',
                },
              ].map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setSearchMode(m.id)}
                  className={`rounded-lg border p-3 text-left transition-all ${
                    searchMode === m.id
                      ? 'border-blue-500 bg-blue-500/10 text-white shadow-sm ring-1 ring-blue-500/30'
                      : 'border-white/10 bg-white/[0.02] text-neutral-400 hover:border-white/20'
                  }`}
                >
                  <span className="text-xs font-semibold block">{m.title}</span>
                  <span className="text-[10px] text-neutral-500 mt-0.5 block">
                    {m.desc}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Fusion & Candidate Top-K Parameters */}
          <div className="rounded-xl border border-white/10 bg-black/30 p-5 space-y-5">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
              Candidate Retrieval & RRF Fusion
            </h3>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-neutral-300">
                  Dense Top-K ({denseTopK})
                </label>
                <input
                  type="range"
                  min={5}
                  max={50}
                  value={denseTopK}
                  onChange={(e) => setDenseTopK(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />
                <span className="text-[10px] text-neutral-500 block">
                  Candidate vector chunks
                </span>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-neutral-300">
                  Sparse Top-K ({sparseTopK})
                </label>
                <input
                  type="range"
                  min={5}
                  max={50}
                  value={sparseTopK}
                  onChange={(e) => setSparseTopK(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />
                <span className="text-[10px] text-neutral-500 block">
                  Candidate BM25 chunks
                </span>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-neutral-300">
                  RRF Constant k ({rrfK})
                </label>
                <input
                  type="range"
                  min={10}
                  max={100}
                  value={rrfK}
                  onChange={(e) => setRrfK(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />
                <span className="text-[10px] text-neutral-500 block">
                  Rank fusion smoothing factor
                </span>
              </div>
            </div>

            {/* Dense vs. Sparse Weight Sliders */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-white/5">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-neutral-300">
                  Dense Weight ({denseWeight.toFixed(2)})
                </label>
                <input
                  type="range"
                  min={0.1}
                  max={2.0}
                  step={0.05}
                  value={denseWeight}
                  onChange={(e) => setDenseWeight(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />
                <span className="text-[10px] text-neutral-500 block">
                  Relative weight for vector similarity
                </span>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-neutral-300">
                  Sparse Weight ({sparseWeight.toFixed(2)})
                </label>
                <input
                  type="range"
                  min={0.1}
                  max={2.0}
                  step={0.05}
                  value={sparseWeight}
                  onChange={(e) => setSparseWeight(Number(e.target.value))}
                  className="w-full accent-blue-500"
                />
                <span className="text-[10px] text-neutral-500 block">
                  Relative weight for BM25 lexical match
                </span>
              </div>
            </div>
          </div>

          {/* Reranker & Final Top-K */}
          <div className="rounded-xl border border-white/10 bg-black/30 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <h3 className="text-xs font-semibold text-white">
                  Cross-Encoder Reranking
                </h3>
                <p className="text-[11px] text-neutral-400">
                  Rescore candidate chunks with local cross-encoder models for maximal citation precision.
                </p>
              </div>
              <input
                type="checkbox"
                checked={enableReranking}
                onChange={(e) => setEnableReranking(e.target.checked)}
                className="h-4 w-4 rounded accent-blue-600"
              />
            </div>

            <div className="space-y-1.5 pt-2 border-t border-white/5">
              <label className="text-xs font-medium text-neutral-300">
                Final Context Chunks to LLM ({finalTopK})
              </label>
              <input
                type="range"
                min={2}
                max={25}
                value={finalTopK}
                onChange={(e) => setFinalTopK(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
              <span className="text-[10px] text-neutral-500 block">
                Top-N highest scoring chunks included in the context prompt
              </span>
            </div>
          </div>

          {/* Save Button */}
          <div className="flex items-center justify-end gap-3 pt-2">
            {saved && (
              <span className="flex items-center gap-1.5 text-xs text-green-400">
                <CheckCircle2 className="h-4 w-4" />
                Settings saved successfully
              </span>
            )}
            <button
              type="submit"
              disabled={saving}
              className="inline-flex items-center gap-1.5 rounded-xl bg-blue-600 px-5 py-2 text-xs font-semibold text-white shadow-md shadow-blue-600/30 hover:bg-blue-500 disabled:opacity-50 transition-colors"
            >
              {saving && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              <span>{saving ? 'Saving…' : 'Save Retrieval Settings'}</span>
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
