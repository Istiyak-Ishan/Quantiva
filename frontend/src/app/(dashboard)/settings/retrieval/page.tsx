"use client";

import React, { useState } from 'react';
import { Sliders, CheckCircle2, RotateCcw } from 'lucide-react';

export default function RetrievalSettingsPage() {
  const [searchMode, setSearchMode] = useState<'hybrid' | 'dense' | 'sparse'>('hybrid');
  const [denseTopK, setDenseTopK] = useState(20);
  const [sparseTopK, setSparseTopK] = useState(20);
  const [rrfK, setRrfK] = useState(60);
  const [enableReranking, setEnableReranking] = useState(true);
  const [finalTopK, setFinalTopK] = useState(8);
  const [enableAgentic, setEnableAgentic] = useState(true);
  const [maxIterations, setMaxIterations] = useState(5);
  const [saved, setSaved] = useState(false);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };

  const handleReset = () => {
    setSearchMode('hybrid');
    setDenseTopK(20);
    setSparseTopK(20);
    setRrfK(60);
    setEnableReranking(true);
    setFinalTopK(8);
    setEnableAgentic(true);
    setMaxIterations(5);
  };

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">Retrieval Settings</h1>
          <p className="text-xs text-neutral-400 mt-1">
            Tune hybrid search fusion, Reciprocal Rank Fusion constants, and cross-encoder rerankers.
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

      <form onSubmit={handleSave} className="space-y-6 max-w-2xl">
        {/* Search Mode */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-5 space-y-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">Search Strategy</h3>

          <div className="grid grid-cols-3 gap-3">
            {(['hybrid', 'dense', 'sparse'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setSearchMode(m)}
                className={`rounded-lg border p-3 text-left transition-all ${
                  searchMode === m
                    ? 'border-blue-500 bg-blue-500/10 text-white'
                    : 'border-white/10 bg-white/[0.02] text-neutral-400 hover:border-white/20'
                }`}
              >
                <span className="text-xs font-semibold capitalize block">{m}</span>
                <span className="text-[10px] text-neutral-500 mt-0.5 block">
                  {m === 'hybrid'
                    ? 'Dense + BM25 FTS5 with RRF'
                    : m === 'dense'
                    ? 'Vector cosine similarity'
                    : 'Keyword BM25 index'}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Fusion & Top-K Parameters */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-5 space-y-5">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">Fusion Parameters</h3>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-neutral-300">Dense Top-K ({denseTopK})</label>
              <input
                type="range"
                min={5}
                max={50}
                value={denseTopK}
                onChange={(e) => setDenseTopK(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
              <span className="text-[10px] text-neutral-500 block">Candidate vector chunks</span>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-neutral-300">Sparse Top-K ({sparseTopK})</label>
              <input
                type="range"
                min={5}
                max={50}
                value={sparseTopK}
                onChange={(e) => setSparseTopK(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
              <span className="text-[10px] text-neutral-500 block">Candidate BM25 chunks</span>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-neutral-300">RRF Constant k ({rrfK})</label>
              <input
                type="range"
                min={10}
                max={100}
                value={rrfK}
                onChange={(e) => setRrfK(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
              <span className="text-[10px] text-neutral-500 block">Rank fusion smoothing factor</span>
            </div>
          </div>
        </div>

        {/* Reranker & Final Top-K */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <h3 className="text-xs font-semibold text-white">Cross-Encoder Reranking</h3>
              <p className="text-[11px] text-neutral-400">
                Score candidate chunks using a local cross-encoder model before sending to context.
              </p>
            </div>
            <input
              type="checkbox"
              checked={enableReranking}
              onChange={(e) => setEnableReranking(e.target.checked)}
              className="h-4 w-4 rounded accent-blue-600"
            />
          </div>

          {enableReranking && (
            <div className="space-y-1.5 pt-2">
              <label className="text-xs font-medium text-neutral-300">Final Top-K to LLM ({finalTopK})</label>
              <input
                type="range"
                min={2}
                max={20}
                value={finalTopK}
                onChange={(e) => setFinalTopK(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
              <span className="text-[10px] text-neutral-500 block">Chunks passed into prompt context</span>
            </div>
          )}
        </div>

        {/* Save Button */}
        <div className="flex items-center justify-end gap-3 pt-2">
          {saved && (
            <span className="flex items-center gap-1.5 text-xs text-green-400">
              <CheckCircle2 className="h-4 w-4" />
              Settings saved
            </span>
          )}
          <button
            type="submit"
            className="rounded-xl bg-blue-600 px-5 py-2 text-xs font-semibold text-white shadow-md shadow-blue-600/30 hover:bg-blue-500 transition-colors"
          >
            Save Settings
          </button>
        </div>
      </form>
    </div>
  );
}
