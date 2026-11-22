"use client";

import React, { useEffect, useState } from 'react';
import {
  Bot,
  Wrench,
  Shield,
  CheckCircle2,
  Loader2,
  Terminal,
  Database,
  Sparkles,
} from 'lucide-react';
import {
  getRetrievalSettings,
  updateRetrievalSettings,
} from '@/lib/api';

export default function AgentSettingsPage() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Agent configuration fields
  const [enableAgentic, setEnableAgentic] = useState(true);
  const [maxIterations, setMaxIterations] = useState(5);
  const [enablePythonSandbox, setEnablePythonSandbox] = useState(true);
  const [enableDataAnalyst, setEnableDataAnalyst] = useState(true);
  const [citationStrictness, setCitationStrictness] = useState<
    'strict' | 'balanced' | 'lenient'
  >('strict');

  useEffect(() => {
    async function loadSettings() {
      try {
        setLoading(true);
        const cfg = await getRetrievalSettings();
        setEnableAgentic(cfg.enable_agentic_mode ?? true);
        setMaxIterations(cfg.max_agent_iterations ?? 5);
      } catch (err) {
        console.error('Failed to load agent settings:', err);
        setError('Could not load agent configuration from backend.');
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
        enable_agentic_mode: enableAgentic,
        max_agent_iterations: maxIterations,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      console.error('Failed to update agent settings:', err);
      setError('Failed to update agent settings.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">
          Agent Settings
        </h1>
        <p className="text-xs text-neutral-400 mt-1">
          Configure ReAct reasoning loops, Sandboxed Python Data Analyst tools, and citation strictness.
        </p>
      </div>

      {error && (
        <div className="rounded-xl border border-red-500/20 bg-red-950/20 p-3 text-xs text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <div className="flex items-center gap-2 text-xs text-neutral-500 py-12 justify-center">
          <Loader2 className="h-4 w-4 animate-spin" />
          <span>Loading agent configuration…</span>
        </div>
      ) : (
        <form onSubmit={handleSave} className="space-y-6 max-w-2xl">
          {/* Agentic Reasoning Mode */}
          <div className="rounded-xl border border-white/10 bg-black/30 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-purple-500/15 text-purple-400">
                  <Bot className="h-4 w-4" />
                </div>
                <div>
                  <h3 className="text-xs font-semibold text-white">
                    ReAct Agent Reasoning Loop
                  </h3>
                  <p className="text-[11px] text-neutral-400">
                    Enable autonomous multi-step reasoning, query reformulation, and observation verification.
                  </p>
                </div>
              </div>
              <input
                type="checkbox"
                checked={enableAgentic}
                onChange={(e) => setEnableAgentic(e.target.checked)}
                className="h-4 w-4 rounded accent-purple-600"
              />
            </div>

            {enableAgentic && (
              <div className="space-y-1.5 pt-3 border-t border-white/5">
                <div className="flex items-center justify-between text-xs">
                  <label className="font-medium text-neutral-300">
                    Max Reasoning Iterations ({maxIterations})
                  </label>
                  <span className="font-mono text-purple-400">{maxIterations} hops</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={10}
                  value={maxIterations}
                  onChange={(e) => setMaxIterations(Number(e.target.value))}
                  className="w-full accent-purple-500"
                />
                <span className="text-[10px] text-neutral-500 block">
                  Upper bound on multi-hop tool invocations per user question (1 to 10).
                </span>
              </div>
            )}
          </div>

          {/* Built-in Agent Tools */}
          <div className="rounded-xl border border-white/10 bg-black/30 p-5 space-y-4">
            <div className="flex items-center gap-2">
              <Wrench className="h-4 w-4 text-teal-400" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
                Agent Tools & Sandboxed Capabilities
              </h3>
            </div>

            <div className="space-y-3">
              <label className="flex items-center justify-between rounded-lg border border-white/5 bg-white/[0.02] p-3">
                <div className="flex items-center gap-2.5">
                  <Database className="h-4 w-4 text-blue-400" />
                  <div>
                    <span className="text-xs font-medium text-white block">
                      Local Document Retrieval Tool
                    </span>
                    <span className="text-[10px] text-neutral-500">
                      Query vector & lexical indexes in active knowledge base
                    </span>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked
                  disabled
                  className="h-4 w-4 rounded accent-teal-600 cursor-not-allowed opacity-75"
                />
              </label>

              <label className="flex items-center justify-between rounded-lg border border-white/5 bg-white/[0.02] p-3 cursor-pointer">
                <div className="flex items-center gap-2.5">
                  <Terminal className="h-4 w-4 text-amber-400" />
                  <div>
                    <span className="text-xs font-medium text-white block">
                      Sandboxed Python Data Analyst Tool
                    </span>
                    <span className="text-[10px] text-neutral-500">
                      Execute sandboxed Python arithmetic, aggregations, and financial math
                    </span>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={enablePythonSandbox}
                  onChange={(e) => setEnablePythonSandbox(e.target.checked)}
                  className="h-4 w-4 rounded accent-blue-600 cursor-pointer"
                />
              </label>

              <label className="flex items-center justify-between rounded-lg border border-white/5 bg-white/[0.02] p-3 cursor-pointer">
                <div className="flex items-center gap-2.5">
                  <Sparkles className="h-4 w-4 text-purple-400" />
                  <div>
                    <span className="text-xs font-medium text-white block">
                      Data Analyst & Tabular Engine
                    </span>
                    <span className="text-[10px] text-neutral-500">
                      Analyze Excel sheets, CSV rows, and structured Markdown tables
                    </span>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={enableDataAnalyst}
                  onChange={(e) => setEnableDataAnalyst(e.target.checked)}
                  className="h-4 w-4 rounded accent-blue-600 cursor-pointer"
                />
              </label>
            </div>
          </div>

          {/* Citation Guardrails & Strictness */}
          <div className="rounded-xl border border-white/10 bg-black/30 p-5 space-y-4">
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-purple-400" />
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
                Citation Strictness & Hallucination Prevention
              </h3>
            </div>

            <div className="grid grid-cols-3 gap-3">
              {[
                {
                  id: 'strict',
                  title: 'Strict Verification',
                  desc: 'Requires verbatim lexical quote overlap for every factual claim.',
                },
                {
                  id: 'balanced',
                  title: 'Balanced',
                  desc: 'Validates semantic chunk alignment with lenient threshold.',
                },
                {
                  id: 'lenient',
                  title: 'Lenient',
                  desc: 'Synthesizes freely when context contains relevant background.',
                },
              ].map((lvl) => (
                <button
                  key={lvl.id}
                  type="button"
                  onClick={() =>
                    setCitationStrictness(lvl.id as 'strict' | 'balanced' | 'lenient')
                  }
                  className={`rounded-lg border p-3 text-left transition-all ${
                    citationStrictness === lvl.id
                      ? 'border-purple-500 bg-purple-500/10 text-white shadow-sm ring-1 ring-purple-500/30'
                      : 'border-white/10 bg-white/[0.02] text-neutral-400 hover:border-white/20'
                  }`}
                >
                  <span className="text-xs font-semibold capitalize block">
                    {lvl.title}
                  </span>
                  <span className="text-[10px] text-neutral-500 mt-1 block leading-relaxed">
                    {lvl.desc}
                  </span>
                </button>
              ))}
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
              className="inline-flex items-center gap-1.5 rounded-xl bg-purple-600 px-5 py-2 text-xs font-semibold text-white shadow-md shadow-purple-600/30 hover:bg-purple-500 disabled:opacity-50 transition-colors"
            >
              {saving && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
              <span>{saving ? 'Saving…' : 'Save Agent Settings'}</span>
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
