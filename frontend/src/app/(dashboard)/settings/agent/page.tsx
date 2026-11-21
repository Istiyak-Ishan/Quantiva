"use client";

import React, { useState } from 'react';
import { Workflow, CheckCircle2, Bot, Shield, Wrench } from 'lucide-react';

export default function AgentSettingsPage() {
  const [enableAgentic, setEnableAgentic] = useState(true);
  const [maxIterations, setMaxIterations] = useState(5);
  const [citationStrictness, setCitationStrictness] = useState<'strict' | 'balanced' | 'lenient'>('strict');
  const [enableCalculator, setEnableCalculator] = useState(true);
  const [enableDataAnalyst, setEnableDataAnalyst] = useState(true);
  const [saved, setSaved] = useState(false);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">Agent Settings</h1>
        <p className="text-xs text-neutral-400 mt-1">
          Configure ReAct reasoning loops, Data Analyst capabilities, and citation guardrails.
        </p>
      </div>

      <form onSubmit={handleSave} className="space-y-6 max-w-2xl">
        {/* Agentic Reasoning Mode */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <Bot className="h-4 w-4 text-blue-400" />
              <div>
                <h3 className="text-xs font-semibold text-white">ReAct Agent Execution</h3>
                <p className="text-[11px] text-neutral-400">
                  Allow the model to formulate thoughts, call retrieval tools, and verify observations iteratively.
                </p>
              </div>
            </div>
            <input
              type="checkbox"
              checked={enableAgentic}
              onChange={(e) => setEnableAgentic(e.target.checked)}
              className="h-4 w-4 rounded accent-blue-600"
            />
          </div>

          {enableAgentic && (
            <div className="space-y-1.5 pt-2">
              <label className="text-xs font-medium text-neutral-300">
                Max Reasoning Iterations ({maxIterations})
              </label>
              <input
                type="range"
                min={2}
                max={10}
                value={maxIterations}
                onChange={(e) => setMaxIterations(Number(e.target.value))}
                className="w-full accent-blue-500"
              />
              <span className="text-[10px] text-neutral-500 block">
                Upper bound on multi-hop tool invocations per user question.
              </span>
            </div>
          )}
        </div>

        {/* Built-in Agent Tools */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-5 space-y-4">
          <div className="flex items-center gap-2">
            <Wrench className="h-4 w-4 text-teal-400" />
            <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
              Agent Tools & Capabilities
            </h3>
          </div>

          <div className="space-y-3">
            <label className="flex items-center justify-between cursor-pointer rounded-lg border border-white/5 bg-white/[0.02] p-3">
              <div>
                <span className="text-xs font-medium text-white block">Local Document Retrieval</span>
                <span className="text-[10px] text-neutral-500">Query vector & lexical indexes in active knowledge base</span>
              </div>
              <input type="checkbox" checked disabled className="h-4 w-4 rounded accent-teal-600 cursor-not-allowed" />
            </label>

            <label className="flex items-center justify-between cursor-pointer rounded-lg border border-white/5 bg-white/[0.02] p-3">
              <div>
                <span className="text-xs font-medium text-white block">Safe Python / Sandbox Calculator</span>
                <span className="text-[10px] text-neutral-500">Perform exact mathematical calculations without LLM hallucinations</span>
              </div>
              <input
                type="checkbox"
                checked={enableCalculator}
                onChange={(e) => setEnableCalculator(e.target.checked)}
                className="h-4 w-4 rounded accent-blue-600"
              />
            </label>

            <label className="flex items-center justify-between cursor-pointer rounded-lg border border-white/5 bg-white/[0.02] p-3">
              <div>
                <span className="text-xs font-medium text-white block">Data Analyst & Tabular Engine</span>
                <span className="text-[10px] text-neutral-500">Analyze CSV, Excel, and structured tables with aggregations</span>
              </div>
              <input
                type="checkbox"
                checked={enableDataAnalyst}
                onChange={(e) => setEnableDataAnalyst(e.target.checked)}
                className="h-4 w-4 rounded accent-blue-600"
              />
            </label>
          </div>
        </div>

        {/* Citation Guardrails */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-5 space-y-4">
          <div className="flex items-center gap-2">
            <Shield className="h-4 w-4 text-purple-400" />
            <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
              Citation Strictness
            </h3>
          </div>

          <div className="grid grid-cols-3 gap-3">
            {(['strict', 'balanced', 'lenient'] as const).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setCitationStrictness(s)}
                className={`rounded-lg border p-3 text-left transition-all ${
                  citationStrictness === s
                    ? 'border-purple-500 bg-purple-500/10 text-white'
                    : 'border-white/10 bg-white/[0.02] text-neutral-400 hover:border-white/20'
                }`}
              >
                <span className="text-xs font-semibold capitalize block">{s}</span>
                <span className="text-[10px] text-neutral-500 mt-0.5 block">
                  {s === 'strict'
                    ? 'Every claim requires exact chunk citation'
                    : s === 'balanced'
                    ? 'General summary allowed with references'
                    : 'Permissive synthesis'}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Save */}
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
