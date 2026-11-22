"use client";

import React from 'react';
import {
  Search,
  Layers,
  Terminal,
  Sparkles,
  Loader2,
  CheckCircle2,
  Cpu,
} from 'lucide-react';

interface StreamingStageIndicatorProps {
  currentStage: string | null;
  isStreaming: boolean;
  className?: string;
}

const STAGES = [
  {
    key: 'search',
    label: 'Searching Dense & FTS5 Indexes',
    matcher: (s: string) => /search|dense|fts|index|retriev/i.test(s),
    icon: Search,
    color: 'text-blue-400 bg-blue-500/10 border-blue-500/20',
  },
  {
    key: 'rerank',
    label: 'Reranking Candidates',
    matcher: (s: string) => /rerank|cross-encoder|score/i.test(s),
    icon: Layers,
    color: 'text-teal-400 bg-teal-500/10 border-teal-500/20',
  },
  {
    key: 'analyst',
    label: 'Python Data Analysis',
    matcher: (s: string) => /python|analys|data|sandbox|execut/i.test(s),
    icon: Terminal,
    color: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  },
  {
    key: 'generate',
    label: 'Generating Cited Response',
    matcher: (s: string) => /generat|cited|synthes|answer|reason/i.test(s),
    icon: Sparkles,
    color: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
  },
];

export function StreamingStageIndicator({
  currentStage,
  isStreaming,
  className = '',
}: StreamingStageIndicatorProps) {
  if (!isStreaming && !currentStage) return null;

  const displayStage = currentStage || 'Analyzing query & indexes…';
  const matchedStage = STAGES.find((st) => st.matcher(displayStage));
  const Icon = matchedStage?.icon || Cpu;
  const stageTheme =
    matchedStage?.color || 'text-blue-400 bg-blue-500/10 border-blue-500/20';

  return (
    <div
      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs backdrop-blur-md transition-all duration-300 shadow-sm ${stageTheme} ${className}`}
      role="status"
      aria-live="polite"
    >
      {isStreaming ? (
        <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0 text-current" />
      ) : (
        <Icon className="h-3.5 w-3.5 shrink-0 text-current" />
      )}
      <span className="font-medium tracking-tight text-neutral-200">
        {displayStage}
      </span>
      {isStreaming && (
        <span className="flex h-1.5 w-1.5 rounded-full bg-current animate-ping" />
      )}
    </div>
  );
}
