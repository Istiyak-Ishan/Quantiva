"use client";

import React, { useState } from 'react';
import {
  Brain,
  ChevronDown,
  ChevronRight,
  Terminal,
  Search,
  CheckCircle2,
  Copy,
  Check,
  Code,
  Sparkles,
} from 'lucide-react';

interface ToolCall {
  tool: string;
  input: Record<string, unknown> | string;
  output?: string;
}

interface ReasoningTraceData {
  reasoning?: string;
  thoughts?: string[];
  tool_calls?: ToolCall[];
  [key: string]: unknown;
}

interface AgentReasoningTraceProps {
  reasoning?: string | null;
  reasoningTrace?: ReasoningTraceData | null;
  defaultOpen?: boolean;
  isStreaming?: boolean;
}

export function AgentReasoningTrace({
  reasoning,
  reasoningTrace,
  defaultOpen = false,
  isStreaming = false,
}: AgentReasoningTraceProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen || isStreaming);
  const [copied, setCopied] = useState(false);

  const rawReasoning = reasoning || reasoningTrace?.reasoning;
  const thoughts = reasoningTrace?.thoughts || [];
  const toolCalls = reasoningTrace?.tool_calls || [];

  if (!rawReasoning && thoughts.length === 0 && toolCalls.length === 0) {
    return null;
  }

  const handleCopy = () => {
    const textToCopy =
      rawReasoning ||
      JSON.stringify({ thoughts, toolCalls }, null, 2);
    void navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const stepCount =
    (thoughts.length || 0) +
    (toolCalls.length || 0) +
    (rawReasoning ? 1 : 0);

  return (
    <div className="my-2.5 rounded-xl border border-white/10 bg-black/40 overflow-hidden shadow-sm transition-all duration-200">
      {/* Accordion Header */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3.5 py-2 text-left bg-white/[0.02] hover:bg-white/[0.05] transition-colors"
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-2 text-xs">
          <div className="flex h-5 w-5 items-center justify-center rounded-md bg-purple-500/15 text-purple-400">
            <Brain className="h-3 w-3" />
          </div>
          <span className="font-semibold text-neutral-300">
            Agent Reasoning & Tool Trace
          </span>
          <span className="text-[10px] font-mono rounded-full bg-white/10 px-1.5 py-0.2 text-neutral-400">
            {stepCount} step{stepCount !== 1 ? 's' : ''}
          </span>
          {isStreaming && (
            <span className="text-[10px] text-purple-400 animate-pulse font-mono flex items-center gap-1">
              • thinking…
            </span>
          )}
        </div>

        <div className="flex items-center gap-1.5 text-neutral-400">
          {isOpen ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </div>
      </button>

      {/* Accordion Body */}
      {isOpen && (
        <div className="px-4 py-3 border-t border-white/5 space-y-3 text-xs bg-black/60">
          <div className="flex justify-end">
            <button
              type="button"
              onClick={handleCopy}
              className="inline-flex items-center gap-1 text-[11px] text-neutral-400 hover:text-white transition-colors"
              title="Copy reasoning trace"
            >
              {copied ? (
                <>
                  <Check className="h-3 w-3 text-emerald-400" />
                  <span className="text-emerald-400">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="h-3 w-3" />
                  <span>Copy Trace</span>
                </>
              )}
            </button>
          </div>

          {/* Tool Invocations if any */}
          {toolCalls.length > 0 && (
            <div className="space-y-2">
              <span className="text-[10px] uppercase font-bold tracking-wider text-neutral-500">
                Executed Tools & Sandboxed Actions
              </span>
              {toolCalls.map((tc, idx) => {
                const isPython =
                  tc.tool.toLowerCase().includes('python') ||
                  tc.tool.toLowerCase().includes('analyst');
                const isSearch =
                  tc.tool.toLowerCase().includes('search') ||
                  tc.tool.toLowerCase().includes('retriev');

                const inputStr =
                  typeof tc.input === 'string'
                    ? tc.input
                    : JSON.stringify(tc.input, null, 2);

                return (
                  <div
                    key={idx}
                    className="rounded-lg border border-white/10 bg-neutral-950/80 p-2.5 space-y-1.5"
                  >
                    <div className="flex items-center justify-between text-[11px]">
                      <div className="flex items-center gap-1.5 font-mono font-medium">
                        {isPython ? (
                          <Terminal className="h-3.5 w-3.5 text-amber-400" />
                        ) : isSearch ? (
                          <Search className="h-3.5 w-3.5 text-blue-400" />
                        ) : (
                          <Code className="h-3.5 w-3.5 text-teal-400" />
                        )}
                        <span className="text-white">{tc.tool}</span>
                      </div>
                      <span className="text-[10px] text-neutral-500">
                        Call #{idx + 1}
                      </span>
                    </div>

                    {/* Tool Input / Script */}
                    <pre className="font-mono text-[10px] text-neutral-300 bg-black/60 p-2 rounded overflow-x-auto whitespace-pre-wrap max-h-48 border border-white/5">
                      {inputStr}
                    </pre>

                    {/* Tool Observation */}
                    {tc.output && (
                      <div className="pt-1">
                        <span className="text-[9px] uppercase font-bold text-neutral-500">
                          Observation
                        </span>
                        <pre className="font-mono text-[10px] text-emerald-300/90 bg-emerald-950/20 p-2 rounded overflow-x-auto whitespace-pre-wrap max-h-36 border border-emerald-500/20">
                          {tc.output}
                        </pre>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Sequential Thoughts */}
          {thoughts.length > 0 && (
            <div className="space-y-1.5">
              <span className="text-[10px] uppercase font-bold tracking-wider text-neutral-500">
                Thought Process
              </span>
              <ul className="space-y-1.5 list-none pl-0">
                {thoughts.map((th, i) => (
                  <li
                    key={i}
                    className="text-neutral-300 text-[11px] leading-relaxed pl-3 border-l-2 border-purple-500/40 italic"
                  >
                    {th}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Raw Text Reasoning */}
          {rawReasoning && (
            <div className="space-y-1">
              <span className="text-[10px] uppercase font-bold tracking-wider text-neutral-500">
                Reasoning Trace
              </span>
              <div className="text-neutral-300 text-[11px] leading-relaxed whitespace-pre-wrap pl-3 border-l-2 border-neutral-700 font-mono bg-neutral-900/40 p-2 rounded">
                {rawReasoning}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
