"use client";

/**
 * MessageBubble — renders a single chat turn with:
 *   • Right-aligned blue bubble for user messages
 *   • Left-aligned glassmorphic bubble for assistant messages
 *   • Streaming text with fade-in tokens and blinking cursor
 *   • Expandable citation badges
 *   • A sliding citation panel from the right edge
 *   • Confidence meter
 *   • Chart display
 */

import React, { useState, useRef, useEffect, memo } from 'react';
import Image from 'next/image';
import { Message, Citation } from '@/store/useChatStore';
import { cn } from '@/lib/utils';
import {
  FileText,
  ChevronRight,
  BarChart2,
  ShieldAlert,
  Brain,
  X,
  ExternalLink,
  Zap,
} from 'lucide-react';

// ─── CITATION PANEL ───────────────────────────────────────────────────────────

interface CitationPanelProps {
  citation: Citation;
  index: number;
  onClose: () => void;
}

function CitationPanel({ citation, index, onClose }: CitationPanelProps) {
  // Close on Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  return (
    <div
      className="citation-panel-enter fixed right-0 top-0 h-full w-[360px] z-50 flex flex-col"
      role="dialog"
      aria-label={`Citation ${index + 1} from ${citation.filename}`}
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 -left-[100vw]"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Panel */}
      <div className="relative h-full glass-strong border-l border-neutral-700/60 flex flex-col shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-neutral-700/60">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-md bg-blue-500/20 border border-blue-500/30 flex items-center justify-center">
              <FileText className="w-3.5 h-3.5 text-blue-400" />
            </div>
            <div>
              <p className="text-sm font-semibold text-neutral-100 truncate max-w-[200px]">
                {citation.filename}
              </p>
              <p className="text-xs text-neutral-500">
                Page / Row {citation.page_or_row}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-md text-neutral-400 hover:text-neutral-100 hover:bg-neutral-700/50 transition-colors"
            aria-label="Close citation panel"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Source excerpt */}
        <div className="flex-1 overflow-y-auto p-5">
          <div className="mb-3 flex items-center gap-2">
            <div className="h-px flex-1 bg-neutral-700/50" />
            <span className="text-[10px] font-semibold text-neutral-500 uppercase tracking-widest">
              Source Text
            </span>
            <div className="h-px flex-1 bg-neutral-700/50" />
          </div>
          <blockquote className="relative pl-4 border-l-2 border-blue-500/60">
            <p className="text-sm text-neutral-300 leading-relaxed italic">
              "{citation.text_excerpt}"
            </p>
          </blockquote>
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t border-neutral-700/60">
          <div className="flex items-center justify-between">
            <span className="text-xs text-neutral-600">
              Source #{index + 1}
            </span>
            <button className="flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300 transition-colors">
              <ExternalLink className="w-3 h-3" />
              Open document
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── CONFIDENCE METER ─────────────────────────────────────────────────────────

function ConfidenceMeter({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color =
    pct >= 80 ? 'bg-emerald-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-500';

  return (
    <div className="flex items-center gap-2 mt-3">
      <Brain className="w-3.5 h-3.5 text-neutral-500" />
      <div className="flex-1 h-1.5 bg-neutral-700/50 rounded-full overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all duration-700', color)}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[11px] font-medium text-neutral-400 tabular-nums">
        {pct}%
      </span>
    </div>
  );
}

// ─── STREAMING TEXT ───────────────────────────────────────────────────────────
// Renders content and — when isStreaming — animates each new word with a
// fade-in and shows a blinking cursor at the end.

function StreamingText({
  content,
  isStreaming,
}: {
  content: string;
  isStreaming?: boolean;
}) {
  const words = content.split(/(\s+)/);

  return (
    <span className="text-[15px] leading-relaxed whitespace-pre-wrap">
      {words.map((word, i) => (
        <span
          key={i}
          style={{
            display: 'inline',
            animation: isStreaming ? `tokenFadeIn 0.12s ease-out both` : 'none',
            animationDelay: isStreaming ? `${Math.min(i, 30) * 8}ms` : '0ms',
          }}
        >
          {word}
        </span>
      ))}
      {isStreaming && <span className="streaming-cursor" aria-hidden="true" />}
    </span>
  );
}

// ─── CITATION BADGE ───────────────────────────────────────────────────────────

interface CitationBadgeProps {
  citation: Citation;
  index: number;
  onOpen: (cite: Citation, idx: number) => void;
  isActive: boolean;
}

function CitationBadge({ citation, index, onOpen, isActive }: CitationBadgeProps) {
  return (
    <button
      id={`citation-badge-${index}`}
      onClick={() => onOpen(citation, index)}
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium',
        'transition-all duration-150 border',
        isActive
          ? 'bg-blue-500/20 text-blue-300 border-blue-500/40 shadow-sm shadow-blue-500/20'
          : 'bg-neutral-800/60 text-neutral-400 border-neutral-700/60 hover:bg-neutral-700/50 hover:text-neutral-200 hover:border-neutral-600',
      )}
      aria-expanded={isActive}
      aria-label={`View source from ${citation.filename} page ${citation.page_or_row}`}
    >
      <FileText className="w-3 h-3 flex-shrink-0" />
      <span className="truncate max-w-[100px]">{citation.filename}</span>
      <span className="opacity-50 tabular-nums">p.{citation.page_or_row}</span>
      <ChevronRight
        className={cn('w-3 h-3 transition-transform duration-150 flex-shrink-0',
          isActive && 'rotate-90')}
      />
    </button>
  );
}

// ─── CACHE HIT BADGE ──────────────────────────────────────────────────────────

function CacheHitBadge() {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
      <Zap className="w-2.5 h-2.5" />
      Cached
    </span>
  );
}

// ─── MAIN MESSAGE BUBBLE ──────────────────────────────────────────────────────

interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
}

export const MessageBubble = memo(function MessageBubble({
  message,
  isStreaming,
}: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const [openCitation, setOpenCitation] = useState<{
    cite: Citation;
    idx: number;
  } | null>(null);
  const bubbleRef = useRef<HTMLDivElement>(null);

  const handleOpenCitation = (cite: Citation, idx: number) => {
    setOpenCitation((prev) =>
      prev?.idx === idx ? null : { cite, idx },
    );
  };

  const hasCitations = !isUser && !!message.citations?.length;
  const hasConfidence = typeof message.confidence === 'number';
  const hasChart = !!message.chart_url;
  const hasFlags = !!(message as any).guardrail_flags?.length;

  return (
    <>
      <div
        ref={bubbleRef}
        className={cn(
          'flex w-full message-enter',
          isUser ? 'justify-end' : 'justify-start',
        )}
      >
        {/* Avatar — assistant only */}
        {!isUser && (
          <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center mr-3 mt-0.5 shadow-lg">
            <Brain className="w-4 h-4 text-white" />
          </div>
        )}

        <div
          className={cn(
            'max-w-[80%] rounded-2xl px-5 py-4 shadow-lg',
            isUser
              ? [
                  'bg-gradient-to-br from-blue-600 to-blue-700 text-white rounded-br-md',
                  'shadow-blue-900/30',
                ]
              : [
                  'glass rounded-bl-md',
                  'text-neutral-100',
                  isStreaming && 'glow-blue',
                ],
          )}
        >
          {/* ── Message content ── */}
          <StreamingText content={message.content} isStreaming={isStreaming} />

          {/* ── Guardrail flags ── */}
          {hasFlags && (
            <div className="mt-3 p-2.5 bg-red-900/20 border border-red-900/40 rounded-lg">
              {(message as any).guardrail_flags.map((flag: string, i: number) => (
                <div key={i} className="flex items-center gap-2 text-xs text-red-400">
                  <ShieldAlert className="w-3.5 h-3.5 flex-shrink-0" />
                  {flag}
                </div>
              ))}
            </div>
          )}

          {/* ── Confidence meter ── */}
          {hasConfidence && !isStreaming && (
            <ConfidenceMeter score={message.confidence!} />
          )}

          {/* ── Chart ── */}
          {hasChart && (
            <div className="mt-4 rounded-xl overflow-hidden border border-neutral-700/60 bg-neutral-900/80">
              <div className="flex items-center gap-2 px-3 py-2 bg-neutral-800/60 border-b border-neutral-700/50">
                <BarChart2 className="w-4 h-4 text-blue-400" />
                <span className="text-xs font-medium text-neutral-300">
                  Generated Chart
                </span>
              </div>
              <div className="relative w-full aspect-video">
                <Image
                  src={message.chart_url!}
                  alt="AI-generated chart"
                  fill
                  className="object-contain p-2"
                />
              </div>
            </div>
          )}

          {/* ── Citations ── */}
          {hasCitations && !isStreaming && (
            <div className="mt-4 pt-3 border-t border-neutral-700/40">
              <div className="flex items-center gap-2 mb-2.5">
                <span className="text-[10px] font-semibold text-neutral-500 uppercase tracking-widest">
                  Sources
                </span>
                <span className="text-[10px] text-neutral-600">
                  {message.citations!.length} reference
                  {message.citations!.length !== 1 ? 's' : ''}
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {message.citations!.map((cite, i) => (
                  <CitationBadge
                    key={i}
                    citation={cite}
                    index={i}
                    onOpen={handleOpenCitation}
                    isActive={openCitation?.idx === i}
                  />
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Timestamp — shown on hover via group */}
        <div className="self-end mb-1 ml-2 opacity-0 group-hover:opacity-100 transition-opacity">
          {message.timestamp && (
            <span className="text-[10px] text-neutral-600">
              {new Date(message.timestamp).toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
              })}
            </span>
          )}
        </div>
      </div>

      {/* ── Sliding citation panel (portal-style, fixed position) ── */}
      {openCitation && (
        <CitationPanel
          citation={openCitation.cite}
          index={openCitation.idx}
          onClose={() => setOpenCitation(null)}
        />
      )}
    </>
  );
});
