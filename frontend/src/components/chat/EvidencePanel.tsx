"use client";

import React, { useEffect, useRef } from 'react';
import {
  FileText,
  X,
  ExternalLink,
  Layers,
  Sparkles,
  CheckCircle2,
  Table as TableIcon,
  Copy,
  Check,
  ChevronRight,
  Quote,
} from 'lucide-react';
import { RetrievedChunk, StructuredCitation } from '@/lib/api';

interface EvidencePanelProps {
  isOpen: boolean;
  onClose: () => void;
  chunks: RetrievedChunk[];
  citations: StructuredCitation[];
  activeCitationIndex: number | null;
  onSelectCitation?: (index: number) => void;
}

export function EvidencePanel({
  isOpen,
  onClose,
  chunks,
  citations,
  activeCitationIndex,
  onSelectCitation,
}: EvidencePanelProps) {
  const itemRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const [copiedId, setCopiedId] = React.useState<string | null>(null);

  // Auto-scroll to selected citation chunk
  useEffect(() => {
    if (activeCitationIndex !== null) {
      const activeCit = citations.find((c) => c.index === activeCitationIndex);
      if (activeCit && activeCit.chunk_id && itemRefs.current[activeCit.chunk_id]) {
        itemRefs.current[activeCit.chunk_id]?.scrollIntoView({
          behavior: 'smooth',
          block: 'center',
        });
      }
    }
  }, [activeCitationIndex, citations]);

  if (!isOpen) return null;

  const handleCopy = (id: string, text: string) => {
    void navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <aside
      className="flex flex-col h-full w-80 md:w-96 border-l border-white/10 bg-[#0c0c10] shadow-2xl z-20 shrink-0 select-text"
      role="complementary"
      aria-label="Supporting evidence and citations"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
        <div className="flex items-center gap-2.5">
          <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-blue-500/10 text-blue-400">
            <Layers className="h-4 w-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-white tracking-tight uppercase">
              Supporting Evidence
            </h3>
            <p className="text-[10px] text-neutral-400">
              {chunks.length} chunks retrieved • {citations.length} cited
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={onClose}
          className="rounded-lg p-1.5 text-neutral-400 hover:bg-white/5 hover:text-white transition-colors"
          title="Close Evidence Panel"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Citations Overview Filter Bar */}
      {citations.length > 0 && (
        <div className="px-5 py-3 border-b border-white/5 bg-black/40">
          <span className="text-[10px] uppercase font-semibold text-neutral-500 block mb-2">
            Cited References
          </span>
          <div className="flex flex-wrap gap-1.5">
            {citations.map((c) => {
              const isActive = activeCitationIndex === c.index;
              return (
                <button
                  key={c.marker}
                  type="button"
                  onClick={() => onSelectCitation?.(c.index)}
                  className={`inline-flex items-center gap-1 rounded-lg px-2 py-1 text-xs font-mono transition-all ${
                    isActive
                      ? 'bg-blue-600 text-white font-bold shadow-sm shadow-blue-600/30 ring-1 ring-blue-400'
                      : 'border border-white/10 bg-white/5 text-neutral-300 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  <span className="font-bold">{c.marker}</span>
                  <span className="truncate max-w-[120px] text-[10px] font-sans">
                    {c.source_file || 'Doc'}
                  </span>
                  {c.page_number !== null && c.page_number !== undefined && (
                    <span className="text-[9px] opacity-75">p.{c.page_number}</span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Chunks List */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {chunks.length === 0 ? (
          <div className="py-20 text-center text-xs text-neutral-500 space-y-2">
            <Layers className="mx-auto h-8 w-8 text-neutral-700" />
            <p>No supporting evidence for this turn.</p>
          </div>
        ) : (
          chunks.map((chunk, idx) => {
            const linkedCitation = citations.find(
              (c) => c.chunk_id === chunk.chunk_id || c.index === idx + 1
            );
            const isHighlighted =
              activeCitationIndex !== null &&
              linkedCitation?.index === activeCitationIndex;

            const filename =
              (chunk.metadata?.filename as string) ||
              (chunk.metadata?.source_file as string) ||
              linkedCitation?.source_file ||
              `Document Chunk #${idx + 1}`;
            const pageNum =
              (chunk.metadata?.page_number as number) ??
              linkedCitation?.page_number;

            return (
              <div
                key={chunk.chunk_id || idx}
                ref={(el) => {
                  itemRefs.current[chunk.chunk_id] = el;
                }}
                className={`rounded-xl border p-4 space-y-3 transition-all ${
                  isHighlighted
                    ? 'border-blue-500 bg-blue-500/10 shadow-lg shadow-blue-500/10 ring-1 ring-blue-500/40'
                    : 'border-white/10 bg-black/30 hover:border-white/20'
                }`}
              >
                {/* Chunk Header */}
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <FileText className="h-4 w-4 shrink-0 text-blue-400" />
                    <div className="min-w-0">
                      <span className="text-xs font-semibold text-white truncate block" title={filename}>
                        {filename}
                      </span>
                      {pageNum !== undefined && pageNum !== null && (
                        <span className="text-[10px] text-neutral-400">Page / Row {pageNum}</span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-1.5 shrink-0">
                    {linkedCitation && (
                      <span className="rounded bg-blue-500/20 px-1.5 py-0.5 text-[10px] font-bold font-mono text-blue-300 border border-blue-500/30">
                        {linkedCitation.marker}
                      </span>
                    )}

                    <button
                      type="button"
                      onClick={() => handleCopy(chunk.chunk_id, chunk.content)}
                      className="p-1 text-neutral-400 hover:text-white transition-colors"
                      title="Copy chunk text"
                    >
                      {copiedId === chunk.chunk_id ? (
                        <Check className="h-3.5 w-3.5 text-green-400" />
                      ) : (
                        <Copy className="h-3.5 w-3.5" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Score Badges Grid */}
                <div className="grid grid-cols-4 gap-1.5 rounded-lg bg-black/40 p-2 text-[9px] font-mono border border-white/5">
                  <div className="flex flex-col">
                    <span className="text-neutral-500">Rerank</span>
                    <span className="text-purple-400 font-bold">
                      {chunk.rerank_score !== undefined && chunk.rerank_score !== null
                        ? chunk.rerank_score.toFixed(3)
                        : '—'}
                    </span>
                  </div>

                  <div className="flex flex-col">
                    <span className="text-neutral-500">RRF</span>
                    <span className="text-teal-400 font-bold">
                      {chunk.rrf_score !== undefined && chunk.rrf_score !== null
                        ? chunk.rrf_score.toFixed(4)
                        : '—'}
                    </span>
                  </div>

                  <div className="flex flex-col">
                    <span className="text-neutral-500">Dense</span>
                    <span className="text-neutral-300 font-bold">
                      {chunk.dense_rank ? `#${chunk.dense_rank}` : '—'}
                    </span>
                  </div>

                  <div className="flex flex-col">
                    <span className="text-neutral-500">Sparse</span>
                    <span className="text-neutral-300 font-bold">
                      {chunk.sparse_rank ? `#${chunk.sparse_rank}` : '—'}
                    </span>
                  </div>
                </div>

                {/* Verbatim Citation Quote if present */}
                {linkedCitation?.quote_snippet && (
                  <div className="rounded-lg bg-blue-950/30 border border-blue-500/20 p-2.5 space-y-1">
                    <span className="text-[9px] uppercase font-bold text-blue-400 flex items-center gap-1">
                      <Quote className="h-3 w-3" /> Matched Verbatim Quote
                    </span>
                    <p className="text-xs text-blue-200/90 italic leading-relaxed">
                      &ldquo;{linkedCitation.quote_snippet}&rdquo;
                    </p>
                  </div>
                )}

                {/* Chunk Body Content */}
                <div className="rounded-lg bg-black/40 p-2.5 text-xs text-neutral-300 font-mono whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto selection:bg-blue-600">
                  {chunk.content}
                </div>
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
}
