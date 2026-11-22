"use client";

import React, { useEffect, useState } from 'react';
import {
  X,
  Layers,
  FileText,
  Table as TableIcon,
  Search,
  CheckCircle2,
  Copy,
  Check,
  Loader2,
  BookOpen,
} from 'lucide-react';
import { DocumentChunk, getDocumentChunks, DocumentItem } from '@/lib/api';

interface ChunkInspectorDrawerProps {
  document: DocumentItem | null;
  isOpen: boolean;
  onClose: () => void;
}

export function ChunkInspectorDrawer({
  document,
  isOpen,
  onClose,
}: ChunkInspectorDrawerProps) {
  const [chunks, setChunks] = useState<DocumentChunk[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState('');
  const [onlyTables, setOnlyTables] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen || !document) {
      setChunks([]);
      return;
    }

    let active = true;
    async function loadChunks() {
      if (!document) return;
      setLoading(true);
      setError(null);
      try {
        const res = await getDocumentChunks(document.id, 200, 0);
        if (active) {
          setChunks(res.chunks || []);
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : 'Failed to load document chunks');
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    void loadChunks();
    return () => {
      active = false;
    };
  }, [isOpen, document]);

  if (!isOpen || !document) return null;

  const filteredChunks = chunks.filter((c) => {
    if (onlyTables && !c.is_table) return false;
    if (!filter) return true;
    const q = filter.toLowerCase();
    return (
      c.content.toLowerCase().includes(q) ||
      (c.section_title && c.section_title.toLowerCase().includes(q)) ||
      (c.sheet_name && c.sheet_name.toLowerCase().includes(q))
    );
  });

  const handleCopy = (id: string, text: string) => {
    void navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/60 backdrop-blur-sm">
      {/* Background click to dismiss */}
      <div className="absolute inset-0" onClick={onClose} />

      {/* Drawer Panel */}
      <div className="absolute inset-y-0 right-0 flex max-w-full pl-10">
        <div className="relative w-screen max-w-2xl border-l border-white/10 bg-[#0c0c10] shadow-2xl flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-white/10 px-6 py-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-blue-500/10 text-blue-400">
                <Layers className="h-5 w-5" />
              </div>
              <div className="min-w-0">
                <h2 className="text-sm font-bold text-white truncate" title={document.filename}>
                  {document.filename}
                </h2>
                <div className="flex items-center gap-2 text-[11px] text-neutral-400 mt-0.5">
                  <span>{chunks.length} extracted chunks</span>
                  <span>•</span>
                  <span>Type: {document.file_type.toUpperCase()}</span>
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={onClose}
              className="flex h-8 w-8 items-center justify-center rounded-lg text-neutral-400 hover:bg-white/5 hover:text-white transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Filter & Search Bar */}
          <div className="flex items-center gap-3 border-b border-white/5 bg-black/40 px-6 py-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-neutral-500" />
              <input
                type="text"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                placeholder="Search chunks by keyword or section…"
                className="w-full rounded-lg border border-white/10 bg-neutral-900 pl-8 pr-3 py-1.5 text-xs text-white placeholder-neutral-500 focus:border-blue-500 focus:outline-none"
              />
            </div>

            <button
              type="button"
              onClick={() => setOnlyTables(!onlyTables)}
              className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
                onlyTables
                  ? 'border-purple-500 bg-purple-500/20 text-purple-300'
                  : 'border-white/10 bg-white/5 text-neutral-400 hover:text-white'
              }`}
            >
              <TableIcon className="h-3.5 w-3.5" />
              Tables Only
            </button>
          </div>

          {/* Chunks List */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {loading ? (
              <div className="flex flex-col items-center justify-center py-20 text-neutral-500 space-y-3">
                <Loader2 className="h-6 w-6 animate-spin text-blue-400" />
                <span className="text-xs">Loading chunk vector representations…</span>
              </div>
            ) : error ? (
              <div className="rounded-xl border border-red-500/20 bg-red-950/20 p-4 text-xs text-red-300">
                {error}
              </div>
            ) : filteredChunks.length === 0 ? (
              <div className="rounded-xl border border-dashed border-white/10 p-12 text-center text-xs text-neutral-500">
                {filter || onlyTables
                  ? 'No chunks match the current filter criteria.'
                  : 'No chunks available for this document.'}
              </div>
            ) : (
              filteredChunks.map((chunk) => (
                <div
                  key={chunk.id}
                  className="rounded-xl border border-white/10 bg-black/30 p-4 space-y-3 hover:border-white/20 transition-all"
                >
                  {/* Chunk Metadata Bar */}
                  <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono">
                    <div className="flex items-center gap-2">
                      <span className="rounded bg-blue-500/10 px-2 py-0.5 font-semibold text-blue-400 border border-blue-500/20">
                        Chunk #{chunk.chunk_index}
                      </span>

                      {chunk.page_number !== null && chunk.page_number !== undefined && (
                        <span className="rounded bg-white/5 px-2 py-0.5 text-neutral-300">
                          Page {chunk.page_number}
                        </span>
                      )}

                      {chunk.is_table && (
                        <span className="inline-flex items-center gap-1 rounded bg-purple-500/15 px-2 py-0.5 text-purple-300 border border-purple-500/20">
                          <TableIcon className="h-3 w-3" />
                          Markdown Table
                        </span>
                      )}

                      {chunk.sheet_name && (
                        <span className="rounded bg-teal-500/10 px-2 py-0.5 text-teal-300">
                          Sheet: {chunk.sheet_name}
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-3">
                      <span className="text-neutral-500">{chunk.char_count} chars</span>

                      <button
                        type="button"
                        onClick={() => handleCopy(chunk.id, chunk.content)}
                        className="text-neutral-400 hover:text-white transition-colors"
                        title="Copy chunk text"
                      >
                        {copiedId === chunk.id ? (
                          <Check className="h-3.5 w-3.5 text-green-400" />
                        ) : (
                          <Copy className="h-3.5 w-3.5" />
                        )}
                      </button>
                    </div>
                  </div>

                  {/* Section Title if present */}
                  {chunk.section_title && (
                    <div className="text-xs font-semibold text-neutral-300 border-l-2 border-blue-500 pl-2">
                      {chunk.section_title}
                    </div>
                  )}

                  {/* Chunk Content Preview */}
                  <div className="rounded-lg bg-black/50 p-3 font-mono text-xs text-neutral-300 whitespace-pre-wrap leading-relaxed max-h-60 overflow-y-auto selection:bg-blue-600">
                    {chunk.content}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
