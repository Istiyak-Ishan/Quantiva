"use client";

import React, { useEffect, useState, useCallback, useRef } from 'react';
import Link from 'next/link';
import {
  Database,
  Plus,
  RefreshCw,
  Search,
  CheckCircle2,
  AlertTriangle,
  Download,
  Upload,
  Trash2,
  Layers,
  ArrowRight,
  FileText,
  RotateCw,
  Loader2,
  Check,
  FolderOpen,
} from 'lucide-react';
import {
  getKnowledgeBases,
  deleteKnowledgeBase,
  reindexKnowledgeBase,
  exportKnowledgeBase,
  importKnowledgeBase,
  KnowledgeBaseItem,
} from '@/lib/api';
import { CreateKBModal } from '@/components/knowledge-bases/CreateKBModal';

export default function KnowledgeBasesPage() {
  const [kbs, setKbs] = useState<KnowledgeBaseItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);

  // Reindexing state per KB
  const [reindexingId, setReindexingId] = useState<string | null>(null);
  const [reindexSuccess, setReindexSuccess] = useState<{ id: string; message: string } | null>(null);

  // Import state
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [importing, setImporting] = useState(false);
  const [importMessage, setImportMessage] = useState<string | null>(null);

  // Delete modal state
  const [deletingKb, setDeletingKb] = useState<KnowledgeBaseItem | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const fetchKBs = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getKnowledgeBases();
      setKbs(data);
    } catch (err) {
      console.warn('Failed to fetch knowledge bases:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchKBs();
  }, [fetchKBs]);

  // 1-Click Reindex
  const handleReindex = async (kbId: string) => {
    setReindexingId(kbId);
    setReindexSuccess(null);
    try {
      const res = await reindexKnowledgeBase(kbId);
      setReindexSuccess({ id: kbId, message: `Reindex job submitted (Job ID: ${res.job_id.slice(0, 8)}).` });
      setTimeout(() => setReindexSuccess(null), 4000);
      await fetchKBs();
    } catch (err) {
      alert(`Reindex failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setReindexingId(null);
    }
  };

  // Export .qkb / ZIP package
  const handleExport = async (kb: KnowledgeBaseItem) => {
    try {
      const safeName = kb.name.toLowerCase().replace(/[^a-z0-9_-]/g, '_');
      await exportKnowledgeBase(kb.id, `${safeName}_${kb.id.slice(0, 8)}.qkb`);
    } catch (err) {
      alert(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  // Import .qkb / ZIP package
  const handleImportFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setImporting(true);
    setImportMessage(null);
    try {
      const res = await importKnowledgeBase(file);
      setImportMessage(`Imported "${res.name}" successfully!`);
      setTimeout(() => setImportMessage(null), 4000);
      await fetchKBs();
    } catch (err) {
      alert(`Import failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setImporting(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // Delete KB
  const confirmDelete = async () => {
    if (!deletingKb) return;
    setIsDeleting(true);
    try {
      await deleteKnowledgeBase(deletingKb.id);
      setDeletingKb(null);
      await fetchKBs();
    } catch (err) {
      alert(`Delete failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsDeleting(false);
    }
  };

  const filteredKbs = kbs.filter(
    (k) =>
      k.name.toLowerCase().includes(search.toLowerCase()) ||
      (k.description && k.description.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-8">
      {/* ── Header & Action Controls ─────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="rounded-lg bg-blue-500/10 p-1.5 text-blue-400 border border-blue-500/20">
              <Database className="h-4 w-4" />
            </span>
            <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">Knowledge Bases</h1>
          </div>
          <p className="text-xs text-neutral-400 mt-1">
            Isolated document collections with locked embedding dimensions and dedicated vector indexes.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={fetchKBs}
            className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-neutral-400 hover:text-white transition-colors"
            title="Refresh"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>

          {/* Import .qkb Package */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleImportFile}
            accept=".zip,.qkb"
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={importing}
            className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3.5 py-2 text-xs font-semibold text-neutral-300 hover:bg-white/10 hover:text-white transition-colors disabled:opacity-50"
          >
            {importing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Upload className="h-3.5 w-3.5" />}
            Import .qkb
          </button>

          {/* New Knowledge Base Button */}
          <button
            type="button"
            onClick={() => setShowCreateModal(true)}
            className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-xs font-semibold text-white shadow-md shadow-blue-600/30 hover:bg-blue-500 transition-colors"
          >
            <Plus className="h-4 w-4" />
            New Knowledge Base
          </button>
        </div>
      </div>

      {importMessage && (
        <div className="rounded-xl border border-green-500/20 bg-green-950/20 p-3 text-xs text-green-300 flex items-center gap-2">
          <CheckCircle2 className="h-4 w-4 text-green-400" />
          {importMessage}
        </div>
      )}

      {/* ── Search Bar & Filter ─────────────────────────────────────────── */}
      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-neutral-500" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search knowledge bases by name or description…"
          className="w-full rounded-xl border border-white/10 bg-black/40 pl-9 pr-4 py-2 text-xs text-white placeholder-neutral-500 focus:border-blue-500 focus:outline-none"
        />
      </div>

      {/* ── Knowledge Bases Grid ─────────────────────────────────────────── */}
      {loading ? (
        <div className="py-20 text-center text-xs text-neutral-500 flex flex-col items-center justify-center space-y-2">
          <Loader2 className="h-6 w-6 animate-spin text-blue-400" />
          <span>Loading knowledge bases…</span>
        </div>
      ) : filteredKbs.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-white/10 p-12 text-center space-y-3">
          <Database className="mx-auto h-10 w-10 text-neutral-600" />
          <h3 className="text-sm font-semibold text-neutral-300">No Knowledge Bases Found</h3>
          <p className="text-xs text-neutral-500 max-w-md mx-auto">
            {search
              ? 'No knowledge bases match your search query.'
              : 'Create a knowledge base to lock an embedding space and start indexing documents.'}
          </p>
          {!search && (
            <button
              type="button"
              onClick={() => setShowCreateModal(true)}
              className="inline-flex items-center gap-1.5 rounded-xl bg-blue-600 px-4 py-2 text-xs font-semibold text-white shadow-md shadow-blue-600/30 hover:bg-blue-500 transition-colors"
            >
              <Plus className="h-4 w-4" />
              Create Knowledge Base
            </button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredKbs.map((kb) => {
            const isReindexReq = kb.is_healthy === false;
            return (
              <div
                key={kb.id}
                className="group relative flex flex-col justify-between rounded-2xl border border-white/10 bg-black/30 p-6 hover:border-white/20 hover:bg-white/[0.02] transition-all space-y-5"
              >
                {/* Top Row: Icon + Health Badge + Dimension */}
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-500/10 text-blue-400">
                      <Database className="h-5 w-5" />
                    </div>

                    <div className="flex items-center gap-2">
                      {/* Health Status Badge */}
                      {isReindexReq ? (
                        <span className="inline-flex items-center gap-1 rounded bg-amber-500/10 px-2 py-0.5 text-[10px] font-semibold text-amber-300 border border-amber-500/20">
                          <AlertTriangle className="h-3 w-3 text-amber-400" />
                          Reindex Required
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded bg-teal-500/10 px-2 py-0.5 text-[10px] font-semibold text-teal-300 border border-teal-500/20">
                          <CheckCircle2 className="h-3 w-3 text-teal-400" />
                          Ready
                        </span>
                      )}

                      {/* Dimensions Badge */}
                      <span className="rounded bg-white/5 px-2 py-0.5 text-[10px] font-mono text-neutral-300 border border-white/5">
                        {kb.embedding_dimensions ?? 384}d
                      </span>
                    </div>
                  </div>

                  <h3 className="text-base font-bold text-white group-hover:text-blue-400 transition-colors">
                    {kb.name}
                  </h3>
                  <p className="mt-1 text-xs text-neutral-400 line-clamp-2">
                    {kb.description || 'Dedicated private document intelligence workspace.'}
                  </p>
                </div>

                {/* Middle: Document & Chunk counts */}
                <div className="grid grid-cols-2 gap-3 py-3 border-y border-white/5 text-xs font-mono">
                  <div className="flex flex-col">
                    <span className="text-[10px] uppercase text-neutral-500 font-sans">Documents</span>
                    <span className="text-white font-semibold">{kb.document_count ?? 0}</span>
                  </div>
                  <div className="flex flex-col">
                    <span className="text-[10px] uppercase text-neutral-500 font-sans">Vector Chunks</span>
                    <span className="text-teal-400 font-semibold">{kb.chunk_count ?? 0}</span>
                  </div>
                </div>

                {/* Reindex feedback if active */}
                {reindexSuccess && reindexSuccess.id === kb.id && (
                  <div className="rounded-lg bg-blue-950/30 border border-blue-500/30 p-2 text-[11px] text-blue-300 font-mono">
                    {reindexSuccess.message}
                  </div>
                )}

                {/* Bottom Row Actions */}
                <div className="flex items-center justify-between pt-1">
                  <div className="flex items-center gap-2">
                    {/* 1-Click Reindex Now */}
                    <button
                      type="button"
                      onClick={() => handleReindex(kb.id)}
                      disabled={reindexingId === kb.id}
                      className={`inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-xs font-medium transition-colors ${
                        isReindexReq
                          ? 'bg-amber-500/20 text-amber-300 hover:bg-amber-500/30'
                          : 'border border-white/10 bg-white/5 text-neutral-300 hover:text-white'
                      }`}
                      title="Trigger Blue-Green Vector Reindex"
                    >
                      <RotateCw className={`h-3 w-3 ${reindexingId === kb.id ? 'animate-spin' : ''}`} />
                      {reindexingId === kb.id ? 'Reindexing…' : 'Reindex'}
                    </button>

                    {/* Export .qkb */}
                    <button
                      type="button"
                      onClick={() => handleExport(kb)}
                      className="inline-flex items-center gap-1 rounded-lg border border-white/10 bg-white/5 px-2.5 py-1 text-xs font-medium text-neutral-300 hover:text-white transition-colors"
                      title="Export Knowledge Base as .qkb package"
                    >
                      <Download className="h-3 w-3" />
                      Export
                    </button>
                  </div>

                  {/* Delete Button */}
                  <button
                    type="button"
                    onClick={() => setDeletingKb(kb)}
                    className="p-1.5 text-neutral-500 hover:text-red-400 transition-colors"
                    title="Delete Knowledge Base"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Create Knowledge Base Modal ──────────────────────────────────── */}
      <CreateKBModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onCreated={fetchKBs}
      />

      {/* ── Delete Confirmation Modal ────────────────────────────────────── */}
      {deletingKb && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-2xl border border-red-500/20 bg-[#0e0e13] p-6 shadow-2xl space-y-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-red-500/10 text-red-400">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white">Delete Knowledge Base</h3>
                <p className="text-xs text-neutral-400">This action cannot be undone.</p>
              </div>
            </div>

            <p className="text-xs text-neutral-300 leading-relaxed">
              Are you sure you want to permanently delete <strong className="text-white">"{deletingKb.name}"</strong>?
              This will purge all indexed vector collections, FTS5 tables, document files, and metadata from your local drive.
            </p>

            <div className="flex items-center justify-end gap-3 pt-3 border-t border-white/5">
              <button
                type="button"
                onClick={() => setDeletingKb(null)}
                disabled={isDeleting}
                className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold text-neutral-300 hover:bg-white/10"
              >
                Cancel
              </button>

              <button
                type="button"
                onClick={confirmDelete}
                disabled={isDeleting}
                className="inline-flex items-center gap-2 rounded-xl bg-red-600 px-4 py-2 text-xs font-semibold text-white shadow-md shadow-red-600/30 hover:bg-red-500 disabled:opacity-50"
              >
                {isDeleting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                Permanently Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
