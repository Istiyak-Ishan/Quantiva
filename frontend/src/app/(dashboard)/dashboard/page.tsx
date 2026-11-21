"use client";

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import {
  MessageSquare,
  Database,
  Files,
  Cpu,
  Sparkles,
  HardDrive,
  CheckCircle2,
  ArrowRight,
  Plus,
  RefreshCw,
  FolderOpen,
} from 'lucide-react';
import {
  getKnowledgeBases,
  getDocuments,
  getStorageStatus,
  getSystemHardware,
  KnowledgeBaseItem,
  DocumentItem,
  StorageStatus,
  SystemHardware,
} from '@/lib/api';
import { invokeTauriCommand, isTauriDesktop } from '@/lib/api-client';

export default function DashboardPage() {
  const [kbs, setKbs] = useState<KnowledgeBaseItem[]>([]);
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [storage, setStorage] = useState<StorageStatus | null>(null);
  const [hw, setHw] = useState<SystemHardware | null>(null);
  const [loading, setLoading] = useState(true);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const [kbsRes, docsRes, storageRes, hwRes] = await Promise.allSettled([
        getKnowledgeBases(),
        getDocuments(),
        getStorageStatus(),
        getSystemHardware(),
      ]);

      if (kbsRes.status === 'fulfilled') setKbs(kbsRes.value);
      if (docsRes.status === 'fulfilled') setDocs(docsRes.value);
      if (storageRes.status === 'fulfilled') setStorage(storageRes.value);
      if (hwRes.status === 'fulfilled') setHw(hwRes.value);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadDashboardData();
  }, []);

  const handleOpenStorage = async () => {
    try {
      if (isTauriDesktop()) {
        await invokeTauriCommand('open_data_folder');
      } else if (storage?.current_data_dir) {
        window.alert(`Quantiva Data Directory:\n${storage.current_data_dir}`);
      }
    } catch (err) {
      console.error('Failed to open data folder:', err);
    }
  };

  const formatBytes = (bytes?: number) => {
    if (!bytes || bytes <= 0) return '0 MB';
    const mb = bytes / (1024 * 1024);
    if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
    return `${mb.toFixed(0)} MB`;
  };

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-8">
      {/* ── Welcome Banner ───────────────────────────────────────────────── */}
      <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-r from-blue-950/40 via-[#0e0e14] to-teal-950/30 p-6 md:p-8 shadow-xl">
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="max-w-xl space-y-2">
            <div className="inline-flex items-center gap-1.5 rounded-full bg-blue-500/10 px-3 py-1 text-xs font-semibold text-blue-400 border border-blue-500/20">
              <Sparkles className="h-3.5 w-3.5" />
              Desktop AI Document Intelligence
            </div>
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white">
              Private Document Intelligence
            </h1>
            <p className="text-xs md:text-sm text-neutral-400 leading-relaxed">
              100% local-first RAG with hybrid dense/sparse search, Reciprocal Rank Fusion, ReAct agent
              orchestration, and citations. All files stay securely on your Windows PC.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Link
              href="/chat/"
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-blue-600/30 hover:bg-blue-500 transition-colors"
            >
              <MessageSquare className="h-4 w-4" />
              New Conversation
            </Link>

            <Link
              href="/knowledge-bases/"
              className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-xs font-semibold text-neutral-200 hover:bg-white/10 transition-colors"
            >
              <Plus className="h-4 w-4" />
              New Knowledge Base
            </Link>
          </div>
        </div>
      </div>

      {/* ── Quick Stats Grid ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Stat 1: Knowledge Bases */}
        <div className="rounded-xl border border-white/10 bg-black/40 p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-neutral-400">
            <span className="text-xs font-medium">Knowledge Bases</span>
            <Database className="h-4 w-4 text-blue-400" />
          </div>
          <div className="mt-4">
            <span className="text-2xl font-bold text-white">{kbs.length}</span>
            <p className="mt-1 text-[11px] text-neutral-500">Dedicated vector collections</p>
          </div>
        </div>

        {/* Stat 2: Total Documents */}
        <div className="rounded-xl border border-white/10 bg-black/40 p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-neutral-400">
            <span className="text-xs font-medium">Indexed Documents</span>
            <Files className="h-4 w-4 text-teal-400" />
          </div>
          <div className="mt-4">
            <span className="text-2xl font-bold text-white">{docs.length}</span>
            <p className="mt-1 text-[11px] text-neutral-500">
              {docs.filter((d) => d.status === 'ready').length} ready for retrieval
            </p>
          </div>
        </div>

        {/* Stat 3: Local Storage */}
        <div className="rounded-xl border border-white/10 bg-black/40 p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-neutral-400">
            <span className="text-xs font-medium">Free Disk Space</span>
            <HardDrive className="h-4 w-4 text-purple-400" />
          </div>
          <div className="mt-4">
            <span className="text-2xl font-bold text-white">
              {storage ? formatBytes(storage.disk_free_bytes) : 'Calculating…'}
            </span>
            <p className="mt-1 text-[11px] text-neutral-500">
              {storage ? `DB: ${formatBytes(storage.database_size_bytes)}` : 'SQLite WAL active'}
            </p>
          </div>
        </div>

        {/* Stat 4: Hardware Recommendation */}
        <div className="rounded-xl border border-white/10 bg-black/40 p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between text-neutral-400">
            <span className="text-xs font-medium">Model Calibration</span>
            <Cpu className="h-4 w-4 text-amber-400" />
          </div>
          <div className="mt-4">
            <span className="text-2xl font-bold text-white truncate block">
              {hw?.recommendation?.tier === 'tier_1_3b'
                ? '3B Tier'
                : hw?.recommendation?.tier === 'tier_3_14b'
                ? '14B Tier'
                : '7B Tier'}
            </span>
            <p className="mt-1 text-[11px] text-neutral-500">
              {hw?.hardware ? `${hw.hardware.ram_total_gb.toFixed(0)} GB RAM / ${hw.hardware.physical_cores} Cores` : 'Calibrated'}
            </p>
          </div>
        </div>
      </div>

      {/* ── Knowledge Bases & Quick Links Section ────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Knowledge Bases Column */}
        <div className="lg:col-span-2 rounded-xl border border-white/10 bg-black/30 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Database className="h-4 w-4 text-blue-400" />
              <h2 className="text-sm font-semibold text-white">Active Knowledge Bases</h2>
            </div>
            <Link
              href="/knowledge-bases/"
              className="inline-flex items-center gap-1 text-xs font-medium text-blue-400 hover:text-blue-300 transition-colors"
            >
              View all <ArrowRight className="h-3 w-3" />
            </Link>
          </div>

          {loading ? (
            <div className="py-8 text-center text-xs text-neutral-500">Loading workspaces…</div>
          ) : kbs.length === 0 ? (
            <div className="rounded-lg border border-dashed border-white/10 p-8 text-center space-y-3">
              <Database className="mx-auto h-8 w-8 text-neutral-600" />
              <div className="space-y-1">
                <p className="text-xs font-medium text-neutral-300">No Knowledge Bases created yet</p>
                <p className="text-[11px] text-neutral-500">
                  Create a knowledge base to lock in an embedding space and start indexing documents.
                </p>
              </div>
              <Link
                href="/knowledge-bases/"
                className="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-blue-500 transition-colors"
              >
                <Plus className="h-3.5 w-3.5" />
                Create Knowledge Base
              </Link>
            </div>
          ) : (
            <div className="divide-y divide-white/5">
              {kbs.slice(0, 5).map((kb) => (
                <div key={kb.id} className="py-3 flex items-center justify-between">
                  <div className="flex flex-col space-y-0.5 min-w-0">
                    <span className="text-xs font-semibold text-white truncate">{kb.name}</span>
                    <span className="text-[11px] text-neutral-500 truncate">
                      {kb.description || 'Dedicated private document collection'}
                    </span>
                  </div>

                  <div className="flex items-center gap-3 shrink-0 ml-4">
                    <span className="rounded bg-white/5 px-2 py-0.5 text-[10px] font-mono text-neutral-400">
                      {kb.chunk_count ?? 0} chunks
                    </span>
                    <span className="rounded bg-teal-500/10 px-2 py-0.5 text-[10px] font-mono text-teal-400 border border-teal-500/20">
                      {kb.embedding_dimensions ?? 384}-d
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* System & Hardware Column */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-teal-400" />
              <h2 className="text-sm font-semibold text-white">Local System Status</h2>
            </div>
            <button
              type="button"
              onClick={loadDashboardData}
              className="text-neutral-400 hover:text-white transition-colors"
              title="Refresh"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>

          <div className="space-y-3 text-xs">
            <div className="rounded-lg border border-white/5 bg-white/[0.02] p-3 space-y-1.5">
              <div className="flex items-center justify-between text-neutral-400">
                <span>SQLite Database</span>
                <span className="text-green-400 flex items-center gap-1 font-medium">
                  <CheckCircle2 className="h-3 w-3" /> WAL Mode
                </span>
              </div>
              <div className="flex items-center justify-between text-neutral-400">
                <span>Vector Engine</span>
                <span className="text-teal-400 flex items-center gap-1 font-medium">
                  <CheckCircle2 className="h-3 w-3" /> Embedded
                </span>
              </div>
              <div className="flex items-center justify-between text-neutral-400">
                <span>Local Isolation</span>
                <span className="text-blue-400 font-mono text-[11px]">127.0.0.1</span>
              </div>
            </div>

            {storage && (
              <div className="rounded-lg border border-white/5 bg-white/[0.02] p-3 space-y-2">
                <span className="text-[10px] uppercase font-semibold text-neutral-500">Data Directory</span>
                <p className="text-[11px] font-mono text-neutral-300 break-all">{storage.current_data_dir}</p>
                <button
                  type="button"
                  onClick={handleOpenStorage}
                  className="inline-flex items-center gap-1 text-[11px] text-blue-400 hover:text-blue-300 transition-colors"
                >
                  <FolderOpen className="h-3 w-3" />
                  Open in Windows Explorer
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
