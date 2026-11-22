"use client";

import React, { useEffect, useState, useCallback } from 'react';
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
  Layers,
  UploadCloud,
  Sliders,
  Clock,
  Loader2,
  AlertTriangle,
  Play,
  RotateCw,
} from 'lucide-react';
import {
  getKnowledgeBases,
  getDocuments,
  getStorageStatus,
  getSystemHardware,
  getLLMConfigs,
  getEmbeddingConfigs,
  getSessions,
  getProcessingJobs,
  KnowledgeBaseItem,
  DocumentItem,
  StorageStatus,
  SystemHardware,
  LLMConfig,
  EmbeddingConfig,
  ProcessingJob,
} from '@/lib/api';
import { invokeTauriCommand, isTauriDesktop } from '@/lib/api-client';
import { CreateKBModal } from '@/components/knowledge-bases/CreateKBModal';

interface SessionItem {
  id: string;
  title: string | null;
  created_at: string;
}

export default function DashboardPage() {
  const [kbs, setKbs] = useState<KnowledgeBaseItem[]>([]);
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [storage, setStorage] = useState<StorageStatus | null>(null);
  const [hw, setHw] = useState<SystemHardware | null>(null);
  const [llmConfigs, setLlmConfigs] = useState<LLMConfig[]>([]);
  const [embeddingConfigs, setEmbeddingConfigs] = useState<EmbeddingConfig[]>([]);
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [loading, setLoading] = useState(true);

  // Modal State
  const [showCreateKbModal, setShowCreateKbModal] = useState(false);

  const loadDashboardData = useCallback(async () => {
    try {
      const [
        kbsRes,
        docsRes,
        storageRes,
        hwRes,
        llmRes,
        embRes,
        sessionsRes,
        jobsRes,
      ] = await Promise.allSettled([
        getKnowledgeBases(),
        getDocuments(),
        getStorageStatus(),
        getSystemHardware(),
        getLLMConfigs(),
        getEmbeddingConfigs(),
        getSessions(),
        getProcessingJobs({ limit: 10 }),
      ]);

      if (kbsRes.status === 'fulfilled') setKbs(kbsRes.value);
      if (docsRes.status === 'fulfilled') setDocs(docsRes.value);
      if (storageRes.status === 'fulfilled') setStorage(storageRes.value);
      if (hwRes.status === 'fulfilled') setHw(hwRes.value);
      if (llmRes.status === 'fulfilled') setLlmConfigs(llmRes.value);
      if (embRes.status === 'fulfilled') setEmbeddingConfigs(embRes.value);
      if (sessionsRes.status === 'fulfilled') setSessions(sessionsRes.value);
      if (jobsRes.status === 'fulfilled') setJobs(jobsRes.value);
    } catch (err) {
      console.warn('Dashboard data fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDashboardData();

    // Auto-refresh processing jobs
    const interval = setInterval(() => {
      void loadDashboardData();
    }, 4000);
    return () => clearInterval(interval);
  }, [loadDashboardData]);

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

  // Active models
  const defaultLLM = llmConfigs.find((c) => c.is_default) || llmConfigs[0];
  const defaultEmb = embeddingConfigs.find((c) => c.is_default) || embeddingConfigs[0];

  // Total vector chunks calculated across KBs or documents
  const totalChunks = kbs.reduce((acc, k) => acc + (k.chunk_count || 0), 0) ||
    docs.reduce((acc, d) => acc + (d.chunk_count || 0), 0);

  const activeJobs = jobs.filter((j) => j.status === 'RUNNING' || j.status === 'PENDING');

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-8">
      {/* ── Welcome Banner & Quick Actions ───────────────────────────────── */}
      <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-r from-blue-950/40 via-[#0e0e14] to-teal-950/30 p-6 md:p-8 shadow-xl">
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="max-w-xl space-y-2">
            <div className="inline-flex items-center gap-1.5 rounded-full bg-blue-500/10 px-3 py-1 text-xs font-semibold text-blue-400 border border-blue-500/20">
              <Sparkles className="h-3.5 w-3.5" />
              Desktop AI Document Intelligence
            </div>
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-white">
              Quantiva Control Center
            </h1>
            <p className="text-xs md:text-sm text-neutral-400 leading-relaxed">
              100% private local-first RAG. Ingest multi-format documents, inspect extracted chunks, and query with verified citations.
            </p>
          </div>

          {/* Quick Actions Buttons */}
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => setShowCreateKbModal(true)}
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2.5 text-xs font-semibold text-white shadow-lg shadow-blue-600/30 hover:bg-blue-500 transition-colors"
            >
              <Plus className="h-4 w-4" />
              New Knowledge Base
            </button>

            <Link
              href="/documents/"
              className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-xs font-semibold text-neutral-200 hover:bg-white/10 transition-colors"
            >
              <UploadCloud className="h-4 w-4 text-teal-400" />
              Upload Documents
            </Link>

            <Link
              href="/chat/"
              className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-xs font-semibold text-neutral-200 hover:bg-white/10 transition-colors"
            >
              <MessageSquare className="h-4 w-4 text-blue-400" />
              Open AI Chat
            </Link>

            <Link
              href="/models/"
              className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-xs font-semibold text-neutral-200 hover:bg-white/10 transition-colors"
            >
              <Sliders className="h-4 w-4 text-purple-400" />
              Configure Models
            </Link>
          </div>
        </div>
      </div>

      {/* ── Summary Cards Grid ───────────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        {/* Card 1: Total Knowledge Bases */}
        <Link
          href="/knowledge-bases/"
          className="rounded-xl border border-white/10 bg-black/40 p-4 flex flex-col justify-between hover:border-white/20 hover:bg-white/[0.02] transition-all"
        >
          <div className="flex items-center justify-between text-neutral-400">
            <span className="text-xs font-medium">Knowledge Bases</span>
            <Database className="h-4 w-4 text-blue-400" />
          </div>
          <div className="mt-3">
            <span className="text-2xl font-bold text-white">{kbs.length}</span>
            <p className="text-[10px] text-neutral-500 mt-0.5">Isolated vector spaces</p>
          </div>
        </Link>

        {/* Card 2: Indexed Documents */}
        <Link
          href="/documents/"
          className="rounded-xl border border-white/10 bg-black/40 p-4 flex flex-col justify-between hover:border-white/20 hover:bg-white/[0.02] transition-all"
        >
          <div className="flex items-center justify-between text-neutral-400">
            <span className="text-xs font-medium">Documents</span>
            <Files className="h-4 w-4 text-teal-400" />
          </div>
          <div className="mt-3">
            <span className="text-2xl font-bold text-white">{docs.length}</span>
            <p className="text-[10px] text-neutral-500 mt-0.5">
              {docs.filter((d) => d.status === 'ready').length} ready to search
            </p>
          </div>
        </Link>

        {/* Card 3: Total Vector Chunks */}
        <div className="rounded-xl border border-white/10 bg-black/40 p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-neutral-400">
            <span className="text-xs font-medium">Vector Chunks</span>
            <Layers className="h-4 w-4 text-purple-400" />
          </div>
          <div className="mt-3">
            <span className="text-2xl font-bold text-white">{totalChunks}</span>
            <p className="text-[10px] text-neutral-500 mt-0.5">FTS5 + dense indexes</p>
          </div>
        </div>

        {/* Card 4: Active LLM Model */}
        <Link
          href="/models/"
          className="rounded-xl border border-white/10 bg-black/40 p-4 flex flex-col justify-between hover:border-white/20 hover:bg-white/[0.02] transition-all"
        >
          <div className="flex items-center justify-between text-neutral-400">
            <span className="text-xs font-medium">Active LLM</span>
            <Sparkles className="h-4 w-4 text-amber-400" />
          </div>
          <div className="mt-3 min-w-0">
            <span className="text-sm font-bold text-white truncate block">
              {defaultLLM?.model_name || defaultLLM?.name || 'Local Ollama'}
            </span>
            <p className="text-[10px] text-neutral-500 mt-0.5 uppercase font-mono">
              {defaultLLM?.provider_type || 'Local Engine'}
            </p>
          </div>
        </Link>

        {/* Card 5: Active Embedding Model */}
        <Link
          href="/models/"
          className="rounded-xl border border-white/10 bg-black/40 p-4 flex flex-col justify-between hover:border-white/20 hover:bg-white/[0.02] transition-all"
        >
          <div className="flex items-center justify-between text-neutral-400">
            <span className="text-xs font-medium">Embedding Model</span>
            <Cpu className="h-4 w-4 text-teal-400" />
          </div>
          <div className="mt-3 min-w-0">
            <span className="text-sm font-bold text-teal-300 truncate block">
              {defaultEmb?.name || defaultEmb?.model_name || 'BGE Small'}
            </span>
            <p className="text-[10px] text-neutral-500 mt-0.5 font-mono">
              {defaultEmb?.dimensions || 384} dimensions
            </p>
          </div>
        </Link>

        {/* Card 6: Processing Jobs */}
        <div className="rounded-xl border border-white/10 bg-black/40 p-4 flex flex-col justify-between">
          <div className="flex items-center justify-between text-neutral-400">
            <span className="text-xs font-medium">Job Center</span>
            <Clock className="h-4 w-4 text-blue-400" />
          </div>
          <div className="mt-3">
            <span className="text-2xl font-bold text-white">
              {activeJobs.length > 0 ? (
                <span className="text-blue-400 flex items-center gap-1.5">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  {activeJobs.length} active
                </span>
              ) : (
                <span className="text-neutral-400 text-lg">Engine Idle</span>
              )}
            </span>
            <p className="text-[10px] text-neutral-500 mt-0.5">
              {jobs.length} recent executions
            </p>
          </div>
        </div>
      </div>

      {/* ── Two Columns: Real-Time Job Center & Recent Conversations ──────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Real-Time Processing Jobs Column */}
        <div className="rounded-2xl border border-white/10 bg-black/30 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-blue-400" />
              <h2 className="text-sm font-bold text-white">Ingestion & Processing Jobs</h2>
            </div>
            <Link
              href="/documents/"
              className="text-xs text-blue-400 hover:text-blue-300 transition-colors flex items-center gap-1"
            >
              Document Manager <ArrowRight className="h-3 w-3" />
            </Link>
          </div>

          {jobs.length === 0 ? (
            <div className="rounded-xl border border-dashed border-white/10 p-8 text-center text-xs text-neutral-500">
              No recent background processing jobs.
            </div>
          ) : (
            <div className="space-y-3">
              {jobs.slice(0, 5).map((job) => (
                <div
                  key={job.id}
                  className="rounded-xl border border-white/5 bg-white/[0.02] p-3.5 space-y-2 text-xs"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-white uppercase font-mono text-[11px]">
                        {job.job_type.replace('_', ' ')}
                      </span>
                      {job.status === 'RUNNING' ? (
                        <span className="inline-flex items-center gap-1 rounded bg-blue-500/10 px-2 py-0.5 text-[10px] font-semibold text-blue-400 border border-blue-500/20">
                          <Loader2 className="h-3 w-3 animate-spin" />
                          {job.stage || 'In Progress'}
                        </span>
                      ) : job.status === 'COMPLETED' ? (
                        <span className="inline-flex items-center gap-1 rounded bg-teal-500/10 px-2 py-0.5 text-[10px] font-semibold text-teal-400 border border-teal-500/20">
                          <CheckCircle2 className="h-3 w-3" />
                          Completed
                        </span>
                      ) : job.status === 'FAILED' ? (
                        <span className="inline-flex items-center gap-1 rounded bg-red-500/10 px-2 py-0.5 text-[10px] font-semibold text-red-400 border border-red-500/20">
                          <AlertTriangle className="h-3 w-3" />
                          Failed
                        </span>
                      ) : (
                        <span className="rounded bg-white/5 px-2 py-0.5 text-[10px] text-neutral-400 font-mono">
                          {job.status}
                        </span>
                      )}
                    </div>

                    <span className="text-[10px] text-neutral-500 font-mono">
                      {job.created_at ? new Date(job.created_at).toLocaleTimeString() : ''}
                    </span>
                  </div>

                  {job.status === 'RUNNING' && (
                    <div className="space-y-1 pt-1">
                      <div className="flex justify-between text-[10px] font-mono text-neutral-400">
                        <span>{job.message || 'Processing chunks…'}</span>
                        <span>{job.progress_percent ? `${job.progress_percent.toFixed(0)}%` : ''}</span>
                      </div>
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5 p-0.5">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-blue-500 to-teal-400"
                          style={{ width: `${Math.min(100, Math.max(10, job.progress_percent || 20))}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {job.error_details && (
                    <p className="text-[11px] text-red-400 font-mono">{job.error_details}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Recent Conversations Column */}
        <div className="rounded-2xl border border-white/10 bg-black/30 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4 text-teal-400" />
              <h2 className="text-sm font-bold text-white">Recent Conversations</h2>
            </div>
            <Link
              href="/chat/"
              className="text-xs text-teal-400 hover:text-teal-300 transition-colors flex items-center gap-1"
            >
              Start New Chat <ArrowRight className="h-3 w-3" />
            </Link>
          </div>

          {sessions.length === 0 ? (
            <div className="rounded-xl border border-dashed border-white/10 p-8 text-center text-xs text-neutral-500 space-y-3">
              <MessageSquare className="mx-auto h-8 w-8 text-neutral-600" />
              <p>No recent conversations.</p>
              <Link
                href="/chat/"
                className="inline-flex items-center gap-1.5 rounded-lg bg-teal-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-teal-500 transition-colors"
              >
                <Plus className="h-3.5 w-3.5" /> Start Conversation
              </Link>
            </div>
          ) : (
            <div className="divide-y divide-white/5">
              {sessions.slice(0, 5).map((s) => (
                <Link
                  key={s.id}
                  href={`/chat/?sessionId=${s.id}`}
                  className="py-3 flex items-center justify-between hover:bg-white/[0.02] px-2 rounded-lg transition-colors group"
                >
                  <div className="min-w-0 space-y-0.5">
                    <span className="text-xs font-semibold text-white group-hover:text-teal-300 transition-colors truncate block">
                      {s.title || 'Untitled Session'}
                    </span>
                    <span className="text-[10px] text-neutral-500 font-mono">
                      {s.created_at ? new Date(s.created_at).toLocaleDateString() : 'Recent'}
                    </span>
                  </div>

                  <ArrowRight className="h-3.5 w-3.5 text-neutral-500 group-hover:text-white transition-colors shrink-0 ml-2" />
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Create Knowledge Base Modal ──────────────────────────────────── */}
      <CreateKBModal
        isOpen={showCreateKbModal}
        onClose={() => setShowCreateKbModal(false)}
        onCreated={loadDashboardData}
      />
    </div>
  );
}
