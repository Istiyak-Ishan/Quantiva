"use client";

import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
  Files,
  FileText,
  UploadCloud,
  RefreshCw,
  Trash2,
  Download,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Search,
  Layers,
  Table as TableIcon,
  RotateCcw,
  XCircle,
  Eye,
  Sliders,
  Database,
  ScanText,
} from 'lucide-react';
import {
  getDocuments,
  deleteDocument,
  reindexDocument,
  openDocument,
  getProcessingJobs,
  cancelJob,
  retryJob,
  getKnowledgeBases,
  DocumentItem,
  ProcessingJob,
  KnowledgeBaseItem,
  api,
} from '@/lib/api';
import { useDropzone } from 'react-dropzone';
import { ChunkInspectorDrawer } from '@/components/documents/ChunkInspectorDrawer';

// Stage mapping to human readable names & sequence numbers
const PIPELINE_STAGES: Record<string, { label: string; step: number }> = {
  VALIDATING: { label: 'Validating File', step: 1 },
  EXTRACTING_TEXT: { label: 'Extracting Text', step: 2 },
  RUNNING_OCR: { label: 'Running OCR', step: 3 },
  CHUNKING: { label: 'Chunking Content', step: 4 },
  GENERATING_EMBEDDINGS: { label: 'Generating Embeddings', step: 5 },
  INDEXING_VECTORS: { label: 'Indexing Vectors & FTS5', step: 6 },
  COMPLETED: { label: 'Available', step: 7 },
  FAILED: { label: 'Failed', step: 0 },
};

export default function DocumentManagerPage() {
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [kbs, setKbs] = useState<KnowledgeBaseItem[]>([]);
  const [selectedKbId, setSelectedKbId] = useState<string>('all');
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');

  // Drag & drop upload state
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);

  // Slide-over Chunk Inspector state
  const [inspectingDoc, setInspectingDoc] = useState<DocumentItem | null>(null);

  // Polling data fetcher
  const fetchData = useCallback(async () => {
    try {
      const [docsRes, jobsRes, kbsRes] = await Promise.allSettled([
        getDocuments(),
        getProcessingJobs(),
        getKnowledgeBases(),
      ]);

      if (docsRes.status === 'fulfilled') setDocs(docsRes.value);
      if (jobsRes.status === 'fulfilled') setJobs(jobsRes.value);
      if (kbsRes.status === 'fulfilled') setKbs(kbsRes.value);
    } catch (err) {
      console.warn('Failed to refresh documents/jobs:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchData();

    // Poll active jobs every 2.5 seconds
    const interval = setInterval(() => {
      void fetchData();
    }, 2500);

    return () => clearInterval(interval);
  }, [fetchData]);

  // Dropzone file upload handler
  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;
      setUploading(true);
      setUploadError(null);
      setUploadSuccess(null);

      try {
        // Upload files
        for (const file of acceptedFiles) {
          const formData = new FormData();
          formData.append('file', file);
          if (selectedKbId && selectedKbId !== 'all') {
            formData.append('knowledge_base_id', selectedKbId);
          }
          await api.post('/documents', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
          });
        }
        setUploadSuccess(`Uploaded ${acceptedFiles.length} document(s). Processing queued.`);
        setTimeout(() => setUploadSuccess(null), 4000);
        await fetchData();
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : 'Upload failed');
      } finally {
        setUploading(false);
      }
    },
    [selectedKbId, fetchData]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'text/csv': ['.csv'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
    },
  });

  const handleDelete = async (id: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    if (!window.confirm('Are you sure you want to delete this document and all its indexed chunks?')) return;
    try {
      await deleteDocument(id);
      setDocs((prev) => prev.filter((d) => d.id !== id));
      if (inspectingDoc?.id === id) setInspectingDoc(null);
    } catch (err) {
      alert('Failed to delete document');
    }
  };

  const handleCancelJob = async (jobId: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    try {
      await cancelJob(jobId);
      await fetchData();
    } catch (err) {
      alert('Failed to cancel job');
    }
  };

  const handleRetryJob = async (jobId: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    try {
      await retryJob(jobId);
      await fetchData();
    } catch (err) {
      alert('Failed to retry job');
    }
  };

  const handleReindex = async (docId: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    try {
      await reindexDocument(docId);
      await fetchData();
    } catch (err) {
      alert('Failed to reindex document');
    }
  };

  const formatSize = (bytes: number) => {
    if (!bytes || bytes <= 0) return '0 B';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // Associate most recent job with document
  const getDocJob = (docId: string): ProcessingJob | undefined => {
    return jobs.find((j) => j.document_id === docId);
  };

  const filteredDocs = docs.filter((d) => {
    const matchesSearch = d.filename.toLowerCase().includes(search.toLowerCase());
    return matchesSearch;
  });

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-8">
      {/* ── Page Header ─────────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="rounded-lg bg-teal-500/10 p-1.5 text-teal-400 border border-teal-500/20">
              <Files className="h-4 w-4" />
            </span>
            <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">Document Manager</h1>
          </div>
          <p className="text-xs text-neutral-400 mt-1">
            Multi-format local ingestion, OCR pipeline monitoring, and chunk inspection.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={fetchData}
            className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-neutral-400 hover:text-white transition-colors"
            title="Refresh"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {/* ── Drag & Drop Multi-Format File Uploader ────────────────────────── */}
      <div
        {...getRootProps()}
        className={`cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition-all ${
          isDragActive
            ? 'border-blue-500 bg-blue-500/10 shadow-lg shadow-blue-500/10'
            : 'border-white/10 bg-black/40 hover:border-white/20 hover:bg-white/[0.02]'
        }`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center space-y-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-500/10 text-blue-400">
            {uploading ? (
              <Loader2 className="h-6 w-6 animate-spin text-blue-400" />
            ) : (
              <UploadCloud className="h-6 w-6" />
            )}
          </div>
          <div>
            <p className="text-sm font-semibold text-white">
              {uploading ? 'Processing documents locally…' : 'Drop documents here, or click to browse'}
            </p>
            <p className="text-xs text-neutral-500 mt-1">
              Supports <strong className="text-neutral-400">PDF, DOCX, XLSX, CSV, TXT, MD</strong> (Processed 100% locally with OCR & Table preservation)
            </p>
          </div>
        </div>
      </div>

      {uploadSuccess && (
        <div className="rounded-xl border border-green-500/20 bg-green-950/20 p-3 text-xs text-green-300 flex items-center gap-2">
          <CheckCircle2 className="h-4 w-4 text-green-400 shrink-0" />
          <span>{uploadSuccess}</span>
        </div>
      )}

      {uploadError && (
        <div className="rounded-xl border border-red-500/20 bg-red-950/20 p-3 text-xs text-red-300 flex items-center gap-2">
          <AlertCircle className="h-4 w-4 text-red-400 shrink-0" />
          <span>{uploadError}</span>
        </div>
      )}

      {/* ── Document Table with Real-Time Pipeline Progress ──────────────── */}
      <div className="space-y-4">
        {/* Search & Filter Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-neutral-500" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Filter documents by filename…"
              className="w-full rounded-xl border border-white/10 bg-black/40 pl-9 pr-4 py-2 text-xs text-white placeholder-neutral-500 focus:border-blue-500 focus:outline-none"
            />
          </div>

          <div className="text-xs text-neutral-500 font-mono">
            {filteredDocs.length} document(s) registered
          </div>
        </div>

        {/* Table */}
        {loading && docs.length === 0 ? (
          <div className="py-20 text-center text-xs text-neutral-500 flex flex-col items-center justify-center space-y-2">
            <Loader2 className="h-6 w-6 animate-spin text-blue-400" />
            <span>Loading documents…</span>
          </div>
        ) : filteredDocs.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-white/10 p-12 text-center text-xs text-neutral-500">
            {search ? 'No documents matched your search filter.' : 'No documents indexed yet. Drop files above to begin.'}
          </div>
        ) : (
          <div className="overflow-hidden rounded-xl border border-white/10 bg-black/30">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-white/5 bg-white/[0.02] text-neutral-400">
                <tr>
                  <th className="px-4 py-3 font-medium">Document</th>
                  <th className="px-4 py-3 font-medium">Size</th>
                  <th className="px-4 py-3 font-medium">Pipeline Stage / Status</th>
                  <th className="px-4 py-3 font-medium">Chunks</th>
                  <th className="px-4 py-3 font-medium text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-neutral-300">
                {filteredDocs.map((doc) => {
                  const job = getDocJob(doc.id);
                  const isProcessing = doc.status === 'processing' || doc.status === 'uploading' || job?.status === 'RUNNING';
                  const stageInfo = job?.stage ? PIPELINE_STAGES[job.stage] || { label: job.stage, step: 2 } : null;

                  return (
                    <tr
                      key={doc.id}
                      onClick={() => setInspectingDoc(doc)}
                      className="hover:bg-white/[0.03] transition-colors cursor-pointer group"
                    >
                      {/* Document Name & Type */}
                      <td className="px-4 py-3.5 flex items-center gap-2.5 font-medium text-white truncate max-w-xs">
                        <FileText className="h-4 w-4 shrink-0 text-blue-400" />
                        <div className="min-w-0">
                          <span className="truncate block font-semibold group-hover:text-blue-400 transition-colors">
                            {doc.filename}
                          </span>
                          <span className="text-[10px] text-neutral-500 font-mono">
                            {doc.file_type ? doc.file_type.toUpperCase() : 'DOCUMENT'}
                          </span>
                        </div>
                      </td>

                      {/* File Size */}
                      <td className="px-4 py-3.5 font-mono text-neutral-400">
                        {formatSize(doc.file_size)}
                      </td>

                      {/* Real-Time Granular Pipeline Stage */}
                      <td className="px-4 py-3.5">
                        {isProcessing ? (
                          <div className="space-y-1.5 max-w-xs">
                            <div className="flex items-center justify-between text-[11px] font-mono">
                              <span className="flex items-center gap-1.5 text-blue-300 font-semibold">
                                <Loader2 className="h-3 w-3 animate-spin text-blue-400" />
                                {stageInfo?.label || 'Processing…'}
                              </span>
                              <span className="text-neutral-400">
                                {job?.progress_percent !== undefined && job?.progress_percent !== null
                                  ? `${job.progress_percent.toFixed(0)}%`
                                  : 'Active'}
                              </span>
                            </div>

                            {/* Granular Progress Bar */}
                            <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/5 p-0.5">
                              <div
                                className="h-full rounded-full bg-gradient-to-r from-blue-500 to-teal-400 transition-all duration-300"
                                style={{
                                  width: `${Math.min(100, Math.max(10, job?.progress_percent || 25))}%`,
                                }}
                              />
                            </div>
                          </div>
                        ) : doc.status === 'ready' ? (
                          <div className="flex items-center gap-2">
                            <span className="inline-flex items-center gap-1 rounded bg-teal-500/10 px-2 py-0.5 text-[10px] font-semibold text-teal-400 border border-teal-500/20">
                              <CheckCircle2 className="h-3 w-3" /> Ready
                            </span>
                          </div>
                        ) : (
                          <div className="flex items-center gap-2">
                            <span className="inline-flex items-center gap-1 rounded bg-red-500/10 px-2 py-0.5 text-[10px] font-semibold text-red-400 border border-red-500/20">
                              <AlertCircle className="h-3 w-3" /> Failed
                            </span>
                            {doc.failure_reason && (
                              <span className="text-[10px] text-red-400/80 truncate max-w-xs">
                                {doc.failure_reason}
                              </span>
                            )}
                          </div>
                        )}
                      </td>

                      {/* Chunk Count */}
                      <td className="px-4 py-3.5 font-mono text-neutral-300">
                        {doc.chunk_count !== null && doc.chunk_count !== undefined ? (
                          <span className="rounded bg-white/5 px-2 py-0.5 text-[11px]">
                            {doc.chunk_count} chunks
                          </span>
                        ) : (
                          <span className="text-neutral-500">—</span>
                        )}
                      </td>

                      {/* Action Buttons */}
                      <td className="px-4 py-3.5 text-right" onClick={(e) => e.stopPropagation()}>
                        <div className="flex items-center justify-end gap-1.5">
                          {/* Inspect Chunks Button */}
                          <button
                            type="button"
                            onClick={() => setInspectingDoc(doc)}
                            className="inline-flex items-center gap-1 rounded-lg border border-white/5 bg-white/5 px-2.5 py-1 text-[11px] font-medium text-neutral-300 hover:text-white hover:bg-white/10 transition-colors"
                            title="Inspect Chunks & Tables"
                          >
                            <Eye className="h-3 w-3 text-blue-400" />
                            Inspect
                          </button>

                          {/* Cancel Job if running */}
                          {job && job.status === 'RUNNING' && (
                            <button
                              type="button"
                              onClick={(e) => handleCancelJob(job.id, e)}
                              className="p-1.5 text-neutral-400 hover:text-amber-400 transition-colors"
                              title="Cancel Job"
                            >
                              <XCircle className="h-3.5 w-3.5" />
                            </button>
                          )}

                          {/* Retry if failed */}
                          {job && (job.status === 'FAILED' || job.status === 'CANCELLED') && (
                            <button
                              type="button"
                              onClick={(e) => handleRetryJob(job.id, e)}
                              className="p-1.5 text-neutral-400 hover:text-blue-400 transition-colors"
                              title="Retry Ingestion"
                            >
                              <RotateCcw className="h-3.5 w-3.5" />
                            </button>
                          )}

                          {/* Reindex */}
                          {doc.status === 'ready' && (
                            <button
                              type="button"
                              onClick={(e) => handleReindex(doc.id, e)}
                              className="p-1.5 text-neutral-400 hover:text-white transition-colors"
                              title="Reindex Document"
                            >
                              <RefreshCw className="h-3.5 w-3.5" />
                            </button>
                          )}

                          {/* Download original */}
                          <button
                            type="button"
                            onClick={() => openDocument(doc.id)}
                            className="p-1.5 text-neutral-400 hover:text-white transition-colors"
                            title="Download Original"
                          >
                            <Download className="h-3.5 w-3.5" />
                          </button>

                          {/* Delete */}
                          <button
                            type="button"
                            onClick={(e) => handleDelete(doc.id, e)}
                            className="p-1.5 text-neutral-500 hover:text-red-400 transition-colors"
                            title="Delete Document"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── Slide-Over Chunk & Table Inspector Drawer ─────────────────────── */}
      <ChunkInspectorDrawer
        document={inspectingDoc}
        isOpen={Boolean(inspectingDoc)}
        onClose={() => setInspectingDoc(null)}
      />
    </div>
  );
}
