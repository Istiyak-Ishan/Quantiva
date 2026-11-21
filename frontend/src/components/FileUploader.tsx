"use client";

/**
 * FileUploader — production-grade drag-and-drop uploader
 *
 * Flow:
 *   1. User drops / selects file
 *   2. Optimistic doc added with status='uploading'
 *   3. File POSTed to /documents/upload with XHR progress tracking
 *   4. On success: tempId replaced with real doc id, status='processing'
 *   5. SSE EventSource opened on /documents/{id}/status
 *   6. SSE events update status until 'ready' or 'failed'
 *   7. EventSource closed automatically
 */

import React, { useCallback, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  UploadCloud,
  FileText,
  CheckCircle,
  Loader2,
  XCircle,
  X,
  FileSpreadsheet,
  FileType,
  AlertCircle,
  RefreshCw,
} from 'lucide-react';
import { useDocumentStore, Document } from '@/store/useDocumentStore';
import {
  uploadDocument,
  watchDocumentStatus,
  deleteDocument,
  reindexDocument,
} from '@/lib/api';
import type { DesktopEventSourceLike } from '@/lib/api-client';
import { cn } from '@/lib/utils';

// ─── FILE TYPE ICON ───────────────────────────────────────────────────────────

function FileIcon({ filename }: { filename: string }) {
  const ext = filename.split('.').pop()?.toLowerCase();
  if (ext === 'pdf') return <FileType className="w-4 h-4 text-red-400 flex-shrink-0" />;
  if (ext === 'xlsx' || ext === 'csv')
    return <FileSpreadsheet className="w-4 h-4 text-emerald-400 flex-shrink-0" />;
  return <FileText className="w-4 h-4 text-blue-400 flex-shrink-0" />;
}

// ─── STATUS BADGE ─────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: Document['status'] }) {
  if (status === 'ready') {
    return (
      <span className="badge-ready flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold">
        <CheckCircle className="w-3 h-3" />
        Ready
      </span>
    );
  }
  if (status === 'processing' || status === 'uploading' || status === 'pending') {
    return (
      <span className="badge-processing flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold">
        <Loader2 className="w-3 h-3 animate-spin" />
        {status === 'uploading' ? 'Uploading…' : 'Indexing…'}
      </span>
    );
  }
  if (status === 'failed') {
    return (
      <span className="badge-failed flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold">
        <AlertCircle className="w-3 h-3" />
        Failed
      </span>
    );
  }
  return null;
}

// ─── UPLOAD PROGRESS BAR ──────────────────────────────────────────────────────

function ProgressBar({ progress }: { progress: number }) {
  return (
    <div className="w-full h-1 bg-neutral-700/50 rounded-full overflow-hidden">
      <div
        className={cn(
          'h-full rounded-full transition-all duration-200',
          progress < 100 ? 'progress-shimmer' : 'bg-emerald-500',
        )}
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}

// ─── DOCUMENT ITEM ────────────────────────────────────────────────────────────

interface DocItemProps {
  doc: Document;
  onRemove: (id: string) => void;
  onRetry: (id: string) => void;
}

function DocItem({ doc, onRemove, onRetry }: DocItemProps) {
  const showProgress =
    (doc.status === 'uploading' || doc.status === 'processing') &&
    doc.uploadProgress > 0 &&
    doc.uploadProgress < 100;

  return (
    <div className="group flex flex-col gap-1.5 p-3 rounded-xl glass glass-hover transition-all">
      <div className="flex items-center gap-2.5">
        <FileIcon filename={doc.filename} />
        <span className="flex-1 text-sm text-neutral-200 truncate min-w-0">
          {doc.filename}
        </span>
        <StatusBadge status={doc.status} />
        {doc.status === 'failed' && !doc.id.startsWith('temp-') && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onRetry(doc.id);
            }}
            className="p-1 rounded text-amber-400 hover:text-amber-200 hover:bg-amber-500/20 transition-all"
            title={`Retry indexing ${doc.filename}`}
            aria-label={`Retry indexing ${doc.filename}`}
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        )}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onRemove(doc.id);
          }}
          className={cn(
            'p-1 rounded transition-all',
            doc.status === 'failed'
              ? 'opacity-100 text-red-400 hover:text-red-200 hover:bg-red-500/20'
              : 'opacity-0 group-hover:opacity-100 text-neutral-500 hover:text-neutral-200 hover:bg-neutral-700/50',
          )}
          title={`Delete ${doc.filename}`}
          aria-label={`Remove ${doc.filename}`}
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {showProgress && <ProgressBar progress={doc.uploadProgress} />}

      {doc.status === 'ready' && doc.chunkCount != null && (
        <p className="text-[10px] text-neutral-600 pl-6">
          {doc.chunkCount} chunks indexed
        </p>
      )}

      {doc.status === 'failed' && doc.failureReason && (
        <p className="text-[10px] text-red-400/80 pl-6 leading-snug">
          {doc.failureReason}
        </p>
      )}
    </div>
  );
}

// ─── FILE UPLOADER ────────────────────────────────────────────────────────────

// Track EventSource refs so we can clean up if the component unmounts
const activeEventSources = new Map<string, EventSource | DesktopEventSourceLike>();

export function FileUploader() {
  const { documents, fetchDocuments, addDocument, updateStatus, removeDocument, replaceId } =
    useDocumentStore();
  const esRefs = useRef(activeEventSources);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const watch = useCallback(
    (id: string) => {
      esRefs.current.get(id)?.close();
      const es = watchDocumentStatus(
        id,
        (statusData) => {
          updateStatus(
            id,
            statusData.status,
            100,
            statusData.chunk_count,
            statusData.failure_reason ?? null,
          );
          if (statusData.status === 'ready' || statusData.status === 'failed') {
            esRefs.current.delete(id);
          }
        },
        () => {
          // SSE error — surface an actionable failure instead of a spinner.
          updateStatus(
            id,
            'failed',
            undefined,
            undefined,
            'Lost connection while indexing. Retry indexing.',
          );
          esRefs.current.delete(id);
        },
      );

      esRefs.current.set(id, es);
    },
    [updateStatus],
  );

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      for (const file of acceptedFiles) {
        // ── 1. Optimistic entry ─────────────────────────────────────────────
        const tempId = `temp-${crypto.randomUUID()}`;
        addDocument({
          id: tempId,
          filename: file.name,
          status: 'uploading',
          uploadProgress: 0,
        });

        try {
          // ── 2. Upload with progress callback ───────────────────────────────
          const res = await uploadDocument(file, (pct) => {
            updateStatus(tempId, 'uploading', pct);
          });

          // ── 3. Replace temp id with real server id ─────────────────────────
          replaceId(tempId, res.id);
          updateStatus(res.id, 'processing', 100, undefined, null);

          // ── 4. Watch SSE status until 'ready' ─────────────────────────────
          watch(res.id);
        } catch {
          updateStatus(tempId, 'failed', undefined, undefined, 'Upload failed. Retry the upload.');
        }
      }
    },
    [addDocument, updateStatus, replaceId, watch],
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } =
    useDropzone({
      onDrop,
      accept: {
        'application/pdf': ['.pdf'],
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
        'text/csv': ['.csv'],
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      },
      maxSize: 25 * 1024 * 1024, // Matches backend default MAX_UPLOAD_BYTES
      multiple: true,
    });

  const handleRemove = async (id: string) => {
    // Close SSE if still watching
    esRefs.current.get(id)?.close();
    esRefs.current.delete(id);
    if (id.startsWith('temp-')) { removeDocument(id); return; }
    try {
      await deleteDocument(id);
      removeDocument(id);
    } catch {
      window.alert('Document cleanup failed. Please retry deletion.');
    }
  };

  const handleRetry = async (id: string) => {
    esRefs.current.get(id)?.close();
    esRefs.current.delete(id);
    try {
      await reindexDocument(id);
      updateStatus(id, 'processing', 100, undefined, null);
      watch(id);
    } catch {
      updateStatus(
        id,
        'failed',
        undefined,
        undefined,
        'Retry failed. The ingestion worker may be unavailable.',
      );
    }
  };

  return (
    <div className="flex flex-col gap-3 w-full">
      {/* ── Drop zone ─────────────────────────────────────────────────────── */}
      <div
        {...getRootProps()}
        id="file-dropzone"
        className={cn(
          'relative border-2 border-dashed rounded-2xl p-6 flex flex-col items-center justify-center',
          'cursor-pointer transition-all duration-200 overflow-hidden',
          isDragActive && !isDragReject
            ? 'border-blue-500 bg-blue-500/8 dropzone-active'
            : isDragReject
              ? 'border-red-500 bg-red-500/8'
              : 'border-neutral-700 bg-neutral-900/30 hover:border-neutral-500 hover:bg-neutral-800/30',
        )}
        aria-label="Drop zone for document upload"
      >
        <input {...getInputProps()} aria-label="File input" />

        {/* Floating orb */}
        <div
          className={cn(
            'w-12 h-12 rounded-2xl flex items-center justify-center mb-3 transition-all duration-200',
            isDragActive && !isDragReject
              ? 'bg-blue-500/20 scale-110'
              : isDragReject
                ? 'bg-red-500/20'
                : 'bg-neutral-800',
          )}
        >
          {isDragReject ? (
            <XCircle className="w-6 h-6 text-red-400" />
          ) : (
            <UploadCloud
              className={cn(
                'w-6 h-6 transition-colors',
                isDragActive ? 'text-blue-400' : 'text-neutral-400',
              )}
            />
          )}
        </div>

        <p className="text-sm font-medium text-neutral-200 text-center">
          {isDragReject
            ? 'File type not supported'
            : isDragActive
              ? 'Drop to upload'
              : 'Drag & drop files here'}
        </p>
        <p className="text-xs text-neutral-500 mt-1 text-center">
          {isDragReject
            ? 'Supported: PDF, DOCX, CSV, XLSX'
            : 'PDF, DOCX, CSV, XLSX · Max 25 MB'}
        </p>

        {!isDragActive && (
          <button
            type="button"
            className="mt-4 px-4 py-1.5 rounded-lg bg-neutral-800 border border-neutral-700 text-xs text-neutral-300 hover:bg-neutral-700 hover:border-neutral-600 transition-colors"
          >
            Browse files
          </button>
        )}
      </div>

      {/* ── Document list ─────────────────────────────────────────────────── */}
      {documents.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between px-1">
            <span className="text-[10px] font-semibold text-neutral-500 uppercase tracking-widest">
              Knowledge Base
            </span>
            <span className="text-[10px] text-neutral-600">
              {documents.filter((d) => d.status === 'ready').length}/
              {documents.length} ready
            </span>
          </div>
          {documents.map((doc) => (
            <DocItem key={doc.id} doc={doc} onRemove={handleRemove} onRetry={handleRetry} />
          ))}
        </div>
      )}
    </div>
  );
}
