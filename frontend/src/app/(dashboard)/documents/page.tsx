"use client";

import React, { useEffect, useState, useCallback } from 'react';
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
} from 'lucide-react';
import {
  getDocuments,
  deleteDocument,
  reindexDocument,
  openDocument,
  DocumentItem,
} from '@/lib/api';
import { useDropzone } from 'react-dropzone';
import { api } from '@/lib/api';

export default function DocumentManagerPage() {
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const fetchDocs = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getDocuments();
      setDocs(data);
    } catch (err) {
      console.warn('Failed to fetch documents:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchDocs();
  }, [fetchDocs]);

  // Dropzone file upload handler
  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;
      setUploading(true);
      setUploadError(null);
      try {
        for (const file of acceptedFiles) {
          const formData = new FormData();
          formData.append('file', file);
          await api.post('/documents', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
          });
        }
        await fetchDocs();
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : 'Upload failed');
      } finally {
        setUploading(false);
      }
    },
    [fetchDocs]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'text/csv': ['.csv'],
    },
  });

  const handleDelete = async (id: string) => {
    if (!window.confirm('Are you sure you want to delete this document?')) return;
    try {
      await deleteDocument(id);
      setDocs((prev) => prev.filter((d) => d.id !== id));
    } catch (err) {
      alert('Failed to delete document');
    }
  };

  const handleReindex = async (id: string) => {
    try {
      await reindexDocument(id);
      await fetchDocs();
    } catch (err) {
      alert('Failed to reindex document');
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const filtered = docs.filter((d) =>
    d.filename.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">Document Manager</h1>
          <p className="text-xs text-neutral-400 mt-1">
            Ingest, parse, and monitor documents for hybrid lexical and semantic retrieval.
          </p>
        </div>

        <button
          type="button"
          onClick={fetchDocs}
          className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-neutral-400 hover:text-white transition-colors"
          title="Refresh"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Upload Dropzone */}
      <div
        {...getRootProps()}
        className={`cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition-all ${
          isDragActive
            ? 'border-blue-500 bg-blue-500/10'
            : 'border-white/10 bg-black/30 hover:border-white/20 hover:bg-white/[0.02]'
        }`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center space-y-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-blue-500/10 text-blue-400">
            {uploading ? <Loader2 className="h-6 w-6 animate-spin" /> : <UploadCloud className="h-6 w-6" />}
          </div>
          <div>
            <p className="text-sm font-semibold text-white">
              {uploading ? 'Processing and indexing document…' : 'Drop documents here, or click to browse'}
            </p>
            <p className="text-xs text-neutral-500 mt-1">Supports PDF, DOCX, TXT, MD, CSV (Processed 100% locally)</p>
          </div>
        </div>
      </div>

      {uploadError && (
        <div className="rounded-xl border border-red-500/20 bg-red-950/20 p-3 text-xs text-red-300">
          {uploadError}
        </div>
      )}

      {/* Search & Document List */}
      <div className="space-y-4">
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-neutral-500" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search documents by filename…"
            className="w-full rounded-xl border border-white/10 bg-black/40 pl-9 pr-4 py-2 text-xs text-white placeholder-neutral-500 focus:border-blue-500 focus:outline-none"
          />
        </div>

        {loading ? (
          <div className="py-16 text-center text-xs text-neutral-500">Loading documents…</div>
        ) : filtered.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-white/10 p-12 text-center text-neutral-500 text-xs">
            No documents found.
          </div>
        ) : (
          <div className="overflow-hidden rounded-xl border border-white/10 bg-black/30">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-white/5 bg-white/[0.02] text-neutral-400">
                <tr>
                  <th className="px-4 py-3 font-medium">Filename</th>
                  <th className="px-4 py-3 font-medium">Size</th>
                  <th className="px-4 py-3 font-medium">Status</th>
                  <th className="px-4 py-3 font-medium">Chunks</th>
                  <th className="px-4 py-3 font-medium text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-neutral-300">
                {filtered.map((doc) => (
                  <tr key={doc.id} className="hover:bg-white/[0.02] transition-colors">
                    <td className="px-4 py-3 flex items-center gap-2.5 font-medium text-white truncate max-w-xs">
                      <FileText className="h-4 w-4 shrink-0 text-blue-400" />
                      <span className="truncate">{doc.filename}</span>
                    </td>
                    <td className="px-4 py-3 font-mono text-neutral-400">{formatSize(doc.file_size)}</td>
                    <td className="px-4 py-3">
                      {doc.status === 'ready' ? (
                        <span className="inline-flex items-center gap-1 rounded bg-teal-500/10 px-2 py-0.5 text-[10px] font-medium text-teal-400 border border-teal-500/20">
                          <CheckCircle2 className="h-3 w-3" /> Ready
                        </span>
                      ) : doc.status === 'processing' || doc.status === 'uploading' ? (
                        <span className="inline-flex items-center gap-1 rounded bg-blue-500/10 px-2 py-0.5 text-[10px] font-medium text-blue-400 border border-blue-500/20">
                          <Loader2 className="h-3 w-3 animate-spin" /> Indexing
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 rounded bg-red-500/10 px-2 py-0.5 text-[10px] font-medium text-red-400 border border-red-500/20">
                          <AlertCircle className="h-3 w-3" /> Failed
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 font-mono text-neutral-400">{doc.chunk_count ?? 0}</td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          type="button"
                          onClick={() => openDocument(doc.id)}
                          className="p-1 text-neutral-400 hover:text-white transition-colors"
                          title="Download"
                        >
                          <Download className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => handleReindex(doc.id)}
                          className="p-1 text-neutral-400 hover:text-white transition-colors"
                          title="Reindex"
                        >
                          <RefreshCw className="h-3.5 w-3.5" />
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDelete(doc.id)}
                          className="p-1 text-red-400/80 hover:text-red-400 transition-colors"
                          title="Delete"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
