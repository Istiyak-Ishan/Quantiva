"use client";

import React, { useEffect, useState } from 'react';
import {
  Database,
  Plus,
  RefreshCw,
  Search,
  CheckCircle2,
  AlertCircle,
  HardDrive,
  Trash2,
  Layers,
  Sparkles,
} from 'lucide-react';
import { getKnowledgeBases, createKnowledgeBase, KnowledgeBaseItem } from '@/lib/api';

export default function KnowledgeBasesPage() {
  const [kbs, setKbs] = useState<KnowledgeBaseItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [creating, setCreating] = useState(false);

  const fetchKBs = async () => {
    setLoading(true);
    try {
      const data = await getKnowledgeBases();
      setKbs(data);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchKBs();
  }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    try {
      await createKnowledgeBase({
        name: newName.trim(),
        description: newDesc.trim() || null,
      });
      setNewName('');
      setNewDesc('');
      setShowCreateModal(false);
      await fetchKBs();
    } finally {
      setCreating(false);
    }
  };

  const filtered = kbs.filter((k) =>
    k.name.toLowerCase().includes(search.toLowerCase()) ||
    (k.description && k.description.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">Knowledge Bases</h1>
          <p className="text-xs text-neutral-400 mt-1">
            Isolated document collections with locked embedding dimensions and dedicated vector indexes.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={fetchKBs}
            className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-neutral-400 hover:text-white transition-colors"
            title="Refresh"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>

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

      {/* Filter / Search Bar */}
      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-neutral-500" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search knowledge bases…"
          className="w-full rounded-xl border border-white/10 bg-black/40 pl-9 pr-4 py-2 text-xs text-white placeholder-neutral-500 focus:border-blue-500 focus:outline-none"
        />
      </div>

      {/* Grid of Knowledge Bases */}
      {loading ? (
        <div className="py-16 text-center text-xs text-neutral-500">Loading knowledge bases…</div>
      ) : filtered.length === 0 ? (
        <div className="rounded-2xl border border-dashed border-white/10 p-12 text-center space-y-3">
          <Database className="mx-auto h-10 w-10 text-neutral-600" />
          <p className="text-sm font-medium text-neutral-300">No knowledge bases found</p>
          <p className="text-xs text-neutral-500">
            {search ? 'No results matched your search query.' : 'Create your first knowledge base to begin.'}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {filtered.map((kb) => (
            <div
              key={kb.id}
              className="group relative flex flex-col justify-between rounded-xl border border-white/10 bg-black/30 p-5 hover:border-white/20 hover:bg-white/[0.02] transition-all"
            >
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-500/10 text-blue-400">
                    <Database className="h-4 w-4" />
                  </div>
                  <span className="inline-flex items-center gap-1 rounded bg-teal-500/10 px-2 py-0.5 text-[10px] font-mono text-teal-400 border border-teal-500/20">
                    <CheckCircle2 className="h-3 w-3" />
                    {kb.embedding_dimensions ?? 384}-d
                  </span>
                </div>

                <h3 className="text-sm font-semibold text-white group-hover:text-blue-400 transition-colors">
                  {kb.name}
                </h3>
                <p className="mt-1 text-xs text-neutral-400 line-clamp-2">
                  {kb.description || 'Dedicated private document intelligence workspace.'}
                </p>
              </div>

              <div className="mt-5 pt-4 border-t border-white/5 flex items-center justify-between text-xs text-neutral-500 font-mono">
                <span>{kb.document_count ?? 0} docs</span>
                <span>{kb.chunk_count ?? 0} chunks</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-2xl border border-white/10 bg-[#0e0e12] p-6 shadow-2xl">
            <h3 className="text-base font-bold text-white">Create Knowledge Base</h3>
            <p className="text-xs text-neutral-400 mt-1">
              Lock in an embedding dimension for this knowledge base. All documents will be indexed into its dedicated vector collection.
            </p>

            <form onSubmit={handleCreate} className="mt-5 space-y-4">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-neutral-300">Name</label>
                <input
                  type="text"
                  required
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="e.g. Legal Contracts, Product Specs"
                  className="w-full rounded-lg border border-white/10 bg-neutral-900 px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-neutral-300">Description</label>
                <textarea
                  rows={3}
                  value={newDesc}
                  onChange={(e) => setNewDesc(e.target.value)}
                  placeholder="Optional notes about what this collection holds"
                  className="w-full rounded-lg border border-white/10 bg-neutral-900 px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none resize-none"
                />
              </div>

              <div className="flex items-center justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowCreateModal(false)}
                  className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-neutral-300 hover:bg-white/10"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={creating || !newName.trim()}
                  className="rounded-lg bg-blue-600 px-4 py-1.5 text-xs font-semibold text-white shadow-md shadow-blue-600/30 hover:bg-blue-500 disabled:opacity-50"
                >
                  {creating ? 'Creating…' : 'Create'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
