"use client";

import React, { useState, useEffect } from 'react';
import {
  Database,
  X,
  Layers,
  Sparkles,
  Loader2,
  AlertTriangle,
  CheckCircle2,
  ShieldCheck,
} from 'lucide-react';
import {
  createKnowledgeBase,
  getEmbeddingConfigs,
  EmbeddingConfig,
} from '@/lib/api';

interface CreateKBModalProps {
  isOpen: boolean;
  onClose: () => void;
  onCreated: () => void;
}

export function CreateKBModal({ isOpen, onClose, onCreated }: CreateKBModalProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [embeddingConfigs, setEmbeddingConfigs] = useState<EmbeddingConfig[]>([]);
  const [selectedEmbeddingId, setSelectedEmbeddingId] = useState<string>('');
  const [loadingConfigs, setLoadingConfigs] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    let active = true;

    async function loadEmbeddings() {
      setLoadingConfigs(true);
      try {
        const configs = await getEmbeddingConfigs();
        if (active) {
          setEmbeddingConfigs(configs);
          const defaultCfg = configs.find((c) => c.is_default) || configs[0];
          if (defaultCfg) {
            setSelectedEmbeddingId(defaultCfg.id);
          }
        }
      } catch (err) {
        console.warn('Failed to fetch embedding configs:', err);
      } finally {
        if (active) setLoadingConfigs(false);
      }
    }

    void loadEmbeddings();
    return () => {
      active = false;
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;

    setSubmitting(true);
    setError(null);
    try {
      await createKnowledgeBase({
        name: name.trim(),
        description: description.trim() || null,
        embedding_config_id: selectedEmbeddingId || null,
      });
      setName('');
      setDescription('');
      onCreated();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create Knowledge Base');
    } finally {
      setSubmitting(false);
    }
  };

  const selectedConfig = embeddingConfigs.find((c) => c.id === selectedEmbeddingId);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
      <div className="relative w-full max-w-lg rounded-2xl border border-white/10 bg-[#0e0e13] shadow-2xl overflow-hidden">
        {/* Glow decoration */}
        <div className="pointer-events-none absolute -top-16 left-1/2 -translate-x-1/2 h-32 w-72 rounded-full bg-blue-600/20 blur-3xl" />

        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-white/10 px-6 py-5">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-500/10 text-blue-400">
              <Database className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-tight">Create Knowledge Base</h2>
              <p className="text-xs text-neutral-400">Dedicated vector space with locked embedding dimensions.</p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1 text-neutral-400 hover:bg-white/5 hover:text-white transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Modal Body Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-5">
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-neutral-300">Knowledge Base Name</label>
            <input
              type="text"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Legal Contracts 2026, Clinical Research"
              className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3.5 py-2 text-xs text-white placeholder-neutral-500 focus:border-blue-500 focus:outline-none"
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-medium text-neutral-300">Description (Optional)</label>
            <textarea
              rows={2}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Brief summary of documents stored in this collection"
              className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3.5 py-2 text-xs text-white placeholder-neutral-500 focus:border-blue-500 focus:outline-none resize-none"
            />
          </div>

          {/* Bound Embedding Configuration Selector */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-neutral-300 flex items-center gap-1.5">
                <Layers className="h-3.5 w-3.5 text-teal-400" />
                Bound Embedding Model
              </label>
              {selectedConfig && (
                <span className="rounded bg-teal-500/10 px-2 py-0.5 text-[10px] font-mono text-teal-400 border border-teal-500/20">
                  Locks at {selectedConfig.dimensions} dimensions
                </span>
              )}
            </div>

            {loadingConfigs ? (
              <div className="flex items-center gap-2 text-xs text-neutral-500 py-2">
                <Loader2 className="h-3.5 w-3.5 animate-spin text-teal-400" />
                Loading embedding configurations…
              </div>
            ) : embeddingConfigs.length > 0 ? (
              <select
                value={selectedEmbeddingId}
                onChange={(e) => setSelectedEmbeddingId(e.target.value)}
                className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3.5 py-2 text-xs text-white focus:border-blue-500 focus:outline-none font-mono"
              >
                {embeddingConfigs.map((cfg) => (
                  <option key={cfg.id} value={cfg.id}>
                    {cfg.name} ({cfg.model_name} • {cfg.dimensions}-d)
                  </option>
                ))}
              </select>
            ) : (
              <p className="text-xs text-neutral-500">
                Using default local Sentence-Transformers (384 dimensions).
              </p>
            )}

            <p className="text-[10px] text-neutral-500">
              The embedding model determines the vector geometry of this knowledge base. All documents uploaded here will
              be encoded with this exact model.
            </p>
          </div>

          {error && (
            <div className="rounded-xl border border-red-500/20 bg-red-950/20 p-3 text-xs text-red-300 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Footer Actions */}
          <div className="flex items-center justify-end gap-3 pt-3 border-t border-white/5">
            <button
              type="button"
              onClick={onClose}
              disabled={submitting}
              className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold text-neutral-300 hover:bg-white/10 transition-colors"
            >
              Cancel
            </button>

            <button
              type="submit"
              disabled={submitting || !name.trim()}
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-5 py-2 text-xs font-semibold text-white shadow-md shadow-blue-600/30 hover:bg-blue-500 disabled:opacity-50 transition-colors"
            >
              {submitting ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Creating Workspace…
                </>
              ) : (
                <>
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Create Knowledge Base
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
