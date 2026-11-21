"use client";

/**
 * /chat — The primary desktop and web chat view.
 *
 * Uses client-side query parameters (?sessionId=...) to support static export
 * (output: 'export') for Tauri 2 desktop distribution without dynamic server segments.
 *
 * Layout:
 *   ┌─────────────────────────────────────────────────────────┐
 *   │  Sidebar (w-72)         │  ChatWindow (flex-1)          │
 *   │  ┌──────────────────┐   │                               │
 *   │  │ Logo + tagline   │   │  Message history              │
 *   │  │──────────────────│   │  Streaming bubble             │
 *   │  │ FileUploader     │   │  Input area                   │
 *   │  │──────────────────│   │                               │
 *   │  │ Session list     │   │                               │
 *   └─────────────────────────────────────────────────────────┘
 */

import React, { Suspense, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useChatStore } from '@/store/useChatStore';
import { useDocumentStore } from '@/store/useDocumentStore';
import {
  getHistory,
  getSessions,
  deleteSession,
  logout,
  errorMessage,
  createSession,
} from '@/lib/api';
import { FileUploader } from '@/components/FileUploader';
import { ChatWindow } from '@/components/ChatWindow';
import {
  Sparkles,
  LayoutGrid,
  MessageSquare,
  Trash2,
  Plus,
  ChevronLeft,
  ChevronRight,
  Loader2,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// ─── SESSION LIST ─────────────────────────────────────────────────────────────

interface SessionMeta {
  id: string;
  title: string | null;
  created_at: string;
}

interface SessionListProps {
  sessions: SessionMeta[];
  currentSessionId: string;
  onDelete: (id: string) => void;
}

function SessionList({ sessions, currentSessionId, onDelete }: SessionListProps) {
  const router = useRouter();
  const [error, setError] = useState('');

  const handleNew = async () => {
    try {
      const res = await createSession([]);
      router.push(`/chat/?sessionId=${res.session_id}`);
    } catch (error) {
      setError(errorMessage(error));
    }
  };

  return (
    <div className="flex flex-col gap-1">
      {error && <p role="alert" className="text-xs text-red-400">{error}</p>}
      <div className="flex items-center justify-between px-1 mb-1">
        <span className="text-[10px] font-semibold text-neutral-500 uppercase tracking-widest">
          Sessions
        </span>
        <button
          onClick={handleNew}
          id="new-session-btn"
          className="p-1 rounded-md text-neutral-500 hover:text-neutral-200 hover:bg-neutral-700/50 transition-colors"
          aria-label="New session"
        >
          <Plus className="w-3.5 h-3.5" />
        </button>
      </div>

      {sessions.length === 0 && (
        <p className="text-xs text-neutral-600 text-center py-3">No previous sessions</p>
      )}

      {sessions.map((s) => {
        const isActive = s.id === currentSessionId;
        const label = s.title ?? `Session ${s.id.slice(0, 8)}`;
        const date = new Date(s.created_at).toLocaleDateString(undefined, {
          month: 'short',
          day: 'numeric',
        });

        return (
          <div
            key={s.id}
            className={cn(
              'group flex items-center gap-2.5 px-3 py-2.5 rounded-xl transition-all cursor-pointer',
              isActive
                ? 'bg-blue-500/15 border border-blue-500/25'
                : 'hover:bg-neutral-800/60 border border-transparent',
            )}
            onClick={() => router.push(`/chat/?sessionId=${s.id}`)}
            role="button"
            tabIndex={0}
            aria-label={`Switch to ${label}`}
          >
            <MessageSquare
              className={cn(
                'w-3.5 h-3.5 flex-shrink-0',
                isActive ? 'text-blue-400' : 'text-neutral-500',
              )}
            />
            <div className="flex-1 min-w-0">
              <p className={cn('text-xs truncate', isActive ? 'text-blue-300' : 'text-neutral-300')}>
                {label}
              </p>
              <p className="text-[10px] text-neutral-600">{date}</p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDelete(s.id);
              }}
              className="opacity-0 group-hover:opacity-100 p-1 rounded text-neutral-600 hover:text-red-400 transition-all"
              aria-label={`Delete ${label}`}
            >
              <Trash2 className="w-3 h-3" />
            </button>
          </div>
        );
      })}
    </div>
  );
}

// ─── SIDEBAR ──────────────────────────────────────────────────────────────────

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  sessionId: string;
}

function Sidebar({ collapsed, onToggle, sessionId }: SidebarProps) {
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const [loadingSessions, setLoadingSessions] = useState(true);
  const router = useRouter();

  useEffect(() => {
    getSessions()
      .then((data) => {
        setSessions(data);
        if (!sessionId && data.length > 0) {
          router.replace(`/chat/?sessionId=${data[0].id}`);
        }
      })
      .catch(() => setSessions([]))
      .finally(() => setLoadingSessions(false));
  }, [sessionId, router]);

  const handleDeleteSession = async (id: string) => {
    try {
      await deleteSession(id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (id === sessionId) router.push('/chat/');
    } catch {
      /* silently ignore */
    }
  };

  if (collapsed) {
    return (
      <div className="flex flex-col items-center py-4 gap-4 w-14 glass rounded-2xl border-neutral-800/60 shadow-xl">
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center shadow-lg">
          <Sparkles className="w-4 h-4 text-white" />
        </div>
        <button
          onClick={onToggle}
          className="p-2 rounded-xl text-neutral-500 hover:text-neutral-200 hover:bg-neutral-700/50 transition-colors"
          aria-label="Expand sidebar"
        >
          <ChevronRight className="w-4 h-4" />
        </button>
        <LayoutGrid className="w-4 h-4 text-neutral-600 mt-2" />
      </div>
    );
  }

  return (
    <div className="w-72 flex-shrink-0 flex flex-col h-full glass rounded-2xl border-neutral-800/60 shadow-2xl overflow-hidden">
      {/* Brand header */}
      <div className="px-5 py-5 border-b border-neutral-800/60">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center shadow-lg">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-base font-bold text-gradient-brand">Quantiva</h1>
              <p className="text-[10px] text-neutral-500 -mt-0.5">Enterprise AI Agent</p>
            </div>
          </div>
          <button
            onClick={onToggle}
            className="p-1.5 rounded-lg text-neutral-500 hover:text-neutral-300 hover:bg-neutral-700/50 transition-colors"
            aria-label="Collapse sidebar"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-6">
        {/* File Uploader */}
        <div>
          <FileUploader />
        </div>

        {/* Session list */}
        <div>
          {loadingSessions ? (
            <div className="flex items-center gap-2 text-xs text-neutral-600 px-1">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Loading sessions…
            </div>
          ) : (
            <SessionList
              sessions={sessions}
              currentSessionId={sessionId}
              onDelete={handleDeleteSession}
            />
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-neutral-800/60">
        <button
          className="text-sm text-neutral-300"
          onClick={async () => {
            try {
              await logout();
              useChatStore.setState({
                messages: [],
                currentSession: null,
                isStreaming: false,
                streamingMessage: '',
              });
              useDocumentStore.setState({ documents: [] });
              router.replace('/login/');
            } catch (error) {
              window.alert(errorMessage(error));
            }
          }}
        >
          Sign out
        </button>
        <p className="text-[10px] text-neutral-600 text-center">
          Powered by Gemini · Qdrant · FastAPI
        </p>
      </div>
    </div>
  );
}

// ─── PAGE CONTENT (INSIDE SUSPENSE) ──────────────────────────────────────────

function ChatPageContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const sessionId = searchParams.get('sessionId') || searchParams.get('id') || '';

  const { setSession, setMessages } = useChatStore();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [loadedSessionId, setLoadedSessionId] = useState<string | null>(null);
  const [error, setError] = useState('');

  const loading = Boolean(sessionId && loadedSessionId !== sessionId);

  // Hydrate documents on mount
  useEffect(() => {
    useDocumentStore.getState().fetchDocuments();
  }, []);

  // Set session in Zustand and load history
  useEffect(() => {
    if (!sessionId) {
      return;
    }

    setSession(sessionId);

    let active = true;
    getHistory(sessionId)
      .then((history) => {
        if (!active) return;
        const msgs = history.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.content,
          citations: m.citations,
          confidence: m.confidence,
          chart_url: m.chart_url,
          guardrail_flags: m.guardrail_flags,
          cache_hit: m.cache_hit,
          timestamp: m.created_at ? Date.parse(m.created_at) : undefined,
        }));
        setMessages(msgs);
      })
      .catch((error) => {
        if (active) {
          setMessages([]);
          setError(errorMessage(error));
        }
      })
      .finally(() => {
        if (active) setLoadedSessionId(sessionId);
      });

    return () => {
      active = false;
    };
  }, [sessionId, setSession, setMessages, router]);

  return (
    <div className="flex h-screen bg-[#0a0a0a] p-3 gap-3 overflow-hidden">
      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <aside
        className={cn(
          'hidden md:flex flex-col transition-all duration-300 ease-in-out',
          sidebarCollapsed ? 'w-14' : 'w-72',
        )}
        aria-label="Sidebar navigation"
      >
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed((v) => !v)}
          sessionId={sessionId}
        />
      </aside>

      {/* ── Main chat area ───────────────────────────────────────────────── */}
      <main className="flex-1 min-w-0 h-full" role="main" aria-label="Chat interface">
        {error ? (
          <p role="alert" className="p-8 text-red-400">
            {error}
          </p>
        ) : loading ? (
          <div className="h-full flex items-center justify-center">
            <div className="flex items-center gap-3 text-neutral-500">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span className="text-sm">Loading conversation…</span>
            </div>
          </div>
        ) : (
          <ChatWindow />
        )}
      </main>
    </div>
  );
}

// ─── ROOT PAGE WITH SUSPENSE BOUNDARY ────────────────────────────────────────

export default function ChatPage() {
  return (
    <Suspense
      fallback={
        <div className="h-screen bg-[#0a0a0a] flex items-center justify-center text-neutral-500">
          <div className="flex items-center gap-3">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span className="text-sm">Loading workspace…</span>
          </div>
        </div>
      }
    >
      <ChatPageContent />
    </Suspense>
  );
}
