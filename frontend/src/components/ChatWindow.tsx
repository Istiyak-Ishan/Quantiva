"use client";

/**
 * ChatWindow — the core chat interface
 *
 * Responsibilities:
 *  - Renders the message history (MessageBubble list)
 *  - Manages the EventSource lifecycle (open on send, close on done/error/unmount)
 *  - Handles optimistic user message insertion
 *  - Auto-scrolls to latest message
 *  - Shows a typing indicator while streaming
 *  - Shows an empty state with example prompts
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { MessageBubble } from './MessageBubble';
import { useChatStore } from '@/store/useChatStore';
import { useDocumentStore } from '@/store/useDocumentStore';
import { streamMessage, StreamEventData } from '@/lib/api';
import type { DesktopEventSourceLike } from '@/lib/api-client';
import {
  Send,
  Loader2,
  StopCircle,
  Sparkles,
  MessageSquare,
} from 'lucide-react';
import { cn } from '@/lib/utils';

// ─── EXAMPLE PROMPTS ──────────────────────────────────────────────────────────

const EXAMPLE_PROMPTS = [
  'Summarize the key findings of this document',
  'What are the main revenue figures mentioned?',
  'List all action items and deadlines',
  'Are there any risk factors highlighted?',
];

// ─── EMPTY STATE ──────────────────────────────────────────────────────────────

function EmptyState({
  hasDocuments,
  onPrompt,
}: {
  hasDocuments: boolean;
  onPrompt: (p: string) => void;
}) {
  return (
    <div className="h-full flex flex-col items-center justify-center gap-6 px-8 text-center">
      {/* Icon */}
      <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 to-violet-500/20 border border-blue-500/20 flex items-center justify-center">
        <Sparkles className="w-8 h-8 text-blue-400" />
      </div>

      <div>
        <h3 className="text-lg font-semibold text-neutral-200 mb-1">
          {hasDocuments ? 'Ask anything about your documents' : 'Upload a document to begin'}
        </h3>
        <p className="text-sm text-neutral-500 max-w-sm">
          {hasDocuments
            ? 'Quantiva will search your documents and return cited answers with confidence scores.'
            : 'Drag & drop a PDF, DOCX, CSV, or XLSX file in the sidebar to get started.'}
        </p>
      </div>

      {hasDocuments && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
          {EXAMPLE_PROMPTS.map((prompt) => (
            <button
              key={prompt}
              onClick={() => onPrompt(prompt)}
              className="text-left p-3 rounded-xl glass glass-hover transition-all text-sm text-neutral-300 hover:text-neutral-100 border border-transparent hover:border-neutral-700"
            >
              <MessageSquare className="w-3.5 h-3.5 text-blue-400 mb-1.5" />
              {prompt}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── TYPING INDICATOR ─────────────────────────────────────────────────────────

function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center shadow-lg flex-shrink-0">
          <Sparkles className="w-4 h-4 text-white" />
        </div>
        <div className="glass rounded-2xl rounded-bl-md px-5 py-4">
          <div className="flex gap-1.5 items-center h-4">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="w-2 h-2 rounded-full bg-neutral-400"
                style={{
                  animation: 'blink 1.2s ease-in-out infinite',
                  animationDelay: `${i * 0.2}s`,
                }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── CHAT WINDOW ──────────────────────────────────────────────────────────────

export function ChatWindow() {
  const {
    messages,
    currentSession,
    isStreaming,
    streamingMessage,
    abortStreaming,
  } = useChatStore();

  const { readyDocIds } = useDocumentStore();
  const readyDocs = readyDocIds();

  const [input, setInput] = useState('');
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState('');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | DesktopEventSourceLike | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // ── Auto-scroll ──────────────────────────────────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingMessage]);

  // ── Auto-resize textarea ──────────────────────────────────────────────────
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 200)}px`;
  }, [input]);

  // ── Cleanup EventSource on unmount ────────────────────────────────────────
  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  // ── Stop streaming ────────────────────────────────────────────────────────
  const handleStop = useCallback(() => {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
    abortStreaming();
  }, [abortStreaming]);

  // ── Send message ──────────────────────────────────────────────────────────
  const handleSend = useCallback(
    async (overrideInput?: string) => {
      const query = (overrideInput ?? input).trim();
      const session = useChatStore.getState().currentSession;
      if (!query || !session || useChatStore.getState().isStreaming) return;

      setInput('');
      setError('');
      setIsConnecting(true);

      // Optimistic user message
      useChatStore.getState().addMessage({ role: 'user', content: query });

      // Close any existing SSE
      eventSourceRef.current?.close();

      try {
        const es = await streamMessage(session, query);
        eventSourceRef.current = es;

        es.onopen = () => setIsConnecting(false);

        es.onmessage = (e: MessageEvent) => {
          setIsConnecting(false);
          try {
            const data: StreamEventData = JSON.parse(e.data);
            if (!data.done) {
              if (data.token) {
                useChatStore.getState().updateStreamingMessage(data.token);
              }
            } else {
              useChatStore.getState().finalizeMessage(data.citations ?? [], data.confidence, data.chart_url, {
                ...(data.answer !== undefined ? { content: data.answer } : {}),
                guardrail_flags: data.guardrail_flags, cache_hit: data.cache_hit, cost_usd: data.cost_usd,
              });
              es.close();
              eventSourceRef.current = null;
            }
          } catch {
            useChatStore.getState().abortStreaming();
            setError('The server returned an invalid stream. Please retry.');
            es.close();
          }
        };

        es.onerror = (event: Event | MessageEvent) => {
          setIsConnecting(false);
          useChatStore.getState().abortStreaming();
          let message = 'Connection interrupted. Please verify backend status and retry.';
          if (event instanceof MessageEvent) {
            try {
              const data = JSON.parse(event.data);
              message = typeof data.error === 'string' ? data.error : data.error?.error ?? message;
            } catch { /* Keep the explicit connection error. */ }
          }
          setError(message);
          es.close();
          eventSourceRef.current = null;
        };
      } catch {
        setIsConnecting(false);
        useChatStore.getState().abortStreaming();
        setError('Unable to open an authenticated chat stream. Please sign in again or retry.');
      }
    },
    [
      input,
    ],
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const canSend =
    input.trim().length > 0 && !!currentSession && !isStreaming && !isConnecting;

  const showStreaming = isStreaming || isConnecting;
  const showTypingIndicator = isConnecting && !streamingMessage;

  return (
    <div className="flex flex-col h-full relative rounded-2xl glass overflow-hidden shadow-2xl">
      {error && <p role="alert" className="p-3 text-sm text-red-400">{error}</p>}
      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-800/60 bg-neutral-900/40">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <h2 className="text-base font-semibold text-neutral-100">
            Quantiva Assistant
          </h2>
        </div>

        <div className="flex items-center gap-2">
          {readyDocs.length === 0 ? (
            <span className="badge-processing text-[11px] px-2.5 py-1 rounded-full text-neutral-400 bg-neutral-800/60 border border-neutral-700/50">
              General Mode (Upload document to ground responses)
            </span>
          ) : (
            <span className="badge-ready text-[11px] px-2.5 py-1 rounded-full">
              {readyDocs.length} doc{readyDocs.length !== 1 ? 's' : ''} active
            </span>
          )}
        </div>
      </div>

      {/* ── Message list ────────────────────────────────────────────────── */}
      <div
        className="flex-1 overflow-y-auto px-6 py-6 flex flex-col gap-5"
        id="chat-message-list"
        role="log"
        aria-live="polite"
        aria-label="Chat messages"
      >
        {messages.length === 0 && !showStreaming ? (
          <EmptyState
            hasDocuments={readyDocs.length > 0}
            onPrompt={(p) => handleSend(p)}
          />
        ) : (
          <>
            {messages.map((msg, i) => (
              <MessageBubble key={msg.id ?? i} message={msg} />
            ))}

            {/* Live streaming bubble */}
            {isStreaming && streamingMessage && (
              <MessageBubble
                message={{ role: 'assistant', content: streamingMessage }}
                isStreaming
              />
            )}

            {/* Typing indicator (connecting but no content yet) */}
            {showTypingIndicator && <TypingIndicator />}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* ── Input area ──────────────────────────────────────────────────── */}
      <div className="px-4 pb-4 pt-3 border-t border-neutral-800/60 bg-neutral-900/30">
        <div
          className={cn(
            'flex items-end gap-3 rounded-2xl border px-4 py-3 transition-all',
            'bg-neutral-900/80',
            input.length > 0
              ? 'border-neutral-600'
              : 'border-neutral-700/60',
            'focus-within:border-blue-500/60 focus-within:shadow-sm focus-within:shadow-blue-500/10',
          )}
        >
          <textarea
            ref={textareaRef}
            id="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isConnecting}
            placeholder={
              readyDocs.length > 0
                ? 'Ask anything about your documents… (Shift+Enter for newline)'
                : 'Ask a question or upload a document in the sidebar…'
            }
            rows={1}
            className={cn(
              'flex-1 resize-none bg-transparent text-sm text-neutral-100',
              'placeholder:text-neutral-500 focus:outline-none',
              'max-h-[200px] leading-relaxed disabled:opacity-40',
            )}
            aria-label="Chat message input"
          />

          {/* Stop / Send button */}
          {isStreaming ? (
            <button
              id="stop-streaming-btn"
              onClick={handleStop}
              className="flex-shrink-0 p-2.5 rounded-xl bg-red-500/20 border border-red-500/30 text-red-400 hover:bg-red-500/30 transition-colors"
              aria-label="Stop streaming"
            >
              <StopCircle className="w-4 h-4" />
            </button>
          ) : (
            <button
              id="send-message-btn"
              onClick={() => handleSend()}
              disabled={!canSend}
              className={cn(
                'flex-shrink-0 p-2.5 rounded-xl transition-all duration-150',
                canSend
                  ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/30 hover:shadow-blue-900/50 hover:scale-105'
                  : 'bg-neutral-800 text-neutral-600 cursor-not-allowed',
              )}
              aria-label="Send message"
            >
              {isConnecting ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          )}
        </div>

        <p className="text-center text-[10px] text-neutral-600 mt-2">
          Quantiva uses hybrid RAG + agentic reasoning. Answers may be imperfect.
        </p>
      </div>
    </div>
  );
}
