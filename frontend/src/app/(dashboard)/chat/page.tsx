"use client";

/**
 * frontend/src/app/(dashboard)/chat/page.tsx
 * ─────────────────────────────────────────────────────────────────────────────
 * Quantiva AI Chat Workspace (Prompt P7.4).
 *
 * Features:
 *  - Top Bar:
 *      • Knowledge Base selector (locks context to chosen KB)
 *      • LLM Model override selector (per-conversation switching)
 *      • Standard RAG vs. Agentic Reasoning (ReAct + Data Analyst) toggle
 *      • Evidence Drawer toggle button
 *  - Conversation History Sidebar:
 *      • List conversations for active KB, create new, delete
 *  - Streaming Message Renderer:
 *      • Real-time StreamingStageIndicator (Searching, Reranking, Python Analysis, Generating)
 *      • Collapsible AgentReasoningTrace (thoughts, sandboxed actions, observations)
 *      • Interactive citation pills ([1], [2]) that highlight and scroll to chunks
 *  - Right-Hand EvidencePanel:
 *      • Rerank Score, RRF Score, Dense Rank, Sparse Rank, Source File, Page Number
 *      • Verbatim quote highlighting and auto-scroll
 */

import React, { Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Sparkles,
  Send,
  StopCircle,
  Plus,
  Trash2,
  MessageSquare,
  Layers,
  Bot,
  Brain,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Database,
  Cpu,
  Clock,
  ExternalLink,
  ChevronDown,
} from 'lucide-react';
import {
  getKnowledgeBases,
  getLLMConfigs,
  listConversations,
  createConversation,
  getConversationMessages,
  deleteConversation,
  streamConversationAsk,
  KnowledgeBaseItem,
  LLMConfig,
  ConversationItem,
  ConversationMessage,
  RetrievedChunk,
  StructuredCitation,
} from '@/lib/api';
import { EvidencePanel } from '@/components/chat/EvidencePanel';
import { StreamingStageIndicator } from '@/components/chat/StreamingStageIndicator';
import { AgentReasoningTrace } from '@/components/chat/AgentReasoningTrace';

// ─── MARKDOWN ANSWER RENDERER WITH INTERACTIVE CITATION PILLS ─────────────────

interface MarkdownAnswerProps {
  content: string;
  onCitationClick: (index: number) => void;
  isStreaming?: boolean;
}

function MarkdownAnswer({
  content,
  onCitationClick,
  isStreaming = false,
}: MarkdownAnswerProps) {
  // Transform inline markers like [1] or [2] into markdown links [#cite-1](#cite-1)
  // so ReactMarkdown renders an <a> tag that we intercept.
  const formattedContent = content.replace(
    /\[(\d+)\]/g,
    '[[cite-$1]](#cite-$1)'
  );

  return (
    <div className="text-[13px] md:text-[14px] leading-relaxed break-words text-neutral-200 select-text [&_p]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ul]:my-2 [&_ol]:list-decimal [&_ol]:pl-5 [&_ol]:my-2 [&_pre]:overflow-x-auto [&_pre]:bg-neutral-950/80 [&_pre]:p-3 [&_pre]:rounded-lg [&_pre]:border [&_pre]:border-white/10 [&_pre]:my-2 [&_table]:block [&_table]:overflow-x-auto [&_td]:border [&_td]:border-white/10 [&_td]:p-2 [&_th]:border [&_th]:border-white/10 [&_th]:p-2 [&_th]:bg-white/5 [&_h1]:text-lg [&_h1]:font-bold [&_h2]:text-base [&_h2]:font-semibold [&_h3]:text-sm [&_h3]:font-semibold">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith('#cite-')) {
              const citIndex = Number(href.replace('#cite-', ''));
              return (
                <button
                  type="button"
                  onClick={() => onCitationClick(citIndex)}
                  className="inline-flex items-center justify-center px-1.5 py-0.5 mx-0.5 rounded text-[11px] font-mono font-bold bg-blue-500/20 text-blue-400 hover:bg-blue-500/35 hover:text-blue-300 border border-blue-500/30 transition-all cursor-pointer align-baseline"
                  title={`View supporting citation [${citIndex}] in Evidence Drawer`}
                >
                  [{citIndex}]
                </button>
              );
            }
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-400 hover:underline"
              >
                {children}
              </a>
            );
          },
        }}
      >
        {formattedContent}
      </ReactMarkdown>
      {isStreaming && (
        <span
          className="inline-block w-1.5 h-4 ml-1 bg-blue-400 animate-pulse align-middle"
          aria-hidden="true"
        />
      )}
    </div>
  );
}

// ─── CHAT WORKSPACE INNER COMPONENT ───────────────────────────────────────────

function ChatWorkspace() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const urlConvId = searchParams.get('convId') || searchParams.get('sessionId') || '';
  const urlKbId = searchParams.get('kbId') || '';

  // Data selections
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBaseItem[]>([]);
  const [selectedKbId, setSelectedKbId] = useState<string>(urlKbId);
  const [llmConfigs, setLlmConfigs] = useState<LLMConfig[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [mode, setMode] = useState<'direct' | 'agentic'>('agentic');

  // Sidebar & conversations
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [conversations, setConversations] = useState<ConversationItem[]>([]);
  const [activeConvId, setActiveConvId] = useState<string>(urlConvId);
  const [loadingConversations, setLoadingConversations] = useState(true);

  // Message history & streaming
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [loadingMessages, setLoadingMessages] = useState(false);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStage, setCurrentStage] = useState<string | null>(null);
  const [streamingAnswer, setStreamingAnswer] = useState<string>('');
  const [streamingReasoning, setStreamingReasoning] = useState<string>('');
  const [streamingSources, setStreamingSources] = useState<RetrievedChunk[]>([]);
  const [streamingCitations, setStreamingCitations] = useState<StructuredCitation[]>([]);

  // Evidence panel state
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [evidenceChunks, setEvidenceChunks] = useState<RetrievedChunk[]>([]);
  const [evidenceCitations, setEvidenceCitations] = useState<StructuredCitation[]>([]);
  const [activeCitationIndex, setActiveCitationIndex] = useState<number | null>(null);

  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  const abortStreamRef = useRef<(() => void) | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll on message updates
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingAnswer, streamingReasoning, currentStage]);

  // Auto-resize input textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 180)}px`;
  }, [input]);

  // Initial load of KBs and LLMs
  useEffect(() => {
    async function loadInitialData() {
      try {
        const [kbs, llms] = await Promise.all([getKnowledgeBases(), getLLMConfigs()]);
        setKnowledgeBases(kbs);
        setLlmConfigs(llms);

        // Set default KB if not in URL
        if (!urlKbId && kbs.length > 0) {
          setSelectedKbId(kbs[0].id);
        }

        // Set default LLM
        const defaultLlm = llms.find((m) => m.is_default) || llms[0];
        if (defaultLlm) {
          setSelectedModelId(defaultLlm.id);
        }
      } catch (err) {
        console.error('Failed to load knowledge bases or models:', err);
      }
    }
    void loadInitialData();
  }, [urlKbId]);

  // Load conversations when selectedKbId changes
  useEffect(() => {
    if (!selectedKbId) return;

    let active = true;
    setLoadingConversations(true);

    listConversations(selectedKbId)
      .then((convs) => {
        if (!active) return;
        setConversations(convs);
        if (convs.length > 0) {
          // If current activeConvId is not in this KB's conversations, switch to first
          const found = convs.some((c) => c.id === activeConvId);
          if (!found) {
            setActiveConvId(convs[0].id);
          }
        } else {
          setActiveConvId('');
          setMessages([]);
        }
      })
      .catch((err) => {
        console.error('Error fetching conversations:', err);
      })
      .finally(() => {
        if (active) setLoadingConversations(false);
      });

    return () => {
      active = false;
    };
  }, [selectedKbId]);

  // Load messages when activeConvId changes
  useEffect(() => {
    if (!activeConvId) {
      setMessages([]);
      return;
    }

    let active = true;
    setLoadingMessages(true);

    getConversationMessages(activeConvId)
      .then((msgs) => {
        if (!active) return;
        setMessages(msgs);

        // If the conversation already has an assistant message with citations, prime the evidence panel
        const lastAsst = [...msgs].reverse().find((m) => m.role === 'assistant');
        if (lastAsst?.retrieved_chunks && lastAsst.retrieved_chunks.length > 0) {
          setEvidenceChunks(lastAsst.retrieved_chunks);
          setEvidenceCitations(lastAsst.citations || []);
        }
      })
      .catch((err) => {
        console.error('Failed to load conversation messages:', err);
      })
      .finally(() => {
        if (active) setLoadingMessages(false);
      });

    return () => {
      active = false;
    };
  }, [activeConvId]);

  // Create new conversation
  const handleCreateNewConversation = async () => {
    if (!selectedKbId) return;
    try {
      const newConv = await createConversation({
        knowledge_base_id: selectedKbId,
        title: 'New Conversation',
        model_config_id: selectedModelId || undefined,
      });
      setConversations((prev) => [newConv, ...prev]);
      setActiveConvId(newConv.id);
      setMessages([]);
      setEvidenceChunks([]);
      setEvidenceCitations([]);
      setActiveCitationIndex(null);
    } catch (err) {
      setErrorBanner('Failed to create a new conversation. Please verify backend status.');
    }
  };

  // Delete conversation
  const handleDeleteConversation = async (convId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('Delete this conversation and its message history?')) return;
    try {
      await deleteConversation(convId);
      const remaining = conversations.filter((c) => c.id !== convId);
      setConversations(remaining);
      if (activeConvId === convId) {
        if (remaining.length > 0) {
          setActiveConvId(remaining[0].id);
        } else {
          setActiveConvId('');
          setMessages([]);
        }
      }
    } catch (err) {
      setErrorBanner('Failed to delete conversation.');
    }
  };

  // Open evidence drawer for a specific citation
  const handleCitationClick = useCallback(
    (
      citationIndex: number,
      chunksOverride?: RetrievedChunk[],
      citationsOverride?: StructuredCitation[]
    ) => {
      if (chunksOverride) setEvidenceChunks(chunksOverride);
      if (citationsOverride) setEvidenceCitations(citationsOverride);
      setActiveCitationIndex(citationIndex);
      setEvidenceOpen(true);
    },
    []
  );

  // Stop active stream
  const handleStopStream = () => {
    if (abortStreamRef.current) {
      abortStreamRef.current();
      abortStreamRef.current = null;
    }
    setIsStreaming(false);
    setCurrentStage(null);
  };

  // Send question
  const handleSend = async (overridePrompt?: string) => {
    const queryText = (overridePrompt ?? input).trim();
    if (!queryText || isStreaming) return;

    if (!selectedKbId) {
      setErrorBanner('Please select a Knowledge Base before sending queries.');
      return;
    }

    let targetConvId = activeConvId;

    // Create conversation on the fly if none active
    if (!targetConvId) {
      try {
        const kb = knowledgeBases.find((k) => k.id === selectedKbId);
        const title = queryText.length > 30 ? `${queryText.slice(0, 30)}…` : queryText;
        const newConv = await createConversation({
          knowledge_base_id: selectedKbId,
          title,
          model_config_id: selectedModelId || undefined,
        });
        setConversations((prev) => [newConv, ...prev]);
        setActiveConvId(newConv.id);
        targetConvId = newConv.id;
      } catch (err) {
        setErrorBanner('Failed to initialize conversation for query.');
        return;
      }
    }

    setInput('');
    setErrorBanner(null);
    setIsStreaming(true);
    setCurrentStage('Searching Dense & FTS5 Indexes…');
    setStreamingAnswer('');
    setStreamingReasoning('');
    setStreamingSources([]);
    setStreamingCitations([]);

    // Optimistically insert user turn
    const optimisticUserTurn: ConversationMessage = {
      id: `temp-user-${Date.now()}`,
      conversation_id: targetConvId,
      role: 'user',
      content: queryText,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimisticUserTurn]);

    let accumulatedAnswer = '';
    let accumulatedReasoning = '';
    let capturedSources: RetrievedChunk[] = [];
    let capturedCitations: StructuredCitation[] = [];

    const abortFn = streamConversationAsk(
      targetConvId,
      {
        query: queryText,
        mode,
        model_config_id: selectedModelId || undefined,
      },
      {
        onStatus: (stage) => {
          setCurrentStage(stage);
        },
        onSources: (sources) => {
          capturedSources = sources;
          setStreamingSources(sources);
          setEvidenceChunks(sources);
        },
        onCitations: (citations) => {
          capturedCitations = citations;
          setStreamingCitations(citations);
          setEvidenceCitations(citations);
        },
        onTokenDelta: (token) => {
          accumulatedAnswer += token;
          setStreamingAnswer(accumulatedAnswer);
        },
        onReasoningDelta: (reasoningToken) => {
          accumulatedReasoning += reasoningToken;
          setStreamingReasoning(accumulatedReasoning);
        },
        onDone: (data) => {
          setIsStreaming(false);
          setCurrentStage(null);

          const finalAsstTurn: ConversationMessage = {
            id: `asst-${Date.now()}`,
            conversation_id: targetConvId,
            role: 'assistant',
            content: accumulatedAnswer,
            citations: capturedCitations,
            retrieved_chunks: capturedSources,
            reasoning_trace: accumulatedReasoning
              ? { reasoning: accumulatedReasoning }
              : null,
            latency_ms: data.latency_ms,
            created_at: new Date().toISOString(),
          };

          setMessages((prev) => [...prev, finalAsstTurn]);
          setStreamingAnswer('');
          setStreamingReasoning('');
          setStreamingSources([]);
          setStreamingCitations([]);
          abortStreamRef.current = null;
        },
        onError: (err) => {
          console.error('SSE Error:', err);
          setIsStreaming(false);
          setCurrentStage(null);
          setErrorBanner('Stream interrupted or backend error occurred.');
          abortStreamRef.current = null;
        },
      }
    );

    abortStreamRef.current = abortFn;
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  };

  const selectedKb = knowledgeBases.find((k) => k.id === selectedKbId);

  return (
    <div className="flex h-full w-full bg-[#0a0a0f] text-neutral-100 overflow-hidden select-none">
      {/* ── LEFT SIDEBAR: CONVERSATION HISTORY ─────────────────────────────── */}
      <aside
        className={`flex flex-col border-r border-white/10 bg-[#0c0c12] transition-all duration-300 ${
          sidebarCollapsed ? 'w-14 items-center' : 'w-64 md:w-72'
        } shrink-0`}
      >
        {/* Sidebar Header */}
        <div className="flex items-center justify-between p-3.5 border-b border-white/10">
          {!sidebarCollapsed && (
            <div className="flex items-center gap-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 shadow-md">
                <Sparkles className="h-4 w-4 text-white" />
              </div>
              <span className="text-xs font-bold text-white tracking-tight">Conversations</span>
            </div>
          )}
          <button
            type="button"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="rounded-lg p-1 text-neutral-400 hover:bg-white/5 hover:text-white transition-colors"
            title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {sidebarCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-2.5 border-b border-white/5">
          <button
            type="button"
            onClick={handleCreateNewConversation}
            className={`w-full flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-3 py-2 text-xs font-semibold text-white shadow-md shadow-blue-600/20 hover:bg-blue-500 transition-colors ${
              sidebarCollapsed ? 'p-2' : ''
            }`}
            title="Start new conversation"
          >
            <Plus className="h-4 w-4" />
            {!sidebarCollapsed && <span>New Chat</span>}
          </button>
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {loadingConversations ? (
            <div className="flex items-center justify-center p-4 text-neutral-500">
              <Loader2 className="h-4 w-4 animate-spin" />
            </div>
          ) : conversations.length === 0 ? (
            !sidebarCollapsed && (
              <p className="p-4 text-center text-xs text-neutral-500">
                No conversations yet. Click + New Chat to begin.
              </p>
            )
          ) : (
            conversations.map((conv) => {
              const isActive = conv.id === activeConvId;
              return (
                <div
                  key={conv.id}
                  onClick={() => setActiveConvId(conv.id)}
                  className={`group flex items-center justify-between rounded-xl px-2.5 py-2 text-xs cursor-pointer transition-all ${
                    isActive
                      ? 'bg-blue-500/15 border border-blue-500/30 text-white'
                      : 'border border-transparent text-neutral-400 hover:bg-white/5 hover:text-neutral-200'
                  }`}
                  title={conv.title}
                >
                  <div className="flex items-center gap-2 truncate min-w-0">
                    <MessageSquare
                      className={`h-3.5 w-3.5 shrink-0 ${
                        isActive ? 'text-blue-400' : 'text-neutral-500'
                      }`}
                    />
                    {!sidebarCollapsed && (
                      <span className="truncate font-medium">{conv.title}</span>
                    )}
                  </div>

                  {!sidebarCollapsed && (
                    <button
                      type="button"
                      onClick={(e) => handleDeleteConversation(conv.id, e)}
                      className="opacity-0 group-hover:opacity-100 p-1 text-neutral-500 hover:text-red-400 transition-opacity"
                      title="Delete conversation"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  )}
                </div>
              );
            })
          )}
        </div>
      </aside>

      {/* ── CENTER WORKSPACE ──────────────────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0 h-full relative">
        {/* ── TOP CONTROL BAR ─────────────────────────────────────────────── */}
        <header className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 border-b border-white/10 bg-[#0e0e14] shrink-0 z-10">
          <div className="flex flex-wrap items-center gap-2.5">
            {/* Knowledge Base Selector */}
            <div className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-black/40 px-2.5 py-1 text-xs">
              <Database className="h-3.5 w-3.5 text-blue-400 shrink-0" />
              <label htmlFor="kb-selector" className="text-[10px] uppercase font-bold text-neutral-500">
                KB:
              </label>
              <select
                id="kb-selector"
                value={selectedKbId}
                onChange={(e) => setSelectedKbId(e.target.value)}
                className="bg-transparent text-white font-medium text-xs focus:outline-none cursor-pointer max-w-[150px] truncate"
              >
                {knowledgeBases.map((kb) => (
                  <option key={kb.id} value={kb.id} className="bg-neutral-900 text-white">
                    {kb.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Model Selector Override */}
            <div className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-black/40 px-2.5 py-1 text-xs">
              <Cpu className="h-3.5 w-3.5 text-teal-400 shrink-0" />
              <label htmlFor="model-selector" className="text-[10px] uppercase font-bold text-neutral-500">
                Model:
              </label>
              <select
                id="model-selector"
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
                className="bg-transparent text-white font-medium text-xs focus:outline-none cursor-pointer max-w-[150px] truncate"
              >
                {llmConfigs.map((m) => (
                  <option key={m.id} value={m.id} className="bg-neutral-900 text-white">
                    {m.name} ({m.model_name})
                  </option>
                ))}
              </select>
            </div>

            {/* Mode Switcher Toggle */}
            <div className="inline-flex items-center rounded-lg border border-white/10 bg-black/40 p-0.5 text-xs">
              <button
                type="button"
                onClick={() => setMode('direct')}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-md transition-colors ${
                  mode === 'direct'
                    ? 'bg-blue-600 text-white font-semibold shadow-sm'
                    : 'text-neutral-400 hover:text-white'
                }`}
                title="Standard RAG: Single-pass hybrid search & direct synthesis"
              >
                <Layers className="h-3 w-3" />
                <span>Standard RAG</span>
              </button>

              <button
                type="button"
                onClick={() => setMode('agentic')}
                className={`flex items-center gap-1 px-2.5 py-1 rounded-md transition-colors ${
                  mode === 'agentic'
                    ? 'bg-purple-600 text-white font-semibold shadow-sm'
                    : 'text-neutral-400 hover:text-white'
                }`}
                title="Agentic Reasoning: Multi-step ReAct thought loop & sandboxed Python analysis"
              >
                <Bot className="h-3 w-3" />
                <span>Agentic Reasoning</span>
              </button>
            </div>
          </div>

          {/* Right Action: Evidence Drawer Toggle */}
          <button
            type="button"
            onClick={() => setEvidenceOpen(!evidenceOpen)}
            className={`inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors ${
              evidenceOpen
                ? 'border-blue-500 bg-blue-500/15 text-blue-300'
                : 'border-white/10 bg-black/40 text-neutral-300 hover:bg-white/5 hover:text-white'
            }`}
            title="Toggle Supporting Evidence & Citations Panel"
          >
            <Layers className="h-3.5 w-3.5" />
            <span>Evidence</span>
            {evidenceChunks.length > 0 && (
              <span className="font-mono text-[10px] rounded-full bg-blue-500/20 px-1.5 text-blue-400">
                {evidenceChunks.length}
              </span>
            )}
          </button>
        </header>

        {/* Error notification banner */}
        {errorBanner && (
          <div className="px-4 py-2 bg-red-500/10 border-b border-red-500/20 text-xs text-red-300 flex items-center justify-between">
            <span>{errorBanner}</span>
            <button
              type="button"
              onClick={() => setErrorBanner(null)}
              className="text-red-400 hover:text-white"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* ── MESSAGE LOG / STREAM VIEW ─────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto px-4 md:px-8 py-6 space-y-6">
          {loadingMessages ? (
            <div className="h-full flex items-center justify-center text-neutral-500 gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-xs">Loading message history…</span>
            </div>
          ) : messages.length === 0 && !isStreaming ? (
            /* Empty state */
            <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto space-y-4 py-12">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 border border-white/10 shadow-xl">
                <Sparkles className="h-7 w-7 text-blue-400" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white">
                  {selectedKb ? `Chat with ${selectedKb.name}` : 'Quantiva AI Assistant'}
                </h3>
                <p className="text-xs text-neutral-400 mt-1 leading-relaxed">
                  Ask questions across all indexed documents, tables, and notes. Responses include
                  interactive inline citations and verifiable evidence chunks.
                </p>
              </div>

              {/* Sample queries */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full pt-2">
                {[
                  'Summarize the core findings and action items',
                  'What are the key financial numbers reported?',
                  'Extract key dates, milestones, and deadlines',
                  'Analyze risk factors and discrepancies',
                ].map((promptText) => (
                  <button
                    key={promptText}
                    type="button"
                    onClick={() => void handleSend(promptText)}
                    className="p-2.5 rounded-xl border border-white/10 bg-white/[0.02] hover:bg-white/[0.06] text-left text-xs text-neutral-300 hover:text-white transition-all shadow-sm"
                  >
                    {promptText}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* Messages list */
            <>
              {messages.map((msg, index) => {
                const isUser = msg.role === 'user';
                return (
                  <div
                    key={msg.id || index}
                    className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}
                  >
                    {/* User Turn */}
                    {isUser ? (
                      <div className="max-w-[85%] md:max-w-[70%] rounded-2xl bg-blue-600 px-4 py-2.5 text-sm text-white shadow-lg shadow-blue-600/15">
                        <p className="whitespace-pre-wrap leading-relaxed select-text">
                          {msg.content}
                        </p>
                      </div>
                    ) : (
                      /* Assistant Turn */
                      <div className="w-full max-w-3xl rounded-2xl border border-white/10 bg-neutral-900/50 p-4 shadow-xl space-y-3">
                        {/* Agent Reasoning Trace Accordion */}
                        {(msg.reasoning_trace || msg.citations) && (
                          <AgentReasoningTrace
                            reasoning={
                              typeof msg.reasoning_trace?.reasoning === 'string'
                                ? msg.reasoning_trace.reasoning
                                : null
                            }
                            reasoningTrace={msg.reasoning_trace}
                          />
                        )}

                        {/* Markdown Content with interactive citation links */}
                        <MarkdownAnswer
                          content={msg.content}
                          onCitationClick={(citeIdx) =>
                            handleCitationClick(
                              citeIdx,
                              msg.retrieved_chunks,
                              msg.citations
                            )
                          }
                        />

                        {/* Structured Citations Footer Bar */}
                        {msg.citations && msg.citations.length > 0 && (
                          <div className="pt-2 border-t border-white/5 flex flex-wrap items-center gap-1.5">
                            <span className="text-[10px] uppercase font-bold text-neutral-500 mr-1">
                              Sources:
                            </span>
                            {msg.citations.map((c) => (
                              <button
                                key={c.marker}
                                type="button"
                                onClick={() =>
                                  handleCitationClick(
                                    c.index,
                                    msg.retrieved_chunks,
                                    msg.citations
                                  )
                                }
                                className="inline-flex items-center gap-1 rounded-md border border-white/10 bg-white/5 px-2 py-0.5 text-[11px] font-mono text-neutral-300 hover:bg-white/10 hover:text-white transition-colors"
                              >
                                <span className="font-bold text-blue-400">{c.marker}</span>
                                <span className="truncate max-w-[140px] font-sans">
                                  {c.source_file || 'Document'}
                                </span>
                                {c.page_number !== null && c.page_number !== undefined && (
                                  <span className="text-[10px] text-neutral-500">
                                    p.{c.page_number}
                                  </span>
                                )}
                              </button>
                            ))}
                          </div>
                        )}

                        {/* Meta stats */}
                        <div className="flex items-center gap-3 text-[10px] text-neutral-500 pt-1">
                          {msg.latency_ms !== undefined && (
                            <span className="flex items-center gap-1">
                              <Clock className="h-3 w-3" />
                              {(msg.latency_ms / 1000).toFixed(2)}s
                            </span>
                          )}
                          {msg.model_used && (
                            <span className="flex items-center gap-1 font-mono">
                              <Cpu className="h-3 w-3" />
                              {msg.model_used}
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Live Streaming Turn */}
              {isStreaming && (
                <div className="w-full max-w-3xl rounded-2xl border border-blue-500/20 bg-neutral-900/60 p-4 shadow-xl space-y-3 animate-in fade-in">
                  {/* Real-time Stage Pill */}
                  <div className="flex items-center justify-between">
                    <StreamingStageIndicator
                      currentStage={currentStage}
                      isStreaming={isStreaming}
                    />
                    <button
                      type="button"
                      onClick={handleStopStream}
                      className="inline-flex items-center gap-1 rounded-lg border border-red-500/20 bg-red-500/10 px-2 py-1 text-[11px] font-medium text-red-400 hover:bg-red-500/20 transition-colors"
                    >
                      <StopCircle className="h-3 w-3" />
                      <span>Stop</span>
                    </button>
                  </div>

                  {/* Streaming Reasoning Trace */}
                  {streamingReasoning && (
                    <AgentReasoningTrace
                      reasoning={streamingReasoning}
                      isStreaming={isStreaming}
                      defaultOpen={true}
                    />
                  )}

                  {/* Streaming Answer Text */}
                  {streamingAnswer && (
                    <MarkdownAnswer
                      content={streamingAnswer}
                      onCitationClick={(citeIdx) =>
                        handleCitationClick(
                          citeIdx,
                          streamingSources,
                          streamingCitations
                        )
                      }
                      isStreaming={isStreaming}
                    />
                  )}
                </div>
              )}

              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* ── BOTTOM INPUT BOX ─────────────────────────────────────────────── */}
        <div className="px-4 pb-4 pt-2 border-t border-white/10 bg-[#0e0e14]">
          <div className="max-w-3xl mx-auto rounded-2xl border border-white/10 bg-black/60 p-2.5 focus-within:border-blue-500/50 shadow-2xl transition-all">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isStreaming}
              placeholder={
                selectedKb
                  ? `Ask anything in ${selectedKb.name}… (Shift+Enter for newline)`
                  : 'Ask a question or select a knowledge base…'
              }
              className="w-full resize-none bg-transparent px-2 py-1 text-sm text-white placeholder:text-neutral-500 focus:outline-none leading-relaxed"
            />

            <div className="flex items-center justify-between pt-2 border-t border-white/5 px-1">
              <span className="text-[10px] text-neutral-500 font-mono">
                {mode === 'agentic' ? 'ReAct Multi-Hop Agent Active' : 'Direct Hybrid Search Active'}
              </span>

              {isStreaming ? (
                <button
                  type="button"
                  onClick={handleStopStream}
                  className="inline-flex items-center gap-1.5 rounded-xl bg-red-600 px-3 py-1.5 text-xs font-semibold text-white shadow-md shadow-red-600/30 hover:bg-red-500 transition-colors"
                >
                  <StopCircle className="h-3.5 w-3.5" />
                  <span>Stop</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => void handleSend()}
                  disabled={!input.trim()}
                  className="inline-flex items-center gap-1.5 rounded-xl bg-blue-600 px-3.5 py-1.5 text-xs font-semibold text-white shadow-md shadow-blue-600/30 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
                >
                  <Send className="h-3.5 w-3.5" />
                  <span>Send</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* ── RIGHT EVIDENCE DRAWER ──────────────────────────────────────────── */}
      <EvidencePanel
        isOpen={evidenceOpen}
        onClose={() => setEvidenceOpen(false)}
        chunks={evidenceChunks}
        citations={evidenceCitations}
        activeCitationIndex={activeCitationIndex}
        onSelectCitation={(idx) => setActiveCitationIndex(idx)}
      />
    </div>
  );
}

// ─── EXPORT WITH SUSPENSE (NEXT.JS STATIC EXPORT SAFE) ─────────────────────────

export default function ChatPage() {
  return (
    <Suspense
      fallback={
        <div className="h-full w-full bg-[#0a0a0f] flex items-center justify-center text-neutral-500">
          <div className="flex items-center gap-3">
            <Loader2 className="h-5 w-5 animate-spin" />
            <span className="text-sm">Initializing Quantiva AI Workspace…</span>
          </div>
        </div>
      }
    >
      <ChatWorkspace />
    </Suspense>
  );
}
