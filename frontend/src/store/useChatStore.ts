import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

// ─── TYPES ───────────────────────────────────────────────────────────────────

export interface Citation {
  doc_id: string;
  filename: string;
  page_or_row: number;
  text_excerpt: string;
}

export interface Message {
  id?: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  confidence?: number;
  chart_url?: string;
  timestamp?: number;
}

// ─── STATE INTERFACE ──────────────────────────────────────────────────────────

interface ChatState {
  // ── Persistent state ──
  messages: Message[];
  currentSession: string | null;

  // ── Streaming state ──
  isStreaming: boolean;
  streamingMessage: string;       // accumulates tokens in-flight
  streamingCitations: Citation[]; // populated on the final 'done' event

  // ── Actions ──
  setSession: (sessionId: string) => void;
  setMessages: (messages: Message[]) => void;
  addMessage: (message: Message) => void;
  clearMessages: () => void;

  /**
   * Called for each SSE token event.
   * First call flips isStreaming → true and seeds streamingMessage.
   * Subsequent calls concatenate to streamingMessage.
   */
  updateStreamingMessage: (chunk: string) => void;

  /**
   * Called on the SSE 'done' event.
   * Moves the accumulated streamingMessage into the messages array as a
   * finalized assistant message, then resets streaming state.
   */
  finalizeMessage: (
    citations?: Citation[],
    confidence?: number,
    chartUrl?: string,
  ) => void;

  /**
   * Hard-reset streaming state (e.g., on EventSource error).
   */
  abortStreaming: () => void;
}

// ─── STORE ───────────────────────────────────────────────────────────────────

export const useChatStore = create<ChatState>()(
  devtools(
    (set) => ({
      messages: [],
      currentSession: null,
      isStreaming: false,
      streamingMessage: '',
      streamingCitations: [],

      setSession: (sessionId) => set({ currentSession: sessionId }, false, 'setSession'),

      setMessages: (messages) => set({ messages }, false, 'setMessages'),

      addMessage: (message) =>
        set(
          (state) => ({
            messages: [
              ...state.messages,
              { ...message, timestamp: message.timestamp ?? Date.now() },
            ],
          }),
          false,
          'addMessage',
        ),

      clearMessages: () => set({ messages: [] }, false, 'clearMessages'),

      // ── Streaming ────────────────────────────────────────────────────────
      updateStreamingMessage: (chunk) =>
        set(
          (state) => {
            if (!state.isStreaming) {
              // First token — start the stream
              return { isStreaming: true, streamingMessage: chunk };
            }
            return { streamingMessage: state.streamingMessage + chunk };
          },
          false,
          'updateStreamingMessage',
        ),

      finalizeMessage: (citations = [], confidence, chartUrl) =>
        set(
          (state) => {
            const finalMsg: Message = {
              role: 'assistant',
              content: state.streamingMessage,
              citations,
              confidence,
              chart_url: chartUrl,
              timestamp: Date.now(),
            };
            return {
              messages: [...state.messages, finalMsg],
              isStreaming: false,
              streamingMessage: '',
              streamingCitations: [],
            };
          },
          false,
          'finalizeMessage',
        ),

      abortStreaming: () =>
        set(
          { isStreaming: false, streamingMessage: '', streamingCitations: [] },
          false,
          'abortStreaming',
        ),
    }),
    { name: 'QuantivaChatStore' },
  ),
);
