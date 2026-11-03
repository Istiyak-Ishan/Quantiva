import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

// ─── TYPES ───────────────────────────────────────────────────────────────────

export type DocumentStatus = 'pending' | 'uploading' | 'processing' | 'ready' | 'failed';

export interface Document {
  id: string;
  filename: string;
  status: DocumentStatus;
  uploadProgress: number;       // 0–100 (upload byte progress via XHR)
  chunkCount?: number | null;   // populated once indexing finishes
  addedAt: number;              // unix timestamp ms
}

// ─── STATE INTERFACE ──────────────────────────────────────────────────────────

interface DocumentState {
  documents: Document[];

  /** Add a new document (skips duplicates by id). */
  addDocument: (doc: Omit<Document, 'addedAt'>) => void;

  /** Update status and/or upload progress for a specific document. */
  updateStatus: (
    id: string,
    status: DocumentStatus,
    progress?: number,
    chunkCount?: number | null,
  ) => void;

  /** Remove a document from local state (does NOT call the API). */
  removeDocument: (id: string) => void;

  /** Replace a temp id (from optimistic creation) with the real server id. */
  replaceId: (tempId: string, realId: string) => void;

  /** Derive selected doc ids for session creation (only 'ready' docs). */
  readyDocIds: () => string[];
}

// ─── STORE ───────────────────────────────────────────────────────────────────

export const useDocumentStore = create<DocumentState>()(
  devtools(
    (set, get) => ({
      documents: [],

      addDocument: (doc) =>
        set(
          (state) => {
            if (state.documents.find((d) => d.id === doc.id)) return state;
            return {
              documents: [
                ...state.documents,
                { ...doc, addedAt: Date.now() },
              ],
            };
          },
          false,
          'addDocument',
        ),

      updateStatus: (id, status, progress, chunkCount) =>
        set(
          (state) => ({
            documents: state.documents.map((doc) =>
              doc.id === id
                ? {
                    ...doc,
                    status,
                    uploadProgress: progress ?? doc.uploadProgress,
                    chunkCount: chunkCount !== undefined ? chunkCount : doc.chunkCount,
                  }
                : doc,
            ),
          }),
          false,
          'updateStatus',
        ),

      removeDocument: (id) =>
        set(
          (state) => ({
            documents: state.documents.filter((doc) => doc.id !== id),
          }),
          false,
          'removeDocument',
        ),

      replaceId: (tempId, realId) =>
        set(
          (state) => ({
            documents: state.documents.map((doc) =>
              doc.id === tempId ? { ...doc, id: realId } : doc,
            ),
          }),
          false,
          'replaceId',
        ),

      readyDocIds: () =>
        get()
          .documents.filter((d) => d.status === 'ready')
          .map((d) => d.id),
    }),
    { name: 'QuantivaDocumentStore' },
  ),
);
