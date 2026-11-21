/**
 * frontend/src/lib/api-client.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Centralized Desktop-Aware API Client for Quantiva.
 *
 * Supports both:
 * 1. Tauri 2 Desktop Shell:
 *    Retrieves dynamic backend URL and IPC token via Tauri's
 *    `invoke("get_backend_connection_info")`.
 * 2. Browser Dev / Web Mode (`npm run dev`):
 *    Falls back to `process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"`
 *    and `"dev-local-token"`.
 *
 * Automatically injects the `X-Quantiva-Token` header on all HTTP requests
 * and SSE streams (`fetchEventSource` / `ReadableStream`).
 */

export interface BackendConnectionInfo {
  baseUrl: string;
  ipcToken: string;
}

export interface SSEMessageEvent {
  id?: string;
  event: string;
  data: string;
}

export interface FetchEventSourceOptions {
  headers?: Record<string, string>;
  method?: string;
  body?: BodyInit | null;
  signal?: AbortSignal;
  onOpen?: (response: Response) => void | Promise<void>;
  onMessage?: (event: SSEMessageEvent) => void;
  onError?: (error: unknown) => void;
  onClose?: () => void;
}

export interface DesktopEventSourceLike {
  onopen: ((event: Event) => void) | null;
  onmessage: ((event: MessageEvent) => void) | null;
  onerror: ((event: Event | MessageEvent) => void) | null;
  close: () => void;
}

let cachedConnectionInfo: BackendConnectionInfo | null = null;
let connectionPromise: Promise<BackendConnectionInfo> | null = null;

/**
 * Returns true if running inside a Tauri 2 desktop webview.
 */
export function isTauriDesktop(): boolean {
  return typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;
}

/**
 * Safely invokes a Tauri IPC command via the internal Tauri runtime bridge.
 */
export async function invokeTauriCommand<T>(
  command: string,
  args: Record<string, unknown> = {},
): Promise<T> {
  if (typeof window !== 'undefined') {
    const internals = (
      window as unknown as {
        __TAURI_INTERNALS__?: {
          invoke: (cmd: string, args?: Record<string, unknown>) => Promise<T>;
        };
      }
    ).__TAURI_INTERNALS__;
    if (internals && typeof internals.invoke === 'function') {
      return await internals.invoke(command, args);
    }

    const tauriGlobal = (
      window as unknown as {
        __TAURI__?: {
          core?: {
            invoke: (cmd: string, args?: Record<string, unknown>) => Promise<T>;
          };
        };
      }
    ).__TAURI__;
    if (tauriGlobal?.core && typeof tauriGlobal.core.invoke === 'function') {
      return await tauriGlobal.core.invoke(command, args);
    }
  }
  throw new Error(`Tauri invoke is unavailable in the current environment.`);
}

/**
 * Retrieves backend connection info (baseUrl and IPC authentication token).
 * In Tauri desktop mode: queries Rust IPC.
 * In dev / web mode: falls back to NEXT_PUBLIC_API_URL and dev-local-token.
 */
export async function getBackendConnectionInfo(): Promise<BackendConnectionInfo> {
  if (cachedConnectionInfo) {
    return cachedConnectionInfo;
  }
  if (connectionPromise) {
    return connectionPromise;
  }

  connectionPromise = (async () => {
    if (isTauriDesktop()) {
      try {
        const info = await invokeTauriCommand<BackendConnectionInfo>(
          'get_backend_connection_info',
        );
        cachedConnectionInfo = {
          baseUrl: info.baseUrl.replace(/\/+$/, ''),
          ipcToken: info.ipcToken,
        };
        return cachedConnectionInfo;
      } catch (err) {
        console.warn(
          'Failed to retrieve backend connection info from Tauri shell. Falling back to dev defaults.',
          err,
        );
      }
    }

    const defaultUrl =
      process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
    cachedConnectionInfo = {
      baseUrl: defaultUrl.replace(/\/+$/, ''),
      ipcToken: process.env.NEXT_PUBLIC_DESKTOP_TOKEN || 'dev-local-token',
    };
    return cachedConnectionInfo;
  })();

  try {
    return await connectionPromise;
  } finally {
    connectionPromise = null;
  }
}

/**
 * Resets cached connection info (useful if backend restarts or port changes).
 */
export function resetConnectionInfoCache(): void {
  cachedConnectionInfo = null;
  connectionPromise = null;
}

/**
 * Resolves a full backend URL from a relative or absolute path.
 */
export function resolveBackendUrl(path: string, baseUrl: string): string {
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }
  const cleanBase = baseUrl.replace(/\/+$/, '');
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  if (cleanPath.startsWith('/api/v1')) {
    return `${cleanBase}${cleanPath}`;
  }
  return `${cleanBase}/api/v1${cleanPath}`;
}

/**
 * Desktop-aware fetch wrapper that automatically injects X-Quantiva-Token
 * and points to the correct backend host/port.
 */
export async function desktopFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const connection = await getBackendConnectionInfo();
  const url = resolveBackendUrl(path, connection.baseUrl);

  const headers = new Headers(init.headers || {});
  if (!headers.has('X-Quantiva-Token') && connection.ipcToken) {
    headers.set('X-Quantiva-Token', connection.ipcToken);
  }
  if (!headers.has('Accept')) {
    headers.set('Accept', 'application/json');
  }

  return await fetch(url, {
    ...init,
    headers,
  });
}

/**
 * Streams Server-Sent Events (SSE) over HTTP fetch with X-Quantiva-Token injection.
 * Works natively in both Tauri webviews and browsers where standard EventSource
 * cannot attach custom authentication headers.
 */
export async function fetchEventSource(
  path: string,
  options: FetchEventSourceOptions,
): Promise<() => void> {
  const controller = new AbortController();
  const signal = options.signal
    ? AbortSignal.any([controller.signal, options.signal])
    : controller.signal;

  (async () => {
    try {
      const connection = await getBackendConnectionInfo();
      const url = resolveBackendUrl(path, connection.baseUrl);

      const headers = new Headers(options.headers || {});
      headers.set('Accept', 'text/event-stream');
      if (!headers.has('X-Quantiva-Token') && connection.ipcToken) {
        headers.set('X-Quantiva-Token', connection.ipcToken);
      }

      const response = await fetch(url, {
        method: options.method || 'GET',
        headers,
        body: options.body,
        signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `SSE request failed with status ${response.status}: ${errorText}`,
        );
      }

      if (options.onOpen) {
        await options.onOpen(response);
      }

      if (!response.body) {
        throw new Error('ReadableStream body not supported or null in response.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let eventType = 'message';
      let eventId: string | undefined = undefined;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.trim()) {
            eventType = 'message';
            eventId = undefined;
            continue;
          }

          if (line.startsWith(':')) {
            // SSE comment / ping line
            continue;
          }

          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim();
            continue;
          }

          if (line.startsWith('id:')) {
            eventId = line.slice(3).trim();
            continue;
          }

          if (line.startsWith('data:')) {
            const data = line.slice(5).trim();
            if (options.onMessage) {
              options.onMessage({
                id: eventId,
                event: eventType,
                data,
              });
            }
          }
        }
      }

      if (options.onClose) {
        options.onClose();
      }
    } catch (err) {
      if (!signal.aborted && options.onError) {
        options.onError(err);
      }
    }
  })();

  return () => {
    controller.abort();
  };
}

/**
 * Creates an EventSource-compatible object powered by fetchEventSource
 * that attaches the required `X-Quantiva-Token` header for desktop streaming.
 */
export function createDesktopSSE(
  path: string,
  options: {
    headers?: Record<string, string>;
    method?: string;
    body?: BodyInit | null;
  } = {},
): DesktopEventSourceLike {
  let isClosed = false;
  let abortStream: (() => void) | null = null;

  const instance: DesktopEventSourceLike = {
    onopen: null,
    onmessage: null,
    onerror: null,
    close: () => {
      isClosed = true;
      if (abortStream) {
        abortStream();
        abortStream = null;
      }
    },
  };

  void fetchEventSource(path, {
    headers: options.headers,
    method: options.method,
    body: options.body,
    onOpen: () => {
      if (isClosed) return;
      if (instance.onopen) {
        instance.onopen(new Event('open'));
      }
    },
    onMessage: (evt) => {
      if (isClosed) return;
      if (instance.onmessage) {
        instance.onmessage(
          new MessageEvent(evt.event, {
            data: evt.data,
            lastEventId: evt.id,
          }),
        );
      }
    },
    onError: (err) => {
      if (isClosed) return;
      if (instance.onerror) {
        const event = new MessageEvent('error', {
          data: JSON.stringify({ error: String(err) }),
        });
        instance.onerror(event);
      }
    },
    onClose: () => {
      isClosed = true;
    },
  }).then((closer) => {
    if (isClosed) {
      closer();
    } else {
      abortStream = closer;
    }
  });

  return instance;
}

/**
 * High-level JSON helper methods.
 */
export const desktopApiClient = {
  fetch: desktopFetch,
  fetchEventSource,
  createDesktopSSE,

  async get<T>(path: string, options?: RequestInit): Promise<T> {
    const res = await desktopFetch(path, { ...options, method: 'GET' });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`GET ${path} failed (${res.status}): ${err}`);
    }
    return (await res.json()) as T;
  },

  async post<T>(
    path: string,
    data?: unknown,
    options?: RequestInit,
  ): Promise<T> {
    const headers = new Headers(options?.headers || {});
    let body: BodyInit | undefined = undefined;

    if (data instanceof FormData) {
      body = data;
    } else if (data !== undefined) {
      if (!headers.has('Content-Type')) {
        headers.set('Content-Type', 'application/json');
      }
      body = JSON.stringify(data);
    }

    const res = await desktopFetch(path, {
      ...options,
      method: 'POST',
      headers,
      body,
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`POST ${path} failed (${res.status}): ${err}`);
    }
    return (await res.json()) as T;
  },

  async put<T>(path: string, data?: unknown, options?: RequestInit): Promise<T> {
    const headers = new Headers(options?.headers || {});
    let body: BodyInit | undefined = undefined;

    if (data instanceof FormData) {
      body = data;
    } else if (data !== undefined) {
      if (!headers.has('Content-Type')) {
        headers.set('Content-Type', 'application/json');
      }
      body = JSON.stringify(data);
    }

    const res = await desktopFetch(path, {
      ...options,
      method: 'PUT',
      headers,
      body,
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`PUT ${path} failed (${res.status}): ${err}`);
    }
    return (await res.json()) as T;
  },

  async delete<T>(path: string, options?: RequestInit): Promise<T> {
    const res = await desktopFetch(path, { ...options, method: 'DELETE' });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`DELETE ${path} failed (${res.status}): ${err}`);
    }
    if (res.status === 204) {
      return null as T;
    }
    return (await res.json()) as T;
  },
};
