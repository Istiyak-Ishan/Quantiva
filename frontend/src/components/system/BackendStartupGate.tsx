"use client";

import React, { useEffect, useState, useCallback } from 'react';
import {
  getBackendConnectionInfo,
  isTauriDesktop,
  invokeTauriCommand,
} from '@/lib/api-client';
import { getSystemHealth, SystemHealth } from '@/lib/api';
import {
  ShieldAlert,
  FolderOpen,
  RotateCw,
  Database,
  Layers,
  Sparkles,
  CheckCircle2,
  Loader2,
  HardDrive,
  Cpu,
} from 'lucide-react';

interface BackendStartupGateProps {
  children: React.ReactNode;
}

export function BackendStartupGate({ children }: BackendStartupGateProps) {
  const [status, setStatus] = useState<'initializing' | 'healthy' | 'error'>('initializing');
  const [message, setMessage] = useState<string>('Connecting to Quantiva backend engine…');
  const [stage, setStage] = useState<string>('init');
  const [elapsed, setElapsed] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [healthData, setHealthData] = useState<SystemHealth | null>(null);
  const [isRetrying, setIsRetrying] = useState<boolean>(false);

  const checkHealth = useCallback(async (): Promise<boolean> => {
    try {
      const info = await getBackendConnectionInfo();
      if (!info.baseUrl) return false;

      const health = await getSystemHealth();
      setHealthData(health);

      if (health.status === 'ok' || health.sqlite_status === 'connected') {
        setStatus('healthy');
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }, []);

  const startPolling = useCallback(() => {
    setStatus('initializing');
    setErrorMessage(null);
    setElapsed(0);

    let active = true;
    let timer: NodeJS.Timeout | null = null;
    let secondsTimer: NodeJS.Timeout | null = null;
    let currentSeconds = 0;

    secondsTimer = setInterval(() => {
      if (!active) return;
      currentSeconds += 1;
      setElapsed(currentSeconds);

      if (currentSeconds < 2) {
        setStage('init');
        setMessage('Connecting to Quantiva desktop backend…');
      } else if (currentSeconds < 5) {
        setStage('database');
        setMessage('Initializing SQLite database & WAL journal…');
      } else if (currentSeconds < 10) {
        setStage('storage');
        setMessage('Mounting local vector store & document storage…');
      } else if (currentSeconds < 25) {
        setStage('providers');
        setMessage('Detecting local hardware & AI model providers…');
      } else if (currentSeconds >= 30) {
        if (active) {
          setStatus('error');
          setErrorMessage(
            'Backend health check timed out after 30 seconds. The backend process may have encountered an issue starting up.',
          );
        }
      }
    }, 1000);

    const poll = async () => {
      if (!active) return;
      const isHealthy = await checkHealth();
      if (isHealthy) {
        if (secondsTimer) clearInterval(secondsTimer);
        return;
      }
      if (active && currentSeconds < 30) {
        timer = setTimeout(poll, 400);
      }
    };

    void poll();

    return () => {
      active = false;
      if (timer) clearTimeout(timer);
      if (secondsTimer) clearInterval(secondsTimer);
    };
  }, [checkHealth]);

  useEffect(() => {
    const cleanup = startPolling();
    return cleanup;
  }, [startPolling]);

  const handleRetry = async () => {
    setIsRetrying(true);
    setErrorMessage(null);
    try {
      if (isTauriDesktop()) {
        await invokeTauriCommand('restart_backend');
      }
      startPolling();
    } catch (err) {
      setErrorMessage(
        err instanceof Error
          ? err.message
          : 'Failed to restart backend process. Please check logs.',
      );
    } finally {
      setIsRetrying(false);
    }
  };

  const handleOpenLogs = async () => {
    try {
      if (isTauriDesktop()) {
        await invokeTauriCommand('open_data_folder', { subfolder: 'logs' });
      } else {
        window.alert('Logs are stored at: %LOCALAPPDATA%\\Quantiva\\logs\\backend.log');
      }
    } catch (err) {
      console.error('Failed to open logs folder:', err);
    }
  };

  if (status === 'healthy') {
    return <>{children}</>;
  }

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-[#070709] px-6 select-none">
      {/* Background radial gradient glow */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-blue-600/10 blur-[140px]" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[350px] h-[350px] rounded-full bg-teal-500/10 blur-[100px]" />
      </div>

      <div className="relative w-full max-w-lg rounded-2xl border border-white/10 bg-[#0f0f13]/90 p-8 shadow-2xl backdrop-blur-xl">
        {/* Header / Logo */}
        <div className="flex flex-col items-center text-center">
          <div className="relative mb-5 flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-tr from-blue-600 to-teal-400 p-[1px] shadow-lg shadow-blue-500/20">
            <div className="flex h-full w-full items-center justify-center rounded-2xl bg-[#0c0c10]">
              <Sparkles className="h-8 w-8 text-blue-400 animate-pulse" />
            </div>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Quantiva</h1>
          <p className="mt-1 text-xs font-medium uppercase tracking-wider text-neutral-400">
            Private AI Document Intelligence
          </p>
        </div>

        {/* Dynamic State: Initializing vs Error */}
        {status === 'initializing' ? (
          <div className="mt-8 flex flex-col space-y-6">
            <div className="flex flex-col space-y-2">
              <div className="flex items-center justify-between text-xs font-medium text-neutral-400">
                <span className="flex items-center gap-2">
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-400" />
                  {message}
                </span>
                <span className="tabular-nums font-mono text-neutral-500">{elapsed}s / 30s</span>
              </div>

              {/* Progress Bar */}
              <div className="h-2 w-full overflow-hidden rounded-full bg-white/5 p-0.5">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-blue-500 via-teal-400 to-indigo-500 transition-all duration-500 ease-out"
                  style={{ width: `${Math.min(100, Math.max(12, (elapsed / 30) * 100))}%` }}
                />
              </div>
            </div>

            {/* Subsystem checklist */}
            <div className="grid grid-cols-3 gap-2 rounded-xl border border-white/5 bg-black/30 p-3">
              <div className="flex flex-col items-center rounded-lg p-2 text-center">
                <Database className="mb-1 h-4 w-4 text-neutral-400" />
                <span className="text-[11px] font-medium text-neutral-300">SQLite DB</span>
                <span className="text-[10px] text-neutral-500">
                  {stage === 'init' ? 'Pending' : 'Ready'}
                </span>
              </div>
              <div className="flex flex-col items-center rounded-lg p-2 text-center">
                <Layers className="mb-1 h-4 w-4 text-neutral-400" />
                <span className="text-[11px] font-medium text-neutral-300">Vector Store</span>
                <span className="text-[10px] text-neutral-500">
                  {['init', 'database'].includes(stage) ? 'Pending' : 'Mounted'}
                </span>
              </div>
              <div className="flex flex-col items-center rounded-lg p-2 text-center">
                <Cpu className="mb-1 h-4 w-4 text-neutral-400" />
                <span className="text-[11px] font-medium text-neutral-300">AI Models</span>
                <span className="text-[10px] text-neutral-500">
                  {stage === 'providers' ? 'Detecting' : 'Waiting'}
                </span>
              </div>
            </div>

            <p className="text-center text-[11px] text-neutral-500">
              Starting local high-concurrency desktop worker. First launch may take a few seconds to verify vector indexes.
            </p>
          </div>
        ) : (
          /* Error State */
          <div className="mt-6 flex flex-col space-y-5">
            <div className="rounded-xl border border-red-500/20 bg-red-950/20 p-4">
              <div className="flex items-start gap-3">
                <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-red-400" />
                <div className="flex flex-col space-y-1">
                  <h3 className="text-sm font-semibold text-red-200">Backend Startup Failure</h3>
                  <p className="text-xs text-red-300/80 leading-relaxed">
                    {errorMessage || 'Unable to establish loopback connection to quantiva-backend.exe.'}
                  </p>
                  {healthData && (
                    <div className="mt-2 text-[10px] font-mono text-red-400/90">
                      SQLite: {healthData.sqlite_status} • Vector: {healthData.vector_store_status}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={handleOpenLogs}
                className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3.5 py-2 text-xs font-medium text-neutral-300 hover:bg-white/10 transition-colors"
              >
                <FolderOpen className="h-3.5 w-3.5 text-neutral-400" />
                View Logs
              </button>

              <button
                type="button"
                onClick={handleRetry}
                disabled={isRetrying}
                className="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-4 py-2 text-xs font-semibold text-white shadow-md shadow-blue-600/30 hover:bg-blue-500 disabled:opacity-50 transition-colors"
              >
                <RotateCw className={`h-3.5 w-3.5 ${isRetrying ? 'animate-spin' : ''}`} />
                {isRetrying ? 'Retrying…' : 'Retry Startup'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Footer Info */}
      <div className="mt-8 flex items-center gap-2 text-[11px] text-neutral-600 font-mono">
        <HardDrive className="h-3 w-3" />
        <span>127.0.0.1 Loopback Isolation • Windows Desktop Edition</span>
      </div>
    </div>
  );
}
