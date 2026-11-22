"use client";

import React, { useEffect, useState } from 'react';
import {
  Settings,
  Cpu,
  Database,
  FolderOpen,
  RotateCw,
  HardDrive,
  RefreshCw,
  FileText,
  AlertCircle,
  CheckCircle2,
  Loader2,
  ArrowRight,
} from 'lucide-react';
import {
  getSystemHealth,
  getSystemHardware,
  getStorageStatus,
  migrateStorageLocation,
  SystemHealth,
  SystemHardware,
  StorageStatus,
} from '@/lib/api';
import { invokeTauriCommand, isTauriDesktop } from '@/lib/api-client';

export default function SystemSettingsPage() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [hw, setHw] = useState<SystemHardware | null>(null);
  const [storage, setStorage] = useState<StorageStatus | null>(null);
  const [loading, setLoading] = useState(true);

  // Migration state
  const [isMigrating, setIsMigrating] = useState(false);
  const [newPathInput, setNewPathInput] = useState('');
  const [migrationMessage, setMigrationMessage] = useState<{
    type: 'success' | 'error';
    text: string;
  } | null>(null);

  const fetchSystemData = async () => {
    setLoading(true);
    try {
      const [hRes, hwRes, sRes] = await Promise.allSettled([
        getSystemHealth(),
        getSystemHardware(),
        getStorageStatus(),
      ]);
      if (hRes.status === 'fulfilled') setHealth(hRes.value);
      if (hwRes.status === 'fulfilled') setHw(hwRes.value);
      if (sRes.status === 'fulfilled') setStorage(sRes.value);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchSystemData();
  }, []);

  const handleOpenStorage = async () => {
    try {
      if (isTauriDesktop()) {
        await invokeTauriCommand('open_data_folder');
      } else if (storage?.current_data_dir) {
        window.alert(`Quantiva Data Directory:\n${storage.current_data_dir}`);
      }
    } catch (err) {
      console.error('Failed to open data folder:', err);
    }
  };

  const handleOpenLogs = async () => {
    try {
      if (isTauriDesktop()) {
        await invokeTauriCommand('open_data_folder', { subfolder: 'logs' });
      } else if (storage?.current_data_dir) {
        window.alert(`Quantiva Logs Directory:\n${storage.current_data_dir}/logs`);
      }
    } catch (err) {
      console.error('Failed to open logs folder:', err);
    }
  };

  const handleRestartBackend = async () => {
    if (!window.confirm('Restart the Quantiva desktop backend process?')) return;
    try {
      if (isTauriDesktop()) {
        await invokeTauriCommand('restart_backend');
      }
      window.location.reload();
    } catch (err) {
      alert('Failed to restart backend');
    }
  };

  const handleMigrateStorage = async (e: React.FormEvent) => {
    e.preventDefault();
    const targetPath = newPathInput.trim();
    if (!targetPath) return;

    if (
      !window.confirm(
        `Migrate Quantiva data directory to:\n${targetPath}\n\nExisting databases, vector stores, and uploads will be moved atomically.`
      )
    ) {
      return;
    }

    setIsMigrating(true);
    setMigrationMessage(null);
    try {
      const res = await migrateStorageLocation(targetPath, true);
      setMigrationMessage({
        type: 'success',
        text: `Data directory migrated successfully to: ${res.new_data_dir}`,
      });
      setNewPathInput('');
      await fetchSystemData();
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'Storage migration failed.';
      setMigrationMessage({
        type: 'error',
        text: message,
      });
    } finally {
      setIsMigrating(false);
    }
  };

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">
            System Settings & Diagnostics
          </h1>
          <p className="text-xs text-neutral-400 mt-1">
            Hardware acceleration, storage paths, SQLite WAL engine health, and log management.
          </p>
        </div>

        <button
          type="button"
          onClick={fetchSystemData}
          className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-neutral-400 hover:text-white transition-colors"
          title="Refresh Diagnostics"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl">
        {/* Backend Subsystems & Storage Info */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-6 space-y-4">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-blue-400" />
            <h2 className="text-sm font-semibold text-white">Subsystems & Storage</h2>
          </div>

          <div className="space-y-2.5 text-xs">
            <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
              <span>Overall Backend Status</span>
              <span className="font-semibold text-emerald-400 capitalize">
                {health?.status || 'Active'}
              </span>
            </div>
            <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
              <span>SQLite Database Engine</span>
              <span className="font-mono text-neutral-200">
                {health?.sqlite_status || 'WAL Mode'}
              </span>
            </div>
            <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
              <span>Vector Engine</span>
              <span className="font-mono text-neutral-200">
                {health?.vector_store_status || 'Embedded'}
              </span>
            </div>
            <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
              <span>App Version</span>
              <span className="font-mono text-neutral-200">
                v{health?.app_version || '1.0.0'}
              </span>
            </div>
            {storage && (
              <>
                <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
                  <span>Indexed Documents</span>
                  <span className="font-mono text-neutral-200">
                    {storage.documents_count} documents
                  </span>
                </div>
                <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
                  <span>Database File Size</span>
                  <span className="font-mono text-neutral-200">
                    {(storage.database_size_bytes / (1024 * 1024)).toFixed(2)} MB
                  </span>
                </div>
                <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
                  <span>Free Disk Space</span>
                  <span className="font-mono text-neutral-200">
                    {(storage.disk_free_bytes / (1024 * 1024 * 1024)).toFixed(1)} GB
                  </span>
                </div>
              </>
            )}
          </div>

          {storage && (
            <div className="pt-2 space-y-2">
              <span className="text-[10px] uppercase font-semibold text-neutral-500">
                Current Data Directory
              </span>
              <p className="font-mono text-[11px] text-neutral-300 break-all bg-neutral-900/60 p-2 rounded border border-white/5">
                {storage.current_data_dir}
              </p>

              <div className="flex flex-wrap gap-2 pt-1">
                <button
                  type="button"
                  onClick={handleOpenStorage}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-neutral-200 hover:bg-white/10 transition-colors"
                >
                  <FolderOpen className="h-3.5 w-3.5" />
                  <span>Open Data Directory</span>
                </button>

                <button
                  type="button"
                  onClick={handleOpenLogs}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-neutral-200 hover:bg-white/10 transition-colors"
                >
                  <FileText className="h-3.5 w-3.5" />
                  <span>Open Logs Directory</span>
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Hardware Calibration Card */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-6 space-y-4">
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-teal-400" />
            <h2 className="text-sm font-semibold text-white">Hardware Diagnostics</h2>
          </div>

          {hw?.hardware ? (
            <div className="space-y-2.5 text-xs">
              <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
                <span>Processor (CPU)</span>
                <span className="font-medium text-neutral-200 truncate max-w-[200px]" title={hw.hardware.cpu_model}>
                  {hw.hardware.cpu_model}
                </span>
              </div>
              <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
                <span>Cores / Threads</span>
                <span className="font-mono text-neutral-200">
                  {hw.hardware.physical_cores} Cores / {hw.hardware.logical_cores} Threads
                </span>
              </div>
              <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
                <span>System Memory (RAM)</span>
                <span className="font-mono text-neutral-200">
                  {hw.hardware.ram_total_gb.toFixed(1)} GB ({hw.hardware.ram_available_gb.toFixed(1)} GB Free)
                </span>
              </div>
              <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
                <span>GPU Acceleration</span>
                <span className="font-medium text-neutral-200 truncate max-w-[200px]">
                  {hw.hardware.gpus?.[0]?.name || 'CPU DirectML'}
                </span>
              </div>
              <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
                <span>Recommended Tier</span>
                <span className="font-semibold text-teal-400">
                  {hw.recommendation?.description || 'Standard Local Tier'}
                </span>
              </div>
            </div>
          ) : (
            <p className="text-xs text-neutral-400">Hardware inspection data is loading…</p>
          )}

          <div className="pt-2">
            <button
              type="button"
              onClick={handleRestartBackend}
              className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-neutral-300 hover:bg-white/10 transition-colors"
            >
              <RotateCw className="h-3.5 w-3.5" />
              <span>Restart Desktop Backend Process</span>
            </button>
          </div>
        </div>
      </div>

      {/* Storage Migration Card */}
      <div className="rounded-xl border border-white/10 bg-black/30 p-6 space-y-4 max-w-4xl">
        <div className="flex items-center gap-2">
          <HardDrive className="h-4 w-4 text-amber-400" />
          <h2 className="text-sm font-semibold text-white">
            Migrate QUANTIVA_DATA_DIR Storage Path
          </h2>
        </div>
        <p className="text-xs text-neutral-400 leading-relaxed">
          Relocate your local Quantiva database, vector chunks, embeddings, and uploaded documents
          to a different hard drive or partition (e.g. to move onto a faster SSD or larger drive).
        </p>

        {migrationMessage && (
          <div
            className={`rounded-lg p-3 text-xs ${
              migrationMessage.type === 'success'
                ? 'bg-emerald-950/20 border border-emerald-500/20 text-emerald-300'
                : 'bg-red-950/20 border border-red-500/20 text-red-300'
            }`}
          >
            {migrationMessage.text}
          </div>
        )}

        <form onSubmit={handleMigrateStorage} className="space-y-3 pt-1">
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
            <input
              type="text"
              placeholder="e.g. D:\QuantivaData or C:\Users\YourName\Quantiva"
              value={newPathInput}
              onChange={(e) => setNewPathInput(e.target.value)}
              disabled={isMigrating}
              className="flex-1 rounded-xl border border-white/10 bg-black/60 px-3.5 py-2 text-xs font-mono text-white placeholder:text-neutral-500 focus:outline-none focus:border-amber-500/60"
            />
            <button
              type="submit"
              disabled={isMigrating || !newPathInput.trim()}
              className="inline-flex items-center justify-center gap-1.5 rounded-xl bg-amber-600 px-4 py-2 text-xs font-semibold text-white shadow-md shadow-amber-600/30 hover:bg-amber-500 disabled:opacity-50 transition-colors"
            >
              {isMigrating ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <ArrowRight className="h-3.5 w-3.5" />
              )}
              <span>{isMigrating ? 'Migrating Files…' : 'Migrate Storage'}</span>
            </button>
          </div>
          <span className="text-[10px] text-neutral-500 block">
            The target directory will be created if it does not already exist. Free disk space is verified before copying.
          </span>
        </form>
      </div>
    </div>
  );
}
