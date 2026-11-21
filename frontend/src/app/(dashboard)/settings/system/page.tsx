"use client";

import React, { useEffect, useState } from 'react';
import {
  Settings,
  Cpu,
  Database,
  ShieldCheck,
  FolderOpen,
  RotateCw,
  Power,
  HardDrive,
  RefreshCw,
  CheckCircle2,
} from 'lucide-react';
import {
  getSystemHealth,
  getSystemHardware,
  getStorageStatus,
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

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">System Settings & Diagnostics</h1>
          <p className="text-xs text-neutral-400 mt-1">
            Low-level hardware diagnostics, database WAL health, and desktop process management.
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
        {/* Backend Subsystems Health */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-6 space-y-4">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-blue-400" />
            <h2 className="text-sm font-semibold text-white">Subsystems & Storage</h2>
          </div>

          <div className="space-y-2.5 text-xs">
            <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
              <span>Overall Backend Status</span>
              <span className="font-semibold text-green-400 capitalize">{health?.status || 'Active'}</span>
            </div>
            <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
              <span>SQLite Database Engine</span>
              <span className="font-mono text-neutral-200">{health?.sqlite_status || 'WAL Mode'}</span>
            </div>
            <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
              <span>Vector Engine</span>
              <span className="font-mono text-neutral-200">{health?.vector_store_status || 'Embedded'}</span>
            </div>
            <div className="flex items-center justify-between border-b border-white/5 pb-2 text-neutral-400">
              <span>App Version</span>
              <span className="font-mono text-neutral-200">{health?.app_version || '1.0.0'}</span>
            </div>
          </div>

          {storage && (
            <div className="pt-2">
              <span className="text-[10px] uppercase font-semibold text-neutral-500">Local Data Directory</span>
              <p className="mt-1 font-mono text-[11px] text-neutral-300 break-all">{storage.current_data_dir}</p>
              <button
                type="button"
                onClick={handleOpenStorage}
                className="mt-3 inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-neutral-200 hover:bg-white/10 transition-colors"
              >
                <FolderOpen className="h-3.5 w-3.5" />
                Open Data Directory in Windows Explorer
              </button>
            </div>
          )}
        </div>

        {/* Hardware Profile Card */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-6 space-y-4">
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-teal-400" />
            <h2 className="text-sm font-semibold text-white">Hardware Calibration</h2>
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
                <span className="font-semibold text-teal-400">{hw.recommendation?.description}</span>
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
              Restart Backend Process
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
