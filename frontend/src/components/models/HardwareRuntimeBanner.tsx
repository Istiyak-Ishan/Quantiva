"use client";

import React, { useState } from 'react';
import {
  Cpu,
  Server,
  RefreshCw,
  CheckCircle2,
  AlertTriangle,
  HardDrive,
  Sparkles,
  Zap,
} from 'lucide-react';
import { SystemHardware, OllamaStatus } from '@/lib/api';

interface HardwareRuntimeBannerProps {
  hardware: SystemHardware | null;
  ollama: OllamaStatus | null;
  loading: boolean;
  onRefresh: () => void;
}

export function HardwareRuntimeBanner({
  hardware,
  ollama,
  loading,
  onRefresh,
}: HardwareRuntimeBannerProps) {
  const hw = hardware?.hardware;
  const rec = hardware?.recommendation;

  return (
    <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-r from-blue-950/30 via-[#0e0e13] to-teal-950/20 p-5 md:p-6 shadow-xl">
      {/* Glow overlay */}
      <div className="pointer-events-none absolute -top-12 -right-12 h-36 w-36 rounded-full bg-blue-500/10 blur-3xl" />

      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        {/* Left: Summary Title and Hardware Specs */}
        <div className="space-y-3 flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-blue-500/10 text-blue-400 border border-blue-500/20">
              <Cpu className="h-4 w-4" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-tight">
                Hardware Calibration & Runtime Overview
              </h2>
              <p className="text-xs text-neutral-400">
                Local hardware capacity dictates optimal offline model quantization and inference speed.
              </p>
            </div>
          </div>

          {/* Hardware Specs Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
            {/* CPU */}
            <div className="rounded-xl border border-white/5 bg-black/40 p-2.5">
              <span className="text-[10px] uppercase font-semibold text-neutral-500">Processor</span>
              <p className="text-xs font-semibold text-white truncate mt-0.5" title={hw?.cpu_model}>
                {hw?.cpu_model || 'Detecting CPU…'}
              </p>
              <span className="text-[10px] text-neutral-400">
                {hw ? `${hw.physical_cores} Cores / ${hw.logical_cores} Threads` : '—'}
              </span>
            </div>

            {/* RAM */}
            <div className="rounded-xl border border-white/5 bg-black/40 p-2.5">
              <span className="text-[10px] uppercase font-semibold text-neutral-500">System RAM</span>
              <p className="text-xs font-semibold text-white mt-0.5">
                {hw ? `${hw.ram_total_gb.toFixed(1)} GB Total` : 'Detecting RAM…'}
              </p>
              <span className="text-[10px] text-teal-400">
                {hw ? `${hw.ram_available_gb.toFixed(1)} GB Available` : '—'}
              </span>
            </div>

            {/* GPU / VRAM */}
            <div className="rounded-xl border border-white/5 bg-black/40 p-2.5">
              <span className="text-[10px] uppercase font-semibold text-neutral-500">GPU / VRAM</span>
              <p className="text-xs font-semibold text-white truncate mt-0.5" title={hw?.gpus?.[0]?.name}>
                {hw?.gpus?.[0]?.name || 'DirectML / Integrated'}
              </p>
              <span className="text-[10px] text-purple-400">
                {hw?.gpus?.[0] ? `${hw.gpus[0].vram_total_gb.toFixed(1)} GB VRAM` : 'CPU Acceleration'}
              </span>
            </div>

            {/* Recommended Tier */}
            <div className="rounded-xl border border-white/5 bg-black/40 p-2.5">
              <span className="text-[10px] uppercase font-semibold text-neutral-500">Recommended Size</span>
              <p className="text-xs font-semibold text-blue-400 mt-0.5 truncate">
                {rec?.llm_models?.[0] || 'qwen2.5:7b'}
              </p>
              <span className="text-[10px] text-neutral-400 truncate block">
                {rec?.description || 'Calibrated for system'}
              </span>
            </div>
          </div>
        </div>

        {/* Right: Ollama Daemon Card */}
        <div className="shrink-0 rounded-xl border border-white/10 bg-black/50 p-4 lg:w-72 flex flex-col justify-between space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Server className="h-4 w-4 text-teal-400" />
              <span className="text-xs font-semibold text-white">Ollama Daemon</span>
            </div>

            <button
              type="button"
              onClick={onRefresh}
              disabled={loading}
              className="flex h-6 w-6 items-center justify-center rounded-md text-neutral-400 hover:bg-white/5 hover:text-white transition-colors"
              title="Refresh Daemon Status"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>

          <div>
            {ollama?.is_running ? (
              <div className="space-y-1">
                <div className="inline-flex items-center gap-1.5 rounded-full bg-teal-500/10 px-2.5 py-0.5 text-[11px] font-medium text-teal-300 border border-teal-500/20">
                  <CheckCircle2 className="h-3.5 w-3.5 text-teal-400" />
                  <span>Connected {ollama.version ? `(v${ollama.version})` : ''}</span>
                </div>
                <p className="text-[10px] text-neutral-400 pt-1 font-mono">
                  {ollama.models?.length ?? 0} local model(s) installed
                </p>
              </div>
            ) : (
              <div className="space-y-1">
                <div className="inline-flex items-center gap-1.5 rounded-full bg-amber-500/10 px-2.5 py-0.5 text-[11px] font-medium text-amber-300 border border-amber-500/20">
                  <AlertTriangle className="h-3.5 w-3.5 text-amber-400" />
                  <span>Not Running</span>
                </div>
                <p className="text-[10px] text-neutral-400 pt-1">
                  Ollama is not responding on 127.0.0.1:11434
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
