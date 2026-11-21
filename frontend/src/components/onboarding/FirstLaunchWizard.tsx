"use client";

import React, { useState, useEffect, useCallback } from 'react';
import {
  ShieldCheck,
  Cpu,
  Cloud,
  CheckCircle2,
  AlertTriangle,
  Sparkles,
  Server,
  Layers,
  ArrowRight,
  ArrowLeft,
  Loader2,
  Key,
  Database,
  Eye,
  EyeOff,
  RefreshCw,
  HardDrive,
  Check,
} from 'lucide-react';
import {
  getSystemHardware,
  getOllamaStatus,
  testLLMConnection,
  testEmbeddingConnection,
  createLLMConfig,
  createKnowledgeBase,
  SystemHardware,
  OllamaStatus,
  LLMConnectionTestResult,
  EmbeddingTestResult,
} from '@/lib/api';

export type OperatingMode = 'fully_local' | 'hybrid' | 'cloud';

interface FirstLaunchWizardProps {
  isOpen: boolean;
  onComplete: () => void;
  onClose?: () => void;
}

export function FirstLaunchWizard({ isOpen, onComplete, onClose }: FirstLaunchWizardProps) {
  // Step state: 1 = Mode Selection, 2 = Model Config & Test, 3 = Create First KB
  const [step, setStep] = useState<1 | 2 | 3>(1);

  // Operating Mode
  const [mode, setMode] = useState<OperatingMode>('fully_local');

  // Hardware detection
  const [hardwareData, setHardwareData] = useState<SystemHardware | null>(null);
  const [hwLoading, setHwLoading] = useState(false);

  // Ollama status
  const [ollama, setOllama] = useState<OllamaStatus | null>(null);
  const [ollamaLoading, setOllamaLoading] = useState(false);

  // Local model choices
  const [selectedLocalModel, setSelectedLocalModel] = useState<string>('qwen2.5:7b');
  const [localEmbeddingModel, setLocalEmbeddingModel] = useState<string>('all-MiniLM-L6-v2');

  // Cloud model choices
  const [cloudProvider, setCloudProvider] = useState<'gemini' | 'openai'>('gemini');
  const [cloudModel, setCloudModel] = useState<string>('gemini-1.5-flash');
  const [cloudApiKey, setCloudApiKey] = useState<string>('');
  const [showApiKey, setShowApiKey] = useState(false);

  // Testing states
  const [llmTesting, setLlmTesting] = useState(false);
  const [llmTestResult, setLlmTestResult] = useState<LLMConnectionTestResult | null>(null);
  const [embeddingTesting, setEmbeddingTesting] = useState(false);
  const [embeddingTestResult, setEmbeddingTestResult] = useState<EmbeddingTestResult | null>(null);

  // Step 3 Knowledge Base
  const [kbName, setKbName] = useState('Default Knowledge Base');
  const [kbDescription, setKbDescription] = useState('Primary local workspace for private documents and research.');
  const [isFinishing, setIsFinishing] = useState(false);
  const [finishError, setFinishError] = useState<string | null>(null);

  // Fetch hardware on mount
  useEffect(() => {
    if (!isOpen) return;
    let active = true;

    async function loadHardware() {
      setHwLoading(true);
      try {
        const hw = await getSystemHardware();
        if (active) {
          setHardwareData(hw);
          // Suggest appropriate model based on recommendation
          if (hw.recommendation?.tier === 'tier_1_3b') {
            setSelectedLocalModel('qwen2.5:3b');
          } else if (hw.recommendation?.tier === 'tier_3_14b') {
            setSelectedLocalModel('qwen2.5:14b');
          } else {
            setSelectedLocalModel('qwen2.5:7b');
          }
        }
      } catch (err) {
        console.warn('Failed to inspect hardware:', err);
      } finally {
        if (active) setHwLoading(false);
      }
    }

    void loadHardware();
    return () => {
      active = false;
    };
  }, [isOpen]);

  // Fetch Ollama status when entering Step 2 in local or hybrid mode
  const checkOllama = useCallback(async () => {
    setOllamaLoading(true);
    try {
      const status = await getOllamaStatus();
      setOllama(status);
      if (status.models && status.models.length > 0) {
        // If user hasn't chosen one, or the chosen one is not in the list, pre-select first installed
        const names = status.models.map((m) => m.name);
        if (!names.includes(selectedLocalModel)) {
          setSelectedLocalModel(names[0]);
        }
      }
    } catch (err) {
      console.warn('Failed to detect Ollama:', err);
    } finally {
      setOllamaLoading(false);
    }
  }, [selectedLocalModel]);

  useEffect(() => {
    if (step === 2 && (mode === 'fully_local' || mode === 'hybrid')) {
      void checkOllama();
    }
  }, [step, mode, checkOllama]);

  // Test LLM Connection handler
  const handleTestLLM = async () => {
    setLlmTesting(true);
    setLlmTestResult(null);
    try {
      if (mode === 'fully_local' || (mode === 'hybrid' && cloudProvider === 'gemini' && !cloudApiKey)) {
        const res = await testLLMConnection({
          provider_type: 'ollama',
          model_name: selectedLocalModel,
          base_url: ollama?.base_url || 'http://127.0.0.1:11434',
        });
        setLlmTestResult(res);
      } else {
        const res = await testLLMConnection({
          provider_type: cloudProvider,
          model_name: cloudModel,
          api_key: cloudApiKey.trim() || undefined,
        });
        setLlmTestResult(res);
      }
    } catch (err) {
      setLlmTestResult({
        success: false,
        latency_ms: 0,
        provider: mode === 'fully_local' ? 'ollama' : cloudProvider,
        model: mode === 'fully_local' ? selectedLocalModel : cloudModel,
        error_message: err instanceof Error ? err.message : 'Connection test failed',
      });
    } finally {
      setLlmTesting(false);
    }
  };

  // Test Embedding Connection handler
  const handleTestEmbedding = async () => {
    setEmbeddingTesting(true);
    setEmbeddingTestResult(null);
    try {
      const res = await testEmbeddingConnection({
        provider_type: 'local_sentence_transformers',
        model_name: localEmbeddingModel,
      });
      setEmbeddingTestResult(res);
    } catch (err) {
      setEmbeddingTestResult({
        success: false,
        latency_ms: 0,
        provider: 'local_sentence_transformers',
        model: localEmbeddingModel,
        error_message: err instanceof Error ? err.message : 'Embedding test failed',
      });
    } finally {
      setEmbeddingTesting(false);
    }
  };

  // Complete setup handler
  const handleFinish = async () => {
    setIsFinishing(true);
    setFinishError(null);
    try {
      // 1. Create Default LLM Config
      if (mode === 'fully_local') {
        await createLLMConfig({
          name: `Ollama ${selectedLocalModel}`,
          provider_type: 'ollama',
          model_name: selectedLocalModel,
          base_url: ollama?.base_url || 'http://127.0.0.1:11434',
          is_default: true,
        });
      } else if (mode === 'cloud' || mode === 'hybrid') {
        if (cloudApiKey.trim()) {
          await createLLMConfig({
            name: `${cloudProvider === 'gemini' ? 'Google Gemini' : 'OpenAI'} (${cloudModel})`,
            provider_type: cloudProvider,
            model_name: cloudModel,
            api_key: cloudApiKey.trim(),
            is_default: true,
          });
        }
      }

      // 2. Create the first Knowledge Base
      await createKnowledgeBase({
        name: kbName.trim() || 'Default Knowledge Base',
        description: kbDescription.trim() || null,
      });

      // 3. Mark onboarding as complete in localStorage
      localStorage.setItem('quantiva_onboarding_completed', 'true');
      localStorage.setItem('quantiva_operating_mode', mode);

      onComplete();
    } catch (err) {
      setFinishError(err instanceof Error ? err.message : 'Failed to finalize setup. Please retry.');
    } finally {
      setIsFinishing(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 px-4 backdrop-blur-md">
      <div className="relative flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#0e0e12] shadow-2xl">
        {/* Header Glow */}
        <div className="pointer-events-none absolute -top-24 left-1/2 -translate-x-1/2 h-48 w-96 rounded-full bg-blue-600/20 blur-[100px]" />

        {/* Wizard Header */}
        <div className="relative border-b border-white/10 px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-blue-600 to-teal-400 p-[1px]">
                <div className="flex h-full w-full items-center justify-center rounded-xl bg-[#0c0c10]">
                  <Sparkles className="h-5 w-5 text-blue-400" />
                </div>
              </div>
              <div>
                <h2 className="text-xl font-bold tracking-tight text-white">Welcome to Quantiva</h2>
                <p className="text-xs text-neutral-400">Desktop First-Launch Setup & Hardware Calibration</p>
              </div>
            </div>

            {/* Step Indicators */}
            <div className="flex items-center gap-2">
              {[
                { num: 1, label: 'Mode' },
                { num: 2, label: 'Models' },
                { num: 3, label: 'Workspace' },
              ].map((s) => (
                <div key={s.num} className="flex items-center gap-1.5">
                  <div
                    className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-semibold transition-colors ${
                      step === s.num
                        ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30'
                        : step > s.num
                        ? 'bg-teal-500/20 text-teal-400 border border-teal-500/30'
                        : 'bg-white/5 text-neutral-500 border border-white/5'
                    }`}
                  >
                    {step > s.num ? <Check className="h-3.5 w-3.5" /> : s.num}
                  </div>
                  <span
                    className={`text-xs font-medium ${
                      step === s.num ? 'text-white' : 'text-neutral-500'
                    }`}
                  >
                    {s.label}
                  </span>
                  {s.num < 3 && <div className="mx-1 h-px w-4 bg-white/10" />}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Wizard Body (Scrollable) */}
        <div className="flex-1 overflow-y-auto px-8 py-6">
          {/* ════════════════════════════════════════════════════════════════════
              STEP 1: SELECT OPERATING MODE & HARDWARE SUMMARY
              ════════════════════════════════════════════════════════════════════ */}
          {step === 1 && (
            <div className="flex flex-col space-y-6">
              <div>
                <h3 className="text-base font-semibold text-white">Choose Your Operating Mode</h3>
                <p className="text-xs text-neutral-400">
                  Select how you want Quantiva to balance privacy, latency, and model capability.
                </p>
              </div>

              {/* Operating Mode Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* 1. Fully Local */}
                <div
                  onClick={() => setMode('fully_local')}
                  className={`relative cursor-pointer rounded-xl border p-5 transition-all flex flex-col justify-between ${
                    mode === 'fully_local'
                      ? 'border-blue-500 bg-blue-500/10 shadow-lg shadow-blue-500/10'
                      : 'border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]'
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-teal-500/10 text-teal-400">
                        <ShieldCheck className="h-5 w-5" />
                      </div>
                      <span className="rounded-full bg-teal-500/20 px-2 py-0.5 text-[10px] font-semibold text-teal-300">
                        Offline Ready
                      </span>
                    </div>
                    <h4 className="text-sm font-semibold text-white">Fully Local</h4>
                    <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed">
                      100% offline. Uses Ollama for LLM, local Sentence Transformers for embeddings, and SQLite/Qdrant
                      on your local drive. Zero cloud traffic.
                    </p>
                  </div>
                  <div className="mt-4 pt-3 border-t border-white/5 flex items-center gap-1.5 text-[11px] font-medium text-teal-400">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>Maximum Privacy & Zero Costs</span>
                  </div>
                </div>

                {/* 2. Hybrid Mode */}
                <div
                  onClick={() => setMode('hybrid')}
                  className={`relative cursor-pointer rounded-xl border p-5 transition-all flex flex-col justify-between ${
                    mode === 'hybrid'
                      ? 'border-blue-500 bg-blue-500/10 shadow-lg shadow-blue-500/10'
                      : 'border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]'
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-purple-500/10 text-purple-400">
                        <Server className="h-5 w-5" />
                      </div>
                      <span className="rounded-full bg-purple-500/20 px-2 py-0.5 text-[10px] font-semibold text-purple-300">
                        Balanced
                      </span>
                    </div>
                    <h4 className="text-sm font-semibold text-white">Hybrid Mode</h4>
                    <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed">
                      Embeddings and vector storage stay strictly on your local PC. Queries can route to high-capacity
                      Cloud or Local LLMs for complex multi-document reasoning.
                    </p>
                  </div>
                  <div className="mt-4 pt-3 border-t border-white/5 flex items-center gap-1.5 text-[11px] font-medium text-purple-400">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>Local Data + Cloud Intelligence</span>
                  </div>
                </div>

                {/* 3. Cloud API Mode */}
                <div
                  onClick={() => setMode('cloud')}
                  className={`relative cursor-pointer rounded-xl border p-5 transition-all flex flex-col justify-between ${
                    mode === 'cloud'
                      ? 'border-blue-500 bg-blue-500/10 shadow-lg shadow-blue-500/10'
                      : 'border-white/10 bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.04]'
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-500/10 text-blue-400">
                        <Cloud className="h-5 w-5" />
                      </div>
                      <span className="rounded-full bg-blue-500/20 px-2 py-0.5 text-[10px] font-semibold text-blue-300">
                        High Speed
                      </span>
                    </div>
                    <h4 className="text-sm font-semibold text-white">Cloud API Mode</h4>
                    <p className="mt-1.5 text-xs text-neutral-400 leading-relaxed">
                      Powered by Google Gemini 1.5/2.0 or OpenAI GPT-4o. Document chunks are securely stored on your
                      local disk, using cloud LLMs only during prompt answering.
                    </p>
                  </div>
                  <div className="mt-4 pt-3 border-t border-white/5 flex items-center gap-1.5 text-[11px] font-medium text-blue-400">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>Instant Setup • No GPU Needed</span>
                  </div>
                </div>
              </div>

              {/* Hardware Detection Summary Card */}
              <div className="rounded-xl border border-white/10 bg-black/40 p-4">
                <div className="flex items-center justify-between pb-3 border-b border-white/5">
                  <div className="flex items-center gap-2 text-xs font-semibold text-neutral-200">
                    <Cpu className="h-4 w-4 text-blue-400" />
                    <span>Detected Hardware Profile</span>
                  </div>
                  {hwLoading ? (
                    <span className="flex items-center gap-1.5 text-[11px] text-neutral-400">
                      <Loader2 className="h-3 w-3 animate-spin text-blue-400" />
                      Detecting hardware…
                    </span>
                  ) : hardwareData ? (
                    <span className="rounded-full bg-blue-500/10 px-2.5 py-0.5 text-[10px] font-medium text-blue-300 border border-blue-500/20">
                      {hardwareData.recommendation?.description || 'Detected'}
                    </span>
                  ) : null}
                </div>

                {hardwareData && (
                  <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                    <div className="flex flex-col">
                      <span className="text-[10px] uppercase tracking-wider text-neutral-500">CPU</span>
                      <span className="mt-0.5 font-medium text-neutral-300 truncate" title={hardwareData.hardware.cpu_model}>
                        {hardwareData.hardware.cpu_model || 'Unknown CPU'}
                      </span>
                      <span className="text-[10px] text-neutral-500">
                        {hardwareData.hardware.physical_cores} Cores / {hardwareData.hardware.logical_cores} Threads
                      </span>
                    </div>

                    <div className="flex flex-col">
                      <span className="text-[10px] uppercase tracking-wider text-neutral-500">System RAM</span>
                      <span className="mt-0.5 font-medium text-neutral-300">
                        {hardwareData.hardware.ram_total_gb.toFixed(1)} GB Total
                      </span>
                      <span className="text-[10px] text-neutral-500">
                        {hardwareData.hardware.ram_available_gb.toFixed(1)} GB Available
                      </span>
                    </div>

                    <div className="flex flex-col">
                      <span className="text-[10px] uppercase tracking-wider text-neutral-500">GPU & VRAM</span>
                      <span className="mt-0.5 font-medium text-neutral-300 truncate">
                        {hardwareData.hardware.gpus?.[0]?.name || 'Integrated / CPU'}
                      </span>
                      <span className="text-[10px] text-neutral-500">
                        {hardwareData.hardware.gpus?.[0]
                          ? `${hardwareData.hardware.gpus[0].vram_total_gb.toFixed(1)} GB VRAM`
                          : 'CPU Acceleration'}
                      </span>
                    </div>

                    <div className="flex flex-col">
                      <span className="text-[10px] uppercase tracking-wider text-neutral-500">Recommended Model</span>
                      <span className="mt-0.5 font-semibold text-teal-400">
                        {hardwareData.recommendation?.llm_models?.[0] || 'qwen2.5:7b'}
                      </span>
                      <span className="text-[10px] text-neutral-400">
                        {hardwareData.recommendation?.tier === 'tier_1_3b' ? '3B Parameters' : '7B Parameters'}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ════════════════════════════════════════════════════════════════════
              STEP 2: CONFIGURE & TEST MODELS
              ════════════════════════════════════════════════════════════════════ */}
          {step === 2 && (
            <div className="flex flex-col space-y-6">
              <div>
                <h3 className="text-base font-semibold text-white">Configure & Calibrate AI Models</h3>
                <p className="text-xs text-neutral-400">
                  Verify your chosen model provider can stream tokens and calculate vector embeddings.
                </p>
              </div>

              {/* LOCAL / HYBRID: OLLAMA CONFIGURATION */}
              {(mode === 'fully_local' || mode === 'hybrid') && (
                <div className="flex flex-col space-y-4 rounded-xl border border-white/10 bg-black/30 p-5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Server className="h-4 w-4 text-teal-400" />
                      <span className="text-sm font-semibold text-white">Ollama Local Daemon</span>
                    </div>

                    <button
                      type="button"
                      onClick={checkOllama}
                      disabled={ollamaLoading}
                      className="inline-flex items-center gap-1.5 text-xs text-neutral-400 hover:text-white transition-colors"
                    >
                      <RefreshCw className={`h-3 w-3 ${ollamaLoading ? 'animate-spin' : ''}`} />
                      Refresh Daemon
                    </button>
                  </div>

                  {/* Daemon Status Banner */}
                  {ollama?.is_running ? (
                    <div className="flex items-center gap-2 rounded-lg border border-teal-500/20 bg-teal-500/10 px-3 py-2 text-xs text-teal-300">
                      <CheckCircle2 className="h-4 w-4 shrink-0 text-teal-400" />
                      <span>
                        Ollama is active on <code className="font-mono">{ollama.base_url}</code>{' '}
                        {ollama.version ? `(v${ollama.version})` : ''} with {ollama.models.length} model(s) installed.
                      </span>
                    </div>
                  ) : (
                    <div className="flex items-start gap-2.5 rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-xs text-amber-300">
                      <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
                      <div className="flex flex-col space-y-1">
                        <span className="font-semibold">Ollama Daemon Not Running</span>
                        <p className="text-amber-300/80">
                          Quantiva could not connect to Ollama at <code className="font-mono">http://127.0.0.1:11434</code>.
                          Please start Ollama from your Windows Start Menu, or switch to Cloud API Mode.
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Model Selector */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                    <div className="flex flex-col space-y-1.5">
                      <label className="text-xs font-medium text-neutral-300">Selected LLM Model</label>
                      {ollama?.models && ollama.models.length > 0 ? (
                        <select
                          value={selectedLocalModel}
                          onChange={(e) => setSelectedLocalModel(e.target.value)}
                          className="rounded-lg border border-white/10 bg-neutral-900 px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
                        >
                          {ollama.models.map((m) => (
                            <option key={m.name} value={m.name}>
                              {m.name} ({m.parameter_size || 'Unknown size'})
                            </option>
                          ))}
                        </select>
                      ) : (
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            value={selectedLocalModel}
                            onChange={(e) => setSelectedLocalModel(e.target.value)}
                            placeholder="e.g. qwen2.5:7b"
                            className="flex-1 rounded-lg border border-white/10 bg-neutral-900 px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
                          />
                        </div>
                      )}
                      <span className="text-[10px] text-neutral-500">
                        Recommended: <strong className="text-neutral-300">qwen2.5:7b</strong> for 16GB RAM / 8GB VRAM,{' '}
                        <strong className="text-neutral-300">qwen2.5:3b</strong> for 8GB RAM.
                      </span>
                    </div>

                    <div className="flex flex-col space-y-1.5">
                      <label className="text-xs font-medium text-neutral-300">Local Embedding Model</label>
                      <input
                        type="text"
                        value={localEmbeddingModel}
                        disabled
                        className="rounded-lg border border-white/10 bg-neutral-900/60 px-3 py-2 text-xs text-neutral-400 cursor-not-allowed"
                      />
                      <span className="text-[10px] text-neutral-500">
                        Local Sentence-Transformers (384 dimensions, offline HuggingFace model).
                      </span>
                    </div>
                  </div>

                  {/* Actions & Test Buttons */}
                  <div className="flex flex-wrap items-center gap-3 pt-2">
                    <button
                      type="button"
                      onClick={handleTestLLM}
                      disabled={llmTesting}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-blue-500/30 bg-blue-500/10 px-3 py-1.5 text-xs font-medium text-blue-300 hover:bg-blue-500/20 transition-colors disabled:opacity-50"
                    >
                      {llmTesting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                      Test Ollama LLM
                    </button>

                    <button
                      type="button"
                      onClick={handleTestEmbedding}
                      disabled={embeddingTesting}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-teal-500/30 bg-teal-500/10 px-3 py-1.5 text-xs font-medium text-teal-300 hover:bg-teal-500/20 transition-colors disabled:opacity-50"
                    >
                      {embeddingTesting ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Layers className="h-3.5 w-3.5" />
                      )}
                      Test Local Embeddings
                    </button>
                  </div>

                  {/* Test Results Output */}
                  {llmTestResult && (
                    <div
                      className={`rounded-lg p-3 text-xs border ${
                        llmTestResult.success
                          ? 'border-green-500/20 bg-green-950/20 text-green-300'
                          : 'border-red-500/20 bg-red-950/20 text-red-300'
                      }`}
                    >
                      {llmTestResult.success ? (
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-1.5 font-medium">
                            <CheckCircle2 className="h-4 w-4 text-green-400" />
                            LLM Test Succeeded ({llmTestResult.model})
                          </span>
                          <span className="font-mono text-[11px] opacity-80">{llmTestResult.latency_ms.toFixed(0)}ms latency</span>
                        </div>
                      ) : (
                        <div className="flex items-start gap-1.5">
                          <AlertTriangle className="h-4 w-4 shrink-0 text-red-400 mt-0.5" />
                          <span>{llmTestResult.error_message || 'Failed to communicate with Ollama model.'}</span>
                        </div>
                      )}
                    </div>
                  )}

                  {embeddingTestResult && (
                    <div
                      className={`rounded-lg p-3 text-xs border ${
                        embeddingTestResult.success
                          ? 'border-teal-500/20 bg-teal-950/20 text-teal-300'
                          : 'border-red-500/20 bg-red-950/20 text-red-300'
                      }`}
                    >
                      {embeddingTestResult.success ? (
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-1.5 font-medium">
                            <CheckCircle2 className="h-4 w-4 text-teal-400" />
                            Embedding Ready • {embeddingTestResult.detected_dimensions} dimensions detected
                          </span>
                          <span className="font-mono text-[11px] opacity-80">{embeddingTestResult.latency_ms.toFixed(0)}ms</span>
                        </div>
                      ) : (
                        <div className="flex items-start gap-1.5">
                          <AlertTriangle className="h-4 w-4 shrink-0 text-red-400 mt-0.5" />
                          <span>{embeddingTestResult.error_message || 'Embedding test failed.'}</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* CLOUD API CONFIGURATION (FOR CLOUD OR HYBRID) */}
              {(mode === 'cloud' || mode === 'hybrid') && (
                <div className="flex flex-col space-y-4 rounded-xl border border-white/10 bg-black/30 p-5">
                  <div className="flex items-center gap-2">
                    <Cloud className="h-4 w-4 text-blue-400" />
                    <span className="text-sm font-semibold text-white">Cloud Provider Credentials</span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Provider */}
                    <div className="flex flex-col space-y-1.5">
                      <label className="text-xs font-medium text-neutral-300">Cloud Provider</label>
                      <select
                        value={cloudProvider}
                        onChange={(e) => {
                          const p = e.target.value as 'gemini' | 'openai';
                          setCloudProvider(p);
                          setCloudModel(p === 'gemini' ? 'gemini-1.5-flash' : 'gpt-4o-mini');
                        }}
                        className="rounded-lg border border-white/10 bg-neutral-900 px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
                      >
                        <option value="gemini">Google Gemini (Recommended)</option>
                        <option value="openai">OpenAI</option>
                      </select>
                    </div>

                    {/* Model */}
                    <div className="flex flex-col space-y-1.5">
                      <label className="text-xs font-medium text-neutral-300">Model Name</label>
                      <input
                        type="text"
                        value={cloudModel}
                        onChange={(e) => setCloudModel(e.target.value)}
                        className="rounded-lg border border-white/10 bg-neutral-900 px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
                      />
                    </div>
                  </div>

                  {/* API Key */}
                  <div className="flex flex-col space-y-1.5">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-medium text-neutral-300">API Key</label>
                      <span className="text-[10px] text-neutral-500">Stored via Windows Credential Manager</span>
                    </div>

                    <div className="relative">
                      <input
                        type={showApiKey ? 'text' : 'password'}
                        value={cloudApiKey}
                        onChange={(e) => setCloudApiKey(e.target.value)}
                        placeholder={`Enter your ${cloudProvider === 'gemini' ? 'AI Studio (Gemini)' : 'OpenAI'} API key`}
                        className="w-full rounded-lg border border-white/10 bg-neutral-900 pl-3 pr-10 py-2 text-xs text-white focus:border-blue-500 focus:outline-none font-mono"
                      />
                      <button
                        type="button"
                        onClick={() => setShowApiKey(!showApiKey)}
                        className="absolute right-2.5 top-1/2 -translate-y-1/2 text-neutral-400 hover:text-white"
                      >
                        {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </button>
                    </div>

                    <p className="text-[10px] text-neutral-500">
                      Your API key is never stored in plaintext SQLite databases or commit logs; it is DPAPI-encrypted
                      in Windows Credential Manager.
                    </p>
                  </div>

                  {/* Test Connection Button */}
                  <div className="pt-1">
                    <button
                      type="button"
                      onClick={handleTestLLM}
                      disabled={llmTesting || !cloudApiKey.trim()}
                      className="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-3.5 py-1.5 text-xs font-semibold text-white shadow-md shadow-blue-600/30 hover:bg-blue-500 disabled:opacity-50 transition-colors"
                    >
                      {llmTesting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Key className="h-3.5 w-3.5" />}
                      Test API Connection
                    </button>
                  </div>

                  {llmTestResult && (
                    <div
                      className={`rounded-lg p-3 text-xs border ${
                        llmTestResult.success
                          ? 'border-green-500/20 bg-green-950/20 text-green-300'
                          : 'border-red-500/20 bg-red-950/20 text-red-300'
                      }`}
                    >
                      {llmTestResult.success ? (
                        <div className="flex items-center justify-between">
                          <span className="flex items-center gap-1.5 font-medium">
                            <CheckCircle2 className="h-4 w-4 text-green-400" />
                            Connected to {llmTestResult.provider} ({llmTestResult.model})
                          </span>
                          <span className="font-mono text-[11px] opacity-80">{llmTestResult.latency_ms.toFixed(0)}ms latency</span>
                        </div>
                      ) : (
                        <div className="flex items-start gap-1.5">
                          <AlertTriangle className="h-4 w-4 shrink-0 text-red-400 mt-0.5" />
                          <span>{llmTestResult.error_message || 'API connection failed. Please check your key.'}</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* ════════════════════════════════════════════════════════════════════
              STEP 3: CREATE FIRST KNOWLEDGE BASE & FINALIZE
              ════════════════════════════════════════════════════════════════════ */}
          {step === 3 && (
            <div className="flex flex-col space-y-6">
              <div>
                <h3 className="text-base font-semibold text-white">Create Your First Knowledge Base</h3>
                <p className="text-xs text-neutral-400">
                  Every document collection in Quantiva has its own dedicated vector index and locked embedding space.
                </p>
              </div>

              <div className="flex flex-col space-y-4 rounded-xl border border-white/10 bg-black/30 p-5">
                <div className="flex flex-col space-y-1.5">
                  <label className="text-xs font-medium text-neutral-300">Knowledge Base Name</label>
                  <input
                    type="text"
                    value={kbName}
                    onChange={(e) => setKbName(e.target.value)}
                    placeholder="e.g. Research Papers, Financial Reports"
                    className="rounded-lg border border-white/10 bg-neutral-900 px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
                  />
                </div>

                <div className="flex flex-col space-y-1.5">
                  <label className="text-xs font-medium text-neutral-300">Description (Optional)</label>
                  <textarea
                    rows={2}
                    value={kbDescription}
                    onChange={(e) => setKbDescription(e.target.value)}
                    placeholder="Brief description of documents stored here"
                    className="rounded-lg border border-white/10 bg-neutral-900 px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none resize-none"
                  />
                </div>
              </div>

              {/* Summary of Configuration */}
              <div className="rounded-xl border border-white/10 bg-neutral-900/40 p-4 text-xs">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-neutral-400">
                  Setup Summary
                </span>
                <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="flex flex-col">
                    <span className="text-neutral-500">Operating Mode</span>
                    <span className="font-medium text-neutral-200 capitalize">{mode.replace('_', ' ')}</span>
                  </div>

                  <div className="flex flex-col">
                    <span className="text-neutral-500">Inference LLM</span>
                    <span className="font-medium text-neutral-200">
                      {mode === 'fully_local' ? selectedLocalModel : `${cloudProvider} (${cloudModel})`}
                    </span>
                  </div>

                  <div className="flex flex-col">
                    <span className="text-neutral-500">Vector Embeddings</span>
                    <span className="font-medium text-neutral-200">Sentence Transformers (384-d)</span>
                  </div>

                  <div className="flex flex-col">
                    <span className="text-neutral-500">Storage Backend</span>
                    <span className="font-medium text-neutral-200">Local SQLite & Vector Store</span>
                  </div>
                </div>
              </div>

              {finishError && (
                <div className="rounded-lg border border-red-500/20 bg-red-950/20 p-3 text-xs text-red-300 flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />
                  <span>{finishError}</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Wizard Footer / Navigation Controls */}
        <div className="relative border-t border-white/10 bg-[#0a0a0d] px-8 py-4 flex items-center justify-between">
          <div>
            {onClose && (
              <button
                type="button"
                onClick={onClose}
                className="text-xs text-neutral-500 hover:text-neutral-300 transition-colors"
              >
                Skip for now
              </button>
            )}
          </div>

          <div className="flex items-center gap-3">
            {step > 1 && (
              <button
                type="button"
                onClick={() => setStep((s) => (s - 1) as 1 | 2)}
                disabled={isFinishing}
                className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3.5 py-2 text-xs font-medium text-neutral-300 hover:bg-white/10 transition-colors"
              >
                <ArrowLeft className="h-3.5 w-3.5" />
                Back
              </button>
            )}

            {step < 3 ? (
              <button
                type="button"
                onClick={() => setStep((s) => (s + 1) as 2 | 3)}
                className="inline-flex items-center gap-1.5 rounded-lg bg-blue-600 px-4 py-2 text-xs font-semibold text-white shadow-md shadow-blue-600/30 hover:bg-blue-500 transition-colors"
              >
                Next Step
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            ) : (
              <button
                type="button"
                onClick={handleFinish}
                disabled={isFinishing || !kbName.trim()}
                className="inline-flex items-center gap-1.5 rounded-lg bg-gradient-to-r from-blue-600 to-teal-500 px-5 py-2 text-xs font-semibold text-white shadow-lg shadow-blue-600/30 hover:opacity-95 disabled:opacity-50 transition-opacity"
              >
                {isFinishing ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Initializing Workspace…
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    Complete Setup & Launch
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
