"use client";

import React, { useState } from 'react';
import {
  Sparkles,
  Server,
  Cloud,
  CheckCircle2,
  AlertTriangle,
  Play,
  Loader2,
  Trash2,
  Key,
  Eye,
  EyeOff,
  DownloadCloud,
  Layers,
  Star,
  Zap,
  Globe,
  Sliders,
  Check,
  X,
} from 'lucide-react';
import {
  LLMConfig,
  OllamaStatus,
  LLMConnectionTestResult,
  testLLMConnection,
  createLLMConfig,
  updateLLMConfig,
  deleteLLMConfig,
} from '@/lib/api';
import { fetchEventSource } from '@/lib/api-client';

interface LLMConfigTabProps {
  configs: LLMConfig[];
  ollama: OllamaStatus | null;
  onRefresh: () => void;
}

export const LLM_PROVIDERS = [
  { id: 'ollama', name: 'Ollama (Local)', isLocal: true, defaultUrl: 'http://127.0.0.1:11434' },
  { id: 'gemini', name: 'Google Gemini', isLocal: false, defaultUrl: '' },
  { id: 'openai', name: 'OpenAI', isLocal: false, defaultUrl: '' },
  { id: 'openai_compatible', name: 'OpenAI-Compatible API', isLocal: false, defaultUrl: 'https://api.together.xyz/v1' },
  { id: 'lm_studio', name: 'LM Studio (Local)', isLocal: true, defaultUrl: 'http://localhost:1234/v1' },
  { id: 'llama_cpp', name: 'llama.cpp (Local)', isLocal: true, defaultUrl: 'http://localhost:8080/v1' },
  { id: 'anthropic', name: 'Anthropic Claude', isLocal: false, defaultUrl: '' },
];

export function LLMConfigTab({ configs, ollama, onRefresh }: LLMConfigTabProps) {
  // Form State
  const [configName, setConfigName] = useState('Ollama Local');
  const [providerType, setProviderType] = useState('ollama');
  const [modelName, setModelName] = useState('qwen2.5:7b');
  const [baseUrl, setBaseUrl] = useState('http://127.0.0.1:11434');
  const [apiKey, setApiKey] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [temperature, setTemperature] = useState(0.2);
  const [maxTokens, setMaxTokens] = useState(4096);
  const [isDefault, setIsDefault] = useState(false);

  // Testing & Saving States
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<LLMConnectionTestResult | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Ollama Model Pull State
  const [pullModelInput, setPullModelInput] = useState('');
  const [isPulling, setIsPulling] = useState(false);
  const [pullStatus, setPullStatus] = useState<string | null>(null);
  const [pullPercent, setPullPercent] = useState<number | null>(null);
  const [pullCompletedBytes, setPullCompletedBytes] = useState<number | null>(null);
  const [pullTotalBytes, setPullTotalBytes] = useState<number | null>(null);
  const [pullAbortController, setPullAbortController] = useState<AbortController | null>(null);

  // Active testing ID for items in the saved configs list
  const [listTestingId, setListTestingId] = useState<string | null>(null);
  const [listTestResult, setListTestResult] = useState<{ id: string; res: LLMConnectionTestResult } | null>(null);

  // Update form defaults when provider changes
  const handleProviderChange = (newProvider: string) => {
    setProviderType(newProvider);
    const prov = LLM_PROVIDERS.find((p) => p.id === newProvider);
    if (prov) {
      setBaseUrl(prov.defaultUrl);
      if (newProvider === 'ollama') {
        setConfigName('Ollama Local');
        setModelName(ollama?.models?.[0]?.name || 'qwen2.5:7b');
      } else if (newProvider === 'gemini') {
        setConfigName('Google Gemini');
        setModelName('gemini-1.5-flash');
      } else if (newProvider === 'openai') {
        setConfigName('OpenAI GPT-4o');
        setModelName('gpt-4o-mini');
      } else if (newProvider === 'anthropic') {
        setConfigName('Claude 3.5 Sonnet');
        setModelName('claude-3-5-sonnet-20241022');
      } else if (newProvider === 'lm_studio') {
        setConfigName('LM Studio Server');
        setModelName('local-model');
      } else if (newProvider === 'llama_cpp') {
        setConfigName('llama.cpp Server');
        setModelName('default');
      } else {
        setConfigName('Custom OpenAI API');
        setModelName('meta-llama/Meta-Llama-3.1-8B-Instruct');
      }
    }
    setTestResult(null);
  };

  // Test current configuration
  const handleTest = async () => {
    setIsTesting(true);
    setTestResult(null);
    try {
      const res = await testLLMConnection({
        provider_type: providerType,
        model_name: modelName.trim(),
        base_url: baseUrl.trim() || undefined,
        api_key: apiKey.trim() || undefined,
      });
      setTestResult(res);
    } catch (err) {
      setTestResult({
        success: false,
        latency_ms: 0,
        provider: providerType,
        model: modelName,
        error_message: err instanceof Error ? err.message : 'Connection test failed',
      });
    } finally {
      setIsTesting(false);
    }
  };

  // Save configuration (optionally set as default)
  const handleSave = async (setDefault = false) => {
    setIsSaving(true);
    setSaveSuccess(null);
    setSaveError(null);
    try {
      await createLLMConfig({
        name: configName.trim() || `${providerType} (${modelName})`,
        provider_type: providerType,
        model_name: modelName.trim(),
        base_url: baseUrl.trim() || undefined,
        api_key: apiKey.trim() || undefined,
        is_default: setDefault || isDefault,
      });
      setSaveSuccess(`Configuration "${configName}" saved successfully!`);
      setTimeout(() => setSaveSuccess(null), 3000);
      onRefresh();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save configuration');
    } finally {
      setIsSaving(false);
    }
  };

  // Set existing config as default
  const handleSetDefault = async (cfg: LLMConfig) => {
    try {
      await updateLLMConfig(cfg.id, { is_default: true });
      onRefresh();
    } catch (err) {
      alert('Failed to set default configuration');
    }
  };

  // Delete saved config
  const handleDelete = async (id: string) => {
    if (!window.confirm('Delete this LLM configuration?')) return;
    try {
      await deleteLLMConfig(id);
      onRefresh();
    } catch (err) {
      alert('Failed to delete configuration');
    }
  };

  // Test item in saved list
  const handleTestListItem = async (cfg: LLMConfig) => {
    setListTestingId(cfg.id);
    setListTestResult(null);
    try {
      const res = await testLLMConnection({
        provider_type: cfg.provider_type,
        model_name: cfg.model_name,
        base_url: cfg.base_url,
      });
      setListTestResult({ id: cfg.id, res });
    } catch (err) {
      setListTestResult({
        id: cfg.id,
        res: {
          success: false,
          latency_ms: 0,
          provider: cfg.provider_type,
          model: cfg.model_name,
          error_message: err instanceof Error ? err.message : 'Connection failed',
        },
      });
    } finally {
      setListTestingId(null);
    }
  };

  // Inline Ollama Model Pull
  const handlePullModel = async (targetModel?: string) => {
    const modelToPull = (targetModel || pullModelInput).trim();
    if (!modelToPull) return;

    const controller = new AbortController();
    setPullAbortController(controller);
    setIsPulling(true);
    setPullStatus('Connecting to Ollama pull stream…');
    setPullPercent(0);
    setPullCompletedBytes(0);
    setPullTotalBytes(0);

    try {
      await fetchEventSource('/models/ollama/pull', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_name: modelToPull,
          base_url: baseUrl || 'http://127.0.0.1:11434',
        }),
        signal: controller.signal,
        onMessage: (evt) => {
          try {
            const data = JSON.parse(evt.data);
            if (data.status) setPullStatus(data.status);
            if (typeof data.percent === 'number') setPullPercent(data.percent);
            if (typeof data.completed === 'number') setPullCompletedBytes(data.completed);
            if (typeof data.total === 'number') setPullTotalBytes(data.total);

            if (data.status === 'success') {
              setIsPulling(false);
              setPullStatus(`Model ${modelToPull} successfully pulled!`);
              setPullPercent(100);
              setPullModelInput('');
              onRefresh();
            }
          } catch {
            // Ignore parse errors
          }
        },
        onError: (err) => {
          setPullStatus(`Pull failed: ${err instanceof Error ? err.message : String(err)}`);
          setIsPulling(false);
        },
        onClose: () => {
          setIsPulling(false);
        },
      });
    } catch (err) {
      setPullStatus(`Pull failed: ${err instanceof Error ? err.message : String(err)}`);
      setIsPulling(false);
    }
  };

  const handleAbortPull = () => {
    if (pullAbortController) {
      pullAbortController.abort();
      setPullAbortController(null);
    }
    setIsPulling(false);
    setPullStatus('Pull aborted by user.');
  };

  const formatBytes = (bytes?: number | null) => {
    if (!bytes || bytes <= 0) return '0 MB';
    const mb = bytes / (1024 * 1024);
    if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
    return `${mb.toFixed(0)} MB`;
  };

  return (
    <div className="space-y-8">
      {/* ── Active Editor Form ───────────────────────────────────────────── */}
      <div className="rounded-2xl border border-white/10 bg-black/40 p-6 md:p-8 space-y-6">
        <div>
          <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-blue-400" />
            Configure LLM Provider
          </h2>
          <p className="text-xs text-neutral-400 mt-1">
            Configure private local LLMs or enterprise cloud reasoning models.
          </p>
        </div>

        {/* Provider Selector Chips */}
        <div className="space-y-2">
          <label className="text-xs font-semibold uppercase tracking-wider text-neutral-400">
            Select Provider
          </label>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2.5">
            {LLM_PROVIDERS.map((prov) => {
              const selected = providerType === prov.id;
              return (
                <button
                  key={prov.id}
                  type="button"
                  onClick={() => handleProviderChange(prov.id)}
                  className={`rounded-xl border p-3 text-left transition-all flex flex-col justify-between ${
                    selected
                      ? 'border-blue-500 bg-blue-500/10 text-white shadow-sm shadow-blue-500/10'
                      : 'border-white/10 bg-white/[0.02] text-neutral-400 hover:border-white/20 hover:bg-white/[0.04]'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    {prov.isLocal ? (
                      <Server className={`h-4 w-4 ${selected ? 'text-teal-400' : 'text-neutral-500'}`} />
                    ) : (
                      <Cloud className={`h-4 w-4 ${selected ? 'text-blue-400' : 'text-neutral-500'}`} />
                    )}
                    <span
                      className={`rounded px-1.5 py-0.5 text-[9px] font-semibold ${
                        prov.isLocal
                          ? 'bg-teal-500/10 text-teal-400 border border-teal-500/20'
                          : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                      }`}
                    >
                      {prov.isLocal ? 'Offline' : 'Cloud'}
                    </span>
                  </div>
                  <span className="text-xs font-semibold leading-tight">{prov.name}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Form Inputs Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 pt-2">
          {/* Config Friendly Name */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-neutral-300">Configuration Name</label>
            <input
              type="text"
              value={configName}
              onChange={(e) => setConfigName(e.target.value)}
              placeholder="e.g. Primary Qwen 7B, Gemini Flash Fast"
              className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3.5 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
            />
          </div>

          {/* Model Name & Selection */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-neutral-300">Model Name / Tag</label>
              {providerType === 'ollama' && ollama?.models && ollama.models.length > 0 && (
                <span className="text-[10px] text-teal-400 font-mono">
                  {ollama.models.length} installed models
                </span>
              )}
            </div>

            {providerType === 'ollama' && ollama?.models && ollama.models.length > 0 ? (
              <select
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3.5 py-2 text-xs text-white focus:border-blue-500 focus:outline-none font-mono"
              >
                {ollama.models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name} ({m.parameter_size || 'Unknown size'})
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="e.g. qwen2.5:7b, gemini-1.5-flash, gpt-4o"
                className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3.5 py-2 text-xs text-white focus:border-blue-500 focus:outline-none font-mono"
              />
            )}
          </div>

          {/* Base URL (if custom or local) */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-neutral-300">Base URL / Endpoint</label>
            <input
              type="text"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="e.g. http://127.0.0.1:11434"
              className="w-full rounded-xl border border-white/10 bg-neutral-900 px-3.5 py-2 text-xs text-white focus:border-blue-500 focus:outline-none font-mono"
            />
          </div>

          {/* API Key (if cloud or authenticated endpoint) */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-neutral-300">API Key</label>
              <span className="text-[10px] text-neutral-500">Windows Credential Manager Encrypted</span>
            </div>
            <div className="relative">
              <input
                type={showApiKey ? 'text' : 'password'}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={providerType === 'ollama' ? 'Optional for local Ollama' : 'Enter secret API key'}
                className="w-full rounded-xl border border-white/10 bg-neutral-900 pl-3.5 pr-10 py-2 text-xs text-white focus:border-blue-500 focus:outline-none font-mono"
              />
              <button
                type="button"
                onClick={() => setShowApiKey(!showApiKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-neutral-400 hover:text-white"
              >
                {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>
        </div>

        {/* Sliders: Temperature & Max Tokens */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 pt-2">
          {/* Temperature */}
          <div className="space-y-2 rounded-xl border border-white/5 bg-black/20 p-4">
            <div className="flex items-center justify-between text-xs font-medium text-neutral-300">
              <span>Temperature</span>
              <span className="font-mono text-blue-400">{temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.0}
              max={2.0}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-blue-500"
            />
            <span className="text-[10px] text-neutral-500 block">
              Lower values (0.1–0.3) provide strict grounded factual answers; higher values allow creativity.
            </span>
          </div>

          {/* Max Tokens */}
          <div className="space-y-2 rounded-xl border border-white/5 bg-black/20 p-4">
            <div className="flex items-center justify-between text-xs font-medium text-neutral-300">
              <span>Maximum Response Tokens</span>
              <span className="font-mono text-blue-400">{maxTokens}</span>
            </div>
            <input
              type="range"
              min={256}
              max={16384}
              step={256}
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))}
              className="w-full accent-blue-500"
            />
            <span className="text-[10px] text-neutral-500 block">
              Maximum token limit generated per response turn.
            </span>
          </div>
        </div>

        {/* Inline Ollama Model Pull Section (Only shown when Ollama is selected) */}
        {providerType === 'ollama' && (
          <div className="rounded-xl border border-teal-500/20 bg-teal-950/10 p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <DownloadCloud className="h-4 w-4 text-teal-400" />
                <h3 className="text-xs font-semibold text-white">Pull New Ollama Model</h3>
              </div>
              <span className="text-[10px] text-neutral-400">Streams directly from Ollama library</span>
            </div>

            <div className="flex flex-col sm:flex-row gap-2">
              <input
                type="text"
                value={pullModelInput}
                onChange={(e) => setPullModelInput(e.target.value)}
                placeholder="e.g. qwen2.5:7b, llama3.2:3b, deepseek-r1:7b"
                disabled={isPulling}
                className="flex-1 rounded-lg border border-white/10 bg-neutral-900 px-3 py-1.5 text-xs text-white placeholder-neutral-500 focus:border-teal-500 focus:outline-none font-mono"
              />

              {isPulling ? (
                <button
                  type="button"
                  onClick={handleAbortPull}
                  className="rounded-lg bg-red-600/80 px-3 py-1.5 text-xs font-semibold text-white hover:bg-red-500 transition-colors"
                >
                  Abort Pull
                </button>
              ) : (
                <button
                  type="button"
                  onClick={() => handlePullModel()}
                  disabled={!pullModelInput.trim()}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-teal-600 px-3.5 py-1.5 text-xs font-semibold text-white hover:bg-teal-500 disabled:opacity-50 transition-colors"
                >
                  <DownloadCloud className="h-3.5 w-3.5" />
                  Pull Model
                </button>
              )}
            </div>

            {/* Quick model preset chips */}
            <div className="flex flex-wrap items-center gap-1.5 text-[10px]">
              <span className="text-neutral-500">Quick Picks:</span>
              {['qwen2.5:7b', 'qwen2.5:3b', 'llama3.2:3b', 'mistral:7b', 'deepseek-r1:7b'].map((name) => (
                <button
                  key={name}
                  type="button"
                  disabled={isPulling}
                  onClick={() => {
                    setPullModelInput(name);
                    void handlePullModel(name);
                  }}
                  className="rounded-md border border-white/5 bg-white/5 px-2 py-0.5 font-mono text-neutral-300 hover:border-teal-500/40 hover:bg-teal-500/10 hover:text-teal-300 transition-colors"
                >
                  {name}
                </button>
              ))}
            </div>

            {/* Pull Progress Bar */}
            {isPulling && (
              <div className="space-y-1.5 pt-1">
                <div className="flex items-center justify-between text-[11px] font-mono text-neutral-400">
                  <span className="flex items-center gap-1.5 text-teal-300 truncate max-w-sm">
                    <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0" />
                    {pullStatus}
                  </span>
                  <span>
                    {pullCompletedBytes && pullTotalBytes
                      ? `${formatBytes(pullCompletedBytes)} / ${formatBytes(pullTotalBytes)}`
                      : ''}{' '}
                    {pullPercent !== null ? `(${pullPercent.toFixed(1)}%)` : ''}
                  </span>
                </div>

                <div className="h-2 w-full overflow-hidden rounded-full bg-white/5 p-0.5">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-teal-500 to-blue-500 transition-all duration-300"
                    style={{ width: `${Math.min(100, Math.max(3, pullPercent || 0))}%` }}
                  />
                </div>
              </div>
            )}

            {!isPulling && pullStatus && (
              <p className="text-[11px] text-teal-300 font-mono">{pullStatus}</p>
            )}
          </div>
        )}

        {/* Action Controls & Testing */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-white/5">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleTest}
              disabled={isTesting || !modelName.trim()}
              className="inline-flex items-center gap-1.5 rounded-xl border border-blue-500/30 bg-blue-500/10 px-4 py-2 text-xs font-semibold text-blue-300 hover:bg-blue-500/20 disabled:opacity-50 transition-colors"
            >
              {isTesting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
              Test Connection
            </button>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => handleSave(false)}
              disabled={isSaving || !modelName.trim()}
              className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold text-neutral-200 hover:bg-white/10 disabled:opacity-50 transition-colors"
            >
              Save Configuration
            </button>

            <button
              type="button"
              onClick={() => handleSave(true)}
              disabled={isSaving || !modelName.trim()}
              className="inline-flex items-center gap-1.5 rounded-xl bg-blue-600 px-4 py-2 text-xs font-semibold text-white shadow-md shadow-blue-600/30 hover:bg-blue-500 disabled:opacity-50 transition-colors"
            >
              <Star className="h-3.5 w-3.5" />
              Save & Set Default
            </button>
          </div>
        </div>

        {/* Test Result & Capabilities Badges */}
        {testResult && (
          <div
            className={`rounded-xl border p-4 text-xs space-y-3 ${
              testResult.success
                ? 'border-green-500/20 bg-green-950/20 text-green-300'
                : 'border-red-500/20 bg-red-950/20 text-red-300'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2 font-medium">
                {testResult.success ? (
                  <CheckCircle2 className="h-4 w-4 text-green-400 shrink-0" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />
                )}
                {testResult.success
                  ? `Connection Verified for ${testResult.provider} (${testResult.model})`
                  : testResult.error_message || 'Connection test failed'}
              </span>
              <span className="font-mono text-[11px] opacity-80">{testResult.latency_ms.toFixed(0)}ms latency</span>
            </div>

            {/* Capabilities Badges */}
            {testResult.capabilities && (
              <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-green-500/20">
                <span className="rounded bg-black/40 px-2 py-0.5 text-[10px] font-mono text-neutral-300 border border-white/5">
                  Context: {testResult.capabilities.max_context_window.toLocaleString()} tokens
                </span>
                <span
                  className={`rounded px-2 py-0.5 text-[10px] font-medium border ${
                    testResult.capabilities.supports_streaming
                      ? 'bg-teal-500/20 text-teal-300 border-teal-500/30'
                      : 'bg-white/5 text-neutral-500 border-white/5'
                  }`}
                >
                  Streaming: {testResult.capabilities.supports_streaming ? 'Yes' : 'No'}
                </span>
                <span
                  className={`rounded px-2 py-0.5 text-[10px] font-medium border ${
                    testResult.capabilities.supports_function_calling
                      ? 'bg-blue-500/20 text-blue-300 border-blue-500/30'
                      : 'bg-white/5 text-neutral-500 border-white/5'
                  }`}
                >
                  Tool Calling: {testResult.capabilities.supports_function_calling ? 'Supported' : 'No'}
                </span>
                <span
                  className={`rounded px-2 py-0.5 text-[10px] font-medium border ${
                    testResult.capabilities.is_local
                      ? 'bg-purple-500/20 text-purple-300 border-purple-500/30'
                      : 'bg-white/5 text-neutral-500 border-white/5'
                  }`}
                >
                  Offline Ready: {testResult.capabilities.is_local ? '100% Local' : 'Cloud'}
                </span>
              </div>
            )}
          </div>
        )}

        {saveSuccess && (
          <div className="rounded-xl border border-green-500/20 bg-green-950/20 p-3 text-xs text-green-300 flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-green-400" />
            {saveSuccess}
          </div>
        )}

        {saveError && (
          <div className="rounded-xl border border-red-500/20 bg-red-950/20 p-3 text-xs text-red-300 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-red-400" />
            {saveError}
          </div>
        )}
      </div>

      {/* ── Saved LLM Configurations List ─────────────────────────────────── */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white">Saved LLM Configurations</h2>
          <span className="text-xs text-neutral-500 font-mono">{configs.length} configured</span>
        </div>

        <div className="overflow-hidden rounded-xl border border-white/10 bg-black/30">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-white/5 bg-white/[0.02] text-neutral-400">
              <tr>
                <th className="px-4 py-3 font-medium">Name</th>
                <th className="px-4 py-3 font-medium">Provider</th>
                <th className="px-4 py-3 font-medium">Model</th>
                <th className="px-4 py-3 font-medium">Context Window</th>
                <th className="px-4 py-3 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-neutral-300">
              {configs.map((cfg) => (
                <tr key={cfg.id} className="hover:bg-white/[0.02] transition-colors">
                  <td className="px-4 py-3 font-medium text-white flex items-center gap-2">
                    <span>{cfg.name}</span>
                    {cfg.is_default && (
                      <span className="rounded bg-teal-500/10 px-1.5 py-0.5 text-[9px] font-semibold text-teal-400 border border-teal-500/20">
                        Default
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 font-mono uppercase text-neutral-400">{cfg.provider_type}</td>
                  <td className="px-4 py-3 font-mono text-neutral-300 truncate max-w-xs">{cfg.model_name}</td>
                  <td className="px-4 py-3 font-mono text-neutral-400">
                    {cfg.context_window ? `${cfg.context_window.toLocaleString()} tok` : 'Dynamic'}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex items-center justify-end gap-2">
                      {!cfg.is_default && (
                        <button
                          type="button"
                          onClick={() => handleSetDefault(cfg)}
                          className="rounded px-2 py-1 text-[11px] font-medium text-neutral-400 hover:text-teal-300 hover:bg-white/5 transition-colors"
                        >
                          Make Default
                        </button>
                      )}

                      <button
                        type="button"
                        onClick={() => handleTestListItem(cfg)}
                        disabled={listTestingId === cfg.id}
                        className="p-1 text-blue-400 hover:text-blue-300 transition-colors"
                        title="Test Connection"
                      >
                        {listTestingId === cfg.id ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <Play className="h-3.5 w-3.5" />
                        )}
                      </button>

                      <button
                        type="button"
                        onClick={() => handleDelete(cfg.id)}
                        className="p-1 text-red-400/80 hover:text-red-400 transition-colors"
                        title="Delete"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {listTestResult && (
          <div
            className={`rounded-xl border p-3 text-xs ${
              listTestResult.res.success
                ? 'border-green-500/20 bg-green-950/20 text-green-300'
                : 'border-red-500/20 bg-red-950/20 text-red-300'
            }`}
          >
            {listTestResult.res.success
              ? `Connection to ${listTestResult.res.model} verified (${listTestResult.res.latency_ms.toFixed(0)}ms latency).`
              : listTestResult.res.error_message || 'Connection test failed.'}
          </div>
        )}
      </div>
    </div>
  );
}
