"use client";

import React, { useState, useEffect, useCallback, useTransition } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  Database,
  Files,
  MessageSquare,
  Sparkles,
  Sliders,
  Workflow,
  BarChart3,
  Archive,
  Settings,
  ChevronLeft,
  ChevronRight,
  ShieldCheck,
  Server,
  Cloud,
  CheckCircle2,
  HardDrive,
  Loader2,
  Wifi,
  WifiOff,
  FolderOpen,
  Wand2,
} from 'lucide-react';
import {
  getStorageStatus,
  getKnowledgeBases,
  getLLMConfigs,
  StorageStatus,
  LLMConfig,
} from '@/lib/api';
import { invokeTauriCommand, isTauriDesktop } from '@/lib/api-client';
import { useDocumentStore } from '@/store/useDocumentStore';
import { FirstLaunchWizard, OperatingMode } from '@/components/onboarding/FirstLaunchWizard';

interface AppShellProps {
  children: React.ReactNode;
}

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
}

const NAV_ITEMS: NavItem[] = [
  { name: 'Dashboard', href: '/dashboard/', icon: LayoutDashboard },
  { name: 'Knowledge Bases', href: '/knowledge-bases/', icon: Database },
  { name: 'Document Manager', href: '/documents/', icon: Files },
  { name: 'AI Chat', href: '/chat/', icon: MessageSquare },
  { name: 'Model Studio', href: '/models/', icon: Sparkles },
  { name: 'Retrieval Settings', href: '/settings/retrieval/', icon: Sliders },
  { name: 'Agent Settings', href: '/settings/agent/', icon: Workflow },
  { name: 'Evaluation', href: '/eval/', icon: BarChart3 },
  { name: 'Backup & Restore', href: '/settings/backup/', icon: Archive },
  { name: 'System Settings', href: '/settings/system/', icon: Settings },
];

export function AppShell({ children }: AppShellProps) {
  const pathname = usePathname();
  const [, startTransition] = useTransition();

  // Sidebar collapse state
  const [collapsed, setCollapsed] = useState(false);

  // Status bar states
  const [isOnline, setIsOnline] = useState(true);
  const [storage, setStorage] = useState<StorageStatus | null>(null);
  const [operatingMode, setOperatingMode] = useState<OperatingMode>('fully_local');
  const [activeLLM, setActiveLLM] = useState<string>('Local Model');

  // Onboarding wizard modal state
  const [showWizard, setShowWizard] = useState(false);

  // Background jobs from document store
  const documents = useDocumentStore((state) => state.documents);
  const activeJobsCount = documents.filter(
    (d) => d.status === 'processing' || d.status === 'uploading'
  ).length;

  // Track online/offline status
  useEffect(() => {
    setIsOnline(typeof navigator !== 'undefined' ? navigator.onLine : true);
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Hydrate sidebar state from localStorage
  useEffect(() => {
    try {
      const savedCollapsed = localStorage.getItem('quantiva_sidebar_collapsed');
      if (savedCollapsed !== null) {
        setCollapsed(savedCollapsed === 'true');
      }
      const savedMode = localStorage.getItem('quantiva_operating_mode') as OperatingMode | null;
      if (savedMode) {
        setOperatingMode(savedMode);
      }
    } catch {
      // Ignore localStorage errors
    }
  }, []);

  const toggleSidebar = () => {
    setCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem('quantiva_sidebar_collapsed', String(next));
      } catch {
        // Ignore
      }
      return next;
    });
  };

  // Check first-launch onboarding & initial system data
  const loadSystemInfo = useCallback(async () => {
    try {
      // 1. Fetch Storage Info
      const st = await getStorageStatus();
      setStorage(st);
    } catch {
      // Non-fatal
    }

    try {
      // 2. Fetch Active Default LLM
      const configs = await getLLMConfigs();
      const defaultLLM = configs.find((c: LLMConfig) => c.is_default);
      if (defaultLLM) {
        setActiveLLM(defaultLLM.model_name || defaultLLM.name);
      }
    } catch {
      // Non-fatal
    }

    try {
      // 3. Check if First-Launch Wizard is needed
      const completed = localStorage.getItem('quantiva_onboarding_completed');
      if (completed !== 'true') {
        const kbs = await getKnowledgeBases();
        if (kbs.length === 0) {
          setShowWizard(true);
        }
      }
    } catch {
      // Non-fatal
    }
  }, []);

  useEffect(() => {
    void loadSystemInfo();
  }, [loadSystemInfo]);

  // Handle clicking storage path to open in Windows Explorer
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

  // Format free disk space
  const formatFreeDisk = (bytes?: number) => {
    if (!bytes || bytes <= 0) return 'Local Storage';
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(1)} GB free`;
  };

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-[#070709] text-neutral-200 select-none">
      {/* ── Main Viewport: Sidebar + Content Area ─────────────────────────── */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Global Sidebar */}
        <aside
          className={`relative flex flex-col border-r border-white/5 bg-[#0a0a0d] transition-all duration-300 ease-in-out z-20 ${
            collapsed ? 'w-16' : 'w-60'
          }`}
          aria-label="Application sections"
        >
          {/* Logo Header */}
          <div className="flex h-14 items-center justify-between px-3 border-b border-white/5">
            <Link
              href="/dashboard/"
              className="flex items-center gap-3 overflow-hidden rounded-lg p-1 hover:bg-white/5 transition-colors"
            >
              <div className="relative flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-tr from-blue-600 to-teal-400 p-[1px] shadow-sm shadow-blue-500/20">
                <div className="flex h-full w-full items-center justify-center rounded-lg bg-[#0c0c10]">
                  <Sparkles className="h-4 w-4 text-blue-400" />
                </div>
              </div>
              {!collapsed && (
                <div className="flex flex-col min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="font-bold text-sm text-white tracking-tight leading-none">Quantiva</span>
                    <span className="rounded bg-blue-500/10 px-1 py-0.2 text-[9px] font-semibold text-blue-400 border border-blue-500/20">
                      DESKTOP
                    </span>
                  </div>
                  <span className="text-[10px] text-neutral-400 truncate mt-0.5">Document Intelligence</span>
                </div>
              )}
            </Link>

            {/* Collapse / Expand Toggle Button */}
            {!collapsed && (
              <button
                type="button"
                onClick={toggleSidebar}
                className="flex h-6 w-6 items-center justify-center rounded-md text-neutral-400 hover:bg-white/5 hover:text-white transition-colors"
                title="Collapse sidebar"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
            )}
          </div>

          {/* Nav Items List */}
          <nav className="flex-1 overflow-y-auto px-2 py-3 space-y-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive =
                item.href === '/dashboard/'
                  ? pathname === '/dashboard/' || pathname === '/'
                  : item.href === '/models/'
                  ? pathname?.startsWith('/models') || pathname?.startsWith('/model-studio')
                  : pathname?.startsWith(item.href.replace(/\/$/, ''));

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  title={collapsed ? item.name : undefined}
                  className={`group relative flex items-center rounded-lg px-2.5 py-2 text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-blue-600/15 text-blue-400 font-semibold border border-blue-500/20 shadow-sm shadow-blue-500/10'
                      : 'text-neutral-400 hover:bg-white/5 hover:text-neutral-200'
                  }`}
                >
                  <Icon
                    className={`h-4 w-4 shrink-0 transition-colors ${
                      isActive ? 'text-blue-400' : 'text-neutral-400 group-hover:text-neutral-200'
                    }`}
                  />

                  {!collapsed && (
                    <span className="ml-3 truncate">{item.name}</span>
                  )}

                  {/* Active Indicator Bar */}
                  {isActive && (
                    <span className="absolute left-0 top-1/2 -translate-y-1/2 h-5 w-1 rounded-r bg-blue-500" />
                  )}
                </Link>
              );
            })}
          </nav>

          {/* Sidebar Footer */}
          <div className="p-2 border-t border-white/5">
            {/* Setup Wizard Launcher */}
            <button
              type="button"
              onClick={() => setShowWizard(true)}
              title="Setup Wizard"
              className={`w-full flex items-center rounded-lg p-2 text-xs font-medium text-neutral-400 hover:bg-white/5 hover:text-white transition-colors ${
                collapsed ? 'justify-center' : 'gap-2.5'
              }`}
            >
              <Wand2 className="h-4 w-4 text-purple-400 shrink-0" />
              {!collapsed && <span>Setup Wizard</span>}
            </button>

            {/* When collapsed, toggle button at bottom */}
            {collapsed && (
              <button
                type="button"
                onClick={toggleSidebar}
                className="mt-1 flex h-8 w-full items-center justify-center rounded-md text-neutral-400 hover:bg-white/5 hover:text-white transition-colors"
                title="Expand sidebar"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            )}
          </div>
        </aside>

        {/* Dynamic Page Content Viewport */}
        <main className="flex-1 min-w-0 h-full overflow-hidden bg-[#070709] relative">
          {children}
        </main>
      </div>

      {/* ── Bottom Status Bar ─────────────────────────────────────────────── */}
      <footer
        className="flex h-7 w-full shrink-0 items-center justify-between border-t border-white/5 bg-[#09090c] px-3 text-[11px] text-neutral-400 font-mono select-none z-30"
        role="contentinfo"
        aria-label="Desktop system status"
      >
        {/* Left Status Group */}
        <div className="flex items-center gap-3">
          {/* Operating Mode Badge */}
          <div className="flex items-center gap-1.5 font-sans font-medium">
            {operatingMode === 'fully_local' ? (
              <span className="inline-flex items-center gap-1 rounded bg-teal-500/10 px-1.5 py-0.5 text-[10px] text-teal-400 border border-teal-500/20">
                <ShieldCheck className="h-3 w-3" />
                Fully Local
              </span>
            ) : operatingMode === 'hybrid' ? (
              <span className="inline-flex items-center gap-1 rounded bg-purple-500/10 px-1.5 py-0.5 text-[10px] text-purple-400 border border-purple-500/20">
                <Server className="h-3 w-3" />
                Hybrid Mode
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 rounded bg-blue-500/10 px-1.5 py-0.5 text-[10px] text-blue-400 border border-blue-500/20">
                <Cloud className="h-3 w-3" />
                Cloud API
              </span>
            )}
          </div>

          <div className="h-3 w-px bg-white/10" />

          {/* Online / Offline indicator */}
          <div className="flex items-center gap-1.5 text-[10px]">
            {isOnline ? (
              <span className="inline-flex items-center gap-1 text-green-400">
                <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
                Online
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-amber-400">
                <WifiOff className="h-3 w-3" />
                Offline (Local Only)
              </span>
            )}
          </div>

          <div className="h-3 w-px bg-white/10" />

          {/* Active LLM Badge */}
          <Link
            href="/models/"
            className="flex items-center gap-1 text-[10px] text-neutral-400 hover:text-white transition-colors"
            title="Click to manage models in Model Studio"
          >
            <Sparkles className="h-3 w-3 text-blue-400" />
            <span className="truncate max-w-[150px]">{activeLLM}</span>
          </Link>
        </div>

        {/* Right Status Group */}
        <div className="flex items-center gap-3">
          {/* Background Jobs Indicator */}
          <div className="flex items-center gap-1.5 text-[10px]">
            {activeJobsCount > 0 ? (
              <span className="inline-flex items-center gap-1 text-blue-400">
                <Loader2 className="h-3 w-3 animate-spin text-blue-400" />
                {activeJobsCount} {activeJobsCount === 1 ? 'job' : 'jobs'} running
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 text-neutral-400">
                <CheckCircle2 className="h-3 w-3 text-neutral-400" />
                Idle
              </span>
            )}
          </div>

          <div className="h-3 w-px bg-white/10" />

          {/* Storage Path Indicator with click to open in Explorer */}
          <button
            type="button"
            onClick={handleOpenStorage}
            title={
              storage?.current_data_dir
                ? `${storage.current_data_dir} (Click to open in Explorer)`
                : 'Local Data Directory'
            }
            className="inline-flex items-center gap-1.5 rounded px-1.5 py-0.5 text-[10px] text-neutral-400 hover:bg-white/5 hover:text-neutral-200 transition-colors"
          >
            <HardDrive className="h-3 w-3 text-neutral-400" />
            <span className="truncate max-w-[180px]">
              {storage ? formatFreeDisk(storage.disk_free_bytes) : 'Local Storage'}
            </span>
            <FolderOpen className="h-3 w-3 text-neutral-400 opacity-60" />
          </button>
        </div>
      </footer>

      {/* ── First-Launch Setup Wizard Modal ───────────────────────────────── */}
      <FirstLaunchWizard
        isOpen={showWizard}
        onComplete={() => {
          setShowWizard(false);
          void loadSystemInfo();
        }}
        onClose={() => setShowWizard(false)}
      />
    </div>
  );
}
