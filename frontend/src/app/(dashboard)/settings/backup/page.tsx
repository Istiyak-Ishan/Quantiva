"use client";

import React, { useEffect, useState } from 'react';
import {
  Archive,
  Download,
  Upload,
  HardDrive,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  Clock,
  FileCheck,
  RefreshCw,
  FolderOpen,
} from 'lucide-react';
import {
  listWorkspaceBackups,
  createWorkspaceBackup,
  restoreWorkspaceBackup,
  BackupItem,
} from '@/lib/api';

export default function BackupRestorePage() {
  const [backups, setBackups] = useState<BackupItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [creatingBackup, setCreatingBackup] = useState(false);
  const [includeModels, setIncludeModels] = useState(false);
  const [restoringPath, setRestoringPath] = useState<string | null>(null);
  const [backupMessage, setBackupMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const fetchBackups = async () => {
    setLoading(true);
    try {
      const data = await listWorkspaceBackups();
      setBackups(data);
    } catch (err) {
      console.error('Failed to list backups:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchBackups();
  }, []);

  const handleCreateBackup = async () => {
    setCreatingBackup(true);
    setBackupMessage(null);
    try {
      const res = await createWorkspaceBackup(includeModels);
      setBackupMessage({
        type: 'success',
        text: `Backup created successfully! Archive: ${res.filename} (${(res.size_bytes / (1024 * 1024)).toFixed(2)} MB)`,
      });
      await fetchBackups();
    } catch (err) {
      setBackupMessage({
        type: 'error',
        text: 'Failed to create backup. Please verify disk write permissions.',
      });
    } finally {
      setCreatingBackup(false);
    }
  };

  const handleRestore = async (archivePath: string) => {
    if (
      !window.confirm(
        `Are you sure you want to restore from this backup archive?\n${archivePath}\n\nQuantiva will create a safety snapshot before restoring.`
      )
    ) {
      return;
    }

    setRestoringPath(archivePath);
    setBackupMessage(null);
    try {
      const res = await restoreWorkspaceBackup(archivePath, true);
      setBackupMessage({
        type: 'success',
        text: `Workspace successfully restored from archive! Status: ${res.status}`,
      });
    } catch (err) {
      setBackupMessage({
        type: 'error',
        text: 'Failed to restore backup archive. Check archive schema integrity.',
      });
    } finally {
      setRestoringPath(null);
    }
  };

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">
            Backup & Restore
          </h1>
          <p className="text-xs text-neutral-400 mt-1">
            Export full self-contained zip archives of SQLite databases, vector indexes, and configuration.
          </p>
        </div>

        <button
          type="button"
          onClick={fetchBackups}
          className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/5 text-neutral-400 hover:text-white transition-colors"
          title="Refresh Backups List"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {backupMessage && (
        <div
          className={`rounded-xl border p-4 text-xs max-w-4xl ${
            backupMessage.type === 'success'
              ? 'border-emerald-500/20 bg-emerald-950/20 text-emerald-300'
              : 'border-red-500/20 bg-red-950/20 text-red-300'
          }`}
        >
          {backupMessage.text}
        </div>
      )}

      {/* Backup Creation Card */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl">
        <div className="rounded-xl border border-white/10 bg-black/30 p-6 flex flex-col justify-between space-y-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-white">
              <Download className="h-4 w-4 text-blue-400" />
              <h2 className="text-sm font-semibold">Create Full Desktop Backup</h2>
            </div>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Generates a single compressed .zip archive containing your SQLite database,
              WAL journal, vector store collections, and locked index metadata.
            </p>

            <div className="pt-2">
              <label className="flex items-center gap-2 text-xs text-neutral-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeModels}
                  onChange={(e) => setIncludeModels(e.target.checked)}
                  className="rounded accent-blue-600 h-3.5 w-3.5"
                />
                <span>Include downloaded local model weights (increases archive size)</span>
              </label>
            </div>
          </div>

          <div className="pt-2">
            <button
              type="button"
              onClick={handleCreateBackup}
              disabled={creatingBackup}
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-xs font-semibold text-white shadow-md shadow-blue-600/30 hover:bg-blue-500 disabled:opacity-50 transition-colors"
            >
              {creatingBackup ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Archive className="h-3.5 w-3.5" />
              )}
              <span>{creatingBackup ? 'Packaging Backup (.zip)…' : 'Export Backup (.zip)'}</span>
            </button>
          </div>
        </div>

        {/* Safety Note Card */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-6 flex flex-col justify-between space-y-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-white">
              <FileCheck className="h-4 w-4 text-teal-400" />
              <h2 className="text-sm font-semibold">Atomic Safety Snapshots</h2>
            </div>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Before restoring any archive, Quantiva automatically creates an immutable safety
              snapshot of your current database state to prevent accidental data loss.
            </p>
          </div>

          <div className="pt-2">
            <div className="inline-flex items-center gap-2 text-xs text-neutral-400">
              <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
              <span>Zero-loss transaction safety enabled</span>
            </div>
          </div>
        </div>
      </div>

      {/* Available Backups Table */}
      <div className="rounded-xl border border-white/10 bg-black/30 p-6 space-y-4 max-w-4xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-white">
            <HardDrive className="h-4 w-4 text-purple-400" />
            <h2 className="text-sm font-semibold">Available Workspace Backups</h2>
          </div>
          <span className="text-xs text-neutral-500">{backups.length} archives detected</span>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-8 text-neutral-500 gap-2 text-xs">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span>Scanning backups directory…</span>
          </div>
        ) : backups.length === 0 ? (
          <div className="py-8 text-center text-xs text-neutral-500">
            No backup archives found. Click &quot;Export Backup&quot; to generate your first backup.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-white/10 text-neutral-400">
                  <th className="pb-3 font-semibold">Filename</th>
                  <th className="pb-3 font-semibold">Size</th>
                  <th className="pb-3 font-semibold">Created Date</th>
                  <th className="pb-3 font-semibold">App / Schema</th>
                  <th className="pb-3 font-semibold text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {backups.map((b) => (
                  <tr key={b.path} className="hover:bg-white/[0.02] transition-colors">
                    <td className="py-3 font-mono font-medium text-white truncate max-w-[200px]" title={b.filename}>
                      {b.filename}
                    </td>
                    <td className="py-3 text-neutral-400 font-mono">
                      {(b.size_bytes / (1024 * 1024)).toFixed(2)} MB
                    </td>
                    <td className="py-3 text-neutral-400">
                      {new Date(b.created_at).toLocaleString()}
                    </td>
                    <td className="py-3 text-neutral-400 font-mono">
                      v{b.app_version || '1.0.0'} (s{b.schema_version ?? 1})
                    </td>
                    <td className="py-3 text-right">
                      <button
                        type="button"
                        onClick={() => void handleRestore(b.path)}
                        disabled={restoringPath === b.path}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-teal-500/30 bg-teal-500/10 px-2.5 py-1 text-xs font-semibold text-teal-300 hover:bg-teal-500/20 disabled:opacity-50 transition-colors"
                      >
                        {restoringPath === b.path ? (
                          <Loader2 className="h-3 w-3 animate-spin" />
                        ) : (
                          <Upload className="h-3 w-3" />
                        )}
                        <span>{restoringPath === b.path ? 'Restoring…' : 'Restore'}</span>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
