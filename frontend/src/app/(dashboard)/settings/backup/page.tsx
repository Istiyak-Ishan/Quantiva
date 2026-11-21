"use client";

import React, { useState } from 'react';
import { Archive, Download, Upload, HardDrive, CheckCircle2, AlertTriangle, Loader2 } from 'lucide-react';
import { api } from '@/lib/api';

export default function BackupRestorePage() {
  const [creatingBackup, setCreatingBackup] = useState(false);
  const [backupMessage, setBackupMessage] = useState<string | null>(null);

  const handleCreateBackup = async () => {
    setCreatingBackup(true);
    setBackupMessage(null);
    try {
      const res = await api.post<{ backup_file: string }>('/settings/backup');
      setBackupMessage(`Backup successfully generated: ${res.data.backup_file}`);
    } catch (err) {
      setBackupMessage('Failed to create backup. Please check disk permissions.');
    } finally {
      setCreatingBackup(false);
    }
  };

  return (
    <div className="h-full w-full overflow-y-auto p-6 md:p-8 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-xl md:text-2xl font-bold text-white tracking-tight">Backup & Restore</h1>
        <p className="text-xs text-neutral-400 mt-1">
          Export full self-contained zip archives of SQLite databases, vector indexes, and configuration.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl">
        {/* Create Backup */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-6 flex flex-col justify-between space-y-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-white">
              <Download className="h-4 w-4 text-blue-400" />
              <h2 className="text-sm font-semibold">Create Full Desktop Backup</h2>
            </div>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Generates a single compressed archive containing your SQLite database, WAL journal, vector store collections,
              and locked index metadata.
            </p>
          </div>

          <div className="pt-2">
            <button
              type="button"
              onClick={handleCreateBackup}
              disabled={creatingBackup}
              className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-4 py-2 text-xs font-semibold text-white shadow-md shadow-blue-600/30 hover:bg-blue-500 disabled:opacity-50 transition-colors"
            >
              {creatingBackup ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Archive className="h-3.5 w-3.5" />}
              {creatingBackup ? 'Packaging Backup…' : 'Export Backup (.zip)'}
            </button>
          </div>
        </div>

        {/* Restore Backup */}
        <div className="rounded-xl border border-white/10 bg-black/30 p-6 flex flex-col justify-between space-y-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-white">
              <Upload className="h-4 w-4 text-teal-400" />
              <h2 className="text-sm font-semibold">Restore From Archive</h2>
            </div>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Restore a previous Quantiva archive. Validates database schema integrity and remounts vector embeddings
              cleanly.
            </p>
          </div>

          <div className="pt-2">
            <label className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-xs font-semibold text-neutral-300 hover:bg-white/10 transition-colors cursor-pointer">
              <Upload className="h-3.5 w-3.5" />
              Select Archive to Restore…
              <input type="file" accept=".zip" className="hidden" />
            </label>
          </div>
        </div>
      </div>

      {backupMessage && (
        <div className="rounded-xl border border-blue-500/20 bg-blue-950/20 p-4 text-xs text-blue-300 max-w-4xl">
          {backupMessage}
        </div>
      )}
    </div>
  );
}
