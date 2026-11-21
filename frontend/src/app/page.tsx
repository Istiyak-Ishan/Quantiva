"use client";

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { isTauriDesktop } from '@/lib/api-client';

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    // In desktop or web mode, route directly to the main dashboard
    router.replace('/dashboard/');
  }, [router]);

  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-4 bg-[#070709] text-neutral-400">
      <div className="flex items-center gap-3">
        <span className="h-4 w-4 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
        <p className="text-sm">Opening Quantiva workspace…</p>
      </div>
    </main>
  );
}
