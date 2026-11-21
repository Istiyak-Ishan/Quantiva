"use client";

import { ReactNode, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';
import { getCurrentUser } from '@/lib/api';
import { isTauriDesktop } from '@/lib/api-client';

export default function DashboardLayout({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>(() =>
    isTauriDesktop() ? 'ready' : 'loading',
  );
  const router = useRouter();

  useEffect(() => {
    if (isTauriDesktop()) return;

    let active = true;
    getCurrentUser()
      .then(() => {
        if (active) setStatus('ready');
      })
      .catch((error) => {
        if (!active) return;
        if (axios.isAxiosError(error) && error.response?.status === 401) {
          router.replace('/login/');
        } else {
          setStatus('error');
        }
      });

    return () => {
      active = false;
    };
  }, [router]);

  if (status !== 'ready') {
    return (
      <main className="p-8 text-neutral-300" role="status">
        {status === 'error'
          ? 'Unable to verify your session. Please reload to retry.'
          : 'Verifying your session…'}
      </main>
    );
  }

  return children;
}
