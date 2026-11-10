"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import axios from 'axios';
import { createSession, getCurrentUser, errorMessage } from '@/lib/api';

export default function Home() {
  const router = useRouter();
  const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    async function open() {
      try {
        await getCurrentUser();
        if (!active) return;
        const session = await createSession([]);
        if (active) router.replace(`/chat/${session.session_id}`);
      } catch (error) {
        if (!active) return;
        if (axios.isAxiosError(error) && error.response?.status === 401) router.replace('/login');
        else setError(errorMessage(error));
      }
    }
    void open();
    return () => { active = false; };
  }, [router]);
  return <main className="min-h-screen flex flex-col items-center justify-center gap-4 text-neutral-300">
    <h1 className="text-2xl font-bold">Quantiva</h1>
    {error ? <><p role="alert">{error}</p><button onClick={() => window.location.reload()} className="text-blue-400">Retry</button></> : <p role="status">Opening your workspace…</p>}
  </main>;
}
