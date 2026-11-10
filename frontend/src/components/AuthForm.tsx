"use client";

import { FormEvent, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { authenticate, errorMessage } from '@/lib/api';

export function AuthForm({ mode }: { mode: 'login' | 'register' }) {
  const router = useRouter();
  const [error, setError] = useState('');
  const [pending, setPending] = useState(false);
  const title = mode === 'login' ? 'Sign in' : 'Create account';

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setPending(true);
    setError('');
    try {
      await authenticate(mode, String(form.get('email')), String(form.get('password')));
      router.replace('/');
    } catch (error) {
      setError(errorMessage(error));
    } finally {
      setPending(false);
    }
  }

  return (
    <main className="min-h-screen flex items-center justify-center p-6 bg-[#0a0a0a]">
      <form onSubmit={submit} className="glass rounded-2xl p-8 w-full max-w-md flex flex-col gap-5">
        <h1 className="text-2xl font-bold text-neutral-100">Quantiva · {title}</h1>
        <label className="text-sm text-neutral-300">Email
          <input name="email" type="email" autoComplete="email" required className="mt-2 p-3 w-full rounded-lg bg-neutral-900 border border-neutral-700" />
        </label>
        <label className="text-sm text-neutral-300">Password
          <input name="password" type="password" minLength={mode === 'register' ? 8 : 1} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} required className="mt-2 p-3 w-full rounded-lg bg-neutral-900 border border-neutral-700" />
        </label>
        {mode === 'register' && <p className="text-xs text-neutral-400">Use at least 8 characters (maximum 72 UTF-8 bytes).</p>}
        {error && <p role="alert" className="text-sm text-red-400">{error}</p>}
        <button disabled={pending} className="rounded-lg bg-blue-600 p-3 text-white disabled:opacity-50">{pending ? 'Please wait…' : title}</button>
        <Link className="text-sm text-blue-400" href={mode === 'login' ? '/register' : '/login'}>
          {mode === 'login' ? 'Create an account' : 'Already registered? Sign in'}
        </Link>
      </form>
    </main>
  );
}
