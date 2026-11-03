"use client";

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { createSession } from '@/lib/api';
import { Sparkles } from 'lucide-react';

const STEPS = [
  'Connecting to backend…',
  'Initializing agent…',
  'Creating session…',
  'Ready',
];

export default function Home() {
  const router = useRouter();
  const [step, setStep] = useState(0);

  useEffect(() => {
    let interval: NodeJS.Timeout;

    // Animate through steps while waiting
    interval = setInterval(() => {
      setStep((s) => Math.min(s + 1, STEPS.length - 2));
    }, 600);

    createSession([])
      .then((res) => {
        clearInterval(interval);
        setStep(STEPS.length - 1);
        router.push(`/chat/${res.session_id}`);
      })
      .catch(() => {
        clearInterval(interval);
        // Graceful degradation — fallback session for local dev without backend
        const fallbackId = `fallback-session-${Date.now()}`;
        router.push(`/chat/${fallbackId}`);
      });

    return () => clearInterval(interval);
  }, [router]);

  return (
    <div className="h-screen w-screen flex flex-col items-center justify-center bg-[#0a0a0a] gap-8">
      {/* Animated logo */}
      <div className="relative">
        <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center shadow-2xl shadow-blue-900/40">
          <Sparkles className="w-10 h-10 text-white" />
        </div>
        {/* Pulsing ring */}
        <div className="absolute inset-0 rounded-3xl border-2 border-blue-500/30 animate-ping" />
      </div>

      <div className="text-center">
        <h1 className="text-2xl font-bold text-gradient-brand mb-2">Quantiva</h1>
        <p className="text-sm text-neutral-500">Enterprise AI Document Agent</p>
      </div>

      {/* Step indicator */}
      <div className="flex flex-col items-center gap-3">
        <div className="flex gap-1.5">
          {STEPS.map((_, i) => (
            <div
              key={i}
              className="h-1 rounded-full transition-all duration-500"
              style={{
                width: i <= step ? '24px' : '8px',
                background:
                  i < step
                    ? '#34d399'
                    : i === step
                    ? '#60a5fa'
                    : '#262626',
              }}
            />
          ))}
        </div>
        <p className="text-xs text-neutral-500 min-h-[16px] transition-all">
          {STEPS[Math.min(step, STEPS.length - 1)]}
        </p>
      </div>
    </div>
  );
}
