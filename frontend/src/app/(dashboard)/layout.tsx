"use client";

import { ReactNode } from 'react';
import { BackendStartupGate } from '@/components/system/BackendStartupGate';
import { AppShell } from '@/components/layout/AppShell';

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <BackendStartupGate>
      <AppShell>{children}</AppShell>
    </BackendStartupGate>
  );
}
