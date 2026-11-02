import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Quantiva — Enterprise AI Document Agent",
  description:
    "Upload documents, ask questions, get cited answers. Quantiva is a production-grade RAG platform with hybrid search, agentic orchestration, and real-time streaming.",
  keywords: ["RAG", "AI", "document intelligence", "LLM", "enterprise"],
  authors: [{ name: "Quantiva" }],
  openGraph: {
    title: "Quantiva — Enterprise AI Document Agent",
    description: "Production-grade document intelligence powered by hybrid RAG.",
    type: "website",
  },
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="h-full overflow-hidden bg-[#0a0a0a]">{children}</body>
    </html>
  );
}
