"use client";

import LiveFeed from "@/components/LiveFeed";
import ViolationLog from "@/components/ViolationLog";
import { useEffect, useState } from "react";

export default function Home() {
  const [stats, setStats] = useState({
    active_tracks: 0,
    incident_rate: "0%",
    confidence: "0%"
  });

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch('http://localhost:8000/stats');
        const data = await res.json();
        setStats(data);
      } catch (err) {
        console.error("Failed to fetch stats", err);
      }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <main className="min-h-screen bg-[#0a0a0a] text-white p-8 font-sans selection:bg-blue-500/30">
      {/* Background decoration */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden pointer-events-none -z-10">
        <div className="absolute top-[10%] left-[10%] w-[40%] h-[40%] bg-blue-600/10 blur-[120px] rounded-full" />
        <div className="absolute bottom-[10%] right-[10%] w-[30%] h-[30%] bg-red-600/5 blur-[100px] rounded-full" />
      </div>

      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <header className="flex justify-between items-center border-b border-white/10 pb-6">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-indigo-700 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">CCTV MONITOR </h1>
              <p className="text-xs text-white/40 font-medium uppercase tracking-widest mt-0.5">Automated Proctoring Intelligence</p>
            </div>
          </div>

          <div className="flex gap-4">
            <div className="px-4 py-2 bg-white/5 rounded-xl border border-white/10 flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-sm font-medium text-white/80">System Online</span>
            </div>
            <button className="bg-white text-black px-6 py-2 rounded-xl text-sm font-bold hover:bg-white/90 transition-all active:scale-95 shadow-lg shadow-white/5">
              Admin Portal
            </button>
          </div>
        </header>

        {/* Dashboard Content */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <div className="lg:col-span-7 space-y-8">
            <LiveFeed />

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="p-6 bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 flex flex-col gap-1">
                <span className="text-white/40 text-[10px] font-bold uppercase tracking-widest">Active Tracks</span>
                <span className="text-3xl font-bold font-mono">{stats.active_tracks}</span>
              </div>
              <div className="p-6 bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 flex flex-col gap-1">
                <span className="text-white/40 text-[10px] font-bold uppercase tracking-widest">Incident Rate</span>
                <span className="text-3xl font-bold font-mono text-red-500">{stats.incident_rate}</span>
              </div>
              <div className="p-6 bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 flex flex-col gap-1">
                <span className="text-white/40 text-[10px] font-bold uppercase tracking-widest">Detection Confidence</span>
                <span className="text-3xl font-bold font-mono text-blue-400">{stats.confidence}</span>
              </div>
            </div>
          </div>

          <div className="lg:col-span-5 h-[calc(100vh-200px)] sticky top-8">
            <ViolationLog />
          </div>
        </div>
      </div>

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.1);
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.2);
        }
      `}</style>
    </main>
  );
}
