"use client";

import React, { useEffect, useState } from 'react';
import EvidencePlayer from './EvidencePlayer';

interface Violation {
    id: string;
    timestamp: string;
    candidate_id: string;
    violation_type: string;
    severity: string;
    evidence_url: string;
    review_status?: string; // PENDING, CONFIRMED, REJECTED
}

export default function ViolationLog() {
    const [violations, setViolations] = useState<Violation[]>([]);
    const [activeVideoUrl, setActiveVideoUrl] = useState<string | null>(null);
    const [loadingId, setLoadingId] = useState<string | null>(null);

    useEffect(() => {
        // Initial Fetch
        const fetchViolations = async () => {
            try {
                const res = await fetch('http://localhost:8000/violations');
                const data = await res.json();
                setViolations(data);
            } catch (error) {
                console.error("Failed to fetch violations", error);
            }
        };
        fetchViolations();

        // WebSocket for REAL-TIME
        const ws = new WebSocket('ws://localhost:8000/ws/violations');

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'VIOLATION') {
                console.log("Real-time Violation Received:", message.data);
                setViolations(prev => {
                    const exists = prev.some(v => v.id === message.data.id);
                    if (exists) {
                        return prev.map(v => v.id === message.data.id ? { ...v, ...message.data } : v);
                    }
                    return [message.data, ...prev].slice(0, 50);
                });
            }
        };

        ws.onopen = () => console.log("WebSocket Connected for Security Alerts");
        ws.onerror = (err) => console.error("WebSocket Error", err);
        ws.onclose = () => console.log("WebSocket Closed");

        return () => ws.close();
    }, []);

    const handleReview = async (id: string, status: string, action: string) => {
        setLoadingId(id);
        try {
            const res = await fetch(`http://localhost:8000/violations/${id}/review`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ status, action })
            });
            if (res.ok) {
                // Update local state immediately
                setViolations(prev => prev.map(v => v.id === id ? { ...v, review_status: status } : v));
            }
        } catch (error) {
            console.error("Failed to submit review", error);
        } finally {
            setLoadingId(null);
        }
    };

    return (
        <div className="flex flex-col gap-4 p-6 bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 shadow-2xl h-full">
            <h2 className="text-xl font-semibold text-white">Security Alerts</h2>
            <div className="overflow-y-auto flex-1 pr-2 space-y-3 custom-scrollbar">
                {violations.length === 0 ? (
                    <p className="text-white/40 text-sm text-center py-10">No violations detected</p>
                ) : (
                    violations.map((v, i) => {
                        const isReviewed = v.review_status && v.review_status !== 'PENDING';
                        return (
                            <div
                                key={`${v.id}-${i}`}
                                className={`p-4 bg-black/40 rounded-xl border border-white/5 flex flex-col gap-2 hover:border-red-500/50 transition-all group ${isReviewed ? 'opacity-50 grayscale-[0.5]' : ''}`}
                            >
                                <div className="flex justify-between items-start">
                                    <span className="text-sm font-bold text-red-400 uppercase tracking-tighter">{v.violation_type}</span>
                                    <span className="text-[10px] text-white/30">{new Date(v.timestamp.length > 12 ? v.timestamp : Number(v.timestamp) * 1000).toLocaleString()}</span>
                                </div>
                                <div className="flex justify-between items-center">
                                    <span className="text-xs text-white/70">Candidate: <span className="text-white font-medium">{v.candidate_id}</span></span>
                                    <div className="flex items-center gap-2">
                                        {v.review_status === 'CONFIRMED' && <span className="text-[10px] font-bold text-red-500 uppercase tracking-widest">Confirmed</span>}
                                        {v.review_status === 'REJECTED' && <span className="text-[10px] font-bold text-green-500 uppercase tracking-widest">Rejected</span>}
                                        <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${v.severity === 'HIGH' ? 'bg-red-500/20 text-red-500' : 'bg-yellow-500/20 text-yellow-500'
                                            }`}>
                                            {v.severity}
                                        </span>
                                    </div>
                                </div>

                                <div className="flex gap-2 mt-1">
                                    {v.evidence_url && (
                                        <button
                                            onClick={() => setActiveVideoUrl(v.evidence_url)}
                                            className="flex-1 py-1.5 bg-red-500/10 hover:bg-red-500/20 text-red-400 text-[10px] font-bold rounded border border-red-500/20 transition-all flex items-center justify-center gap-2"
                                        >
                                            <span>▶</span> EVIDENCE
                                        </button>
                                    )}

                                    {!isReviewed && (
                                        <>
                                            <button
                                                disabled={loadingId === v.id}
                                                onClick={() => handleReview(v.id, 'CONFIRMED', 'MARK_MALPRACTICE')}
                                                className="px-3 py-1.5 bg-red-600 hover:bg-red-700 text-white text-[10px] font-bold rounded shadow-lg shadow-red-900/20 transition-all active:scale-95 disabled:opacity-50"
                                            >
                                                MALPRACTICE
                                            </button>
                                            <button
                                                disabled={loadingId === v.id}
                                                onClick={() => handleReview(v.id, 'REJECTED', 'MARK_FALSE_POSITIVE')}
                                                className="px-3 py-1.5 bg-white/10 hover:bg-white/20 text-white text-[10px] font-bold rounded border border-white/10 transition-all active:scale-95 disabled:opacity-50"
                                            >
                                                REJECT
                                            </button>
                                        </>
                                    )}
                                </div>
                            </div>
                        );
                    })
                )}
            </div>

            {/* Evidence Modal */}
            {activeVideoUrl && (
                <EvidencePlayer
                    url={activeVideoUrl}
                    onClose={() => setActiveVideoUrl(null)}
                />
            )}
        </div>
    );
}
