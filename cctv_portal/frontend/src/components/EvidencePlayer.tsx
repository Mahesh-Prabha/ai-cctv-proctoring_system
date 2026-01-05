"use client";

import React, { useState, useEffect } from 'react';

interface EvidencePlayerProps {
    url: string;
    onClose: () => void;
}

export default function EvidencePlayer({ url, onClose }: EvidencePlayerProps) {
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in zoom-in duration-200">
            <div className="relative w-full max-w-4xl bg-zinc-900 rounded-3xl overflow-hidden shadow-2xl border border-white/10">
                {/* Header */}
                <div className="absolute top-0 inset-x-0 p-6 flex justify-between items-center bg-gradient-to-b from-black/80 to-transparent z-10">
                    <div>
                        <h3 className="text-lg font-bold text-white uppercase tracking-tighter">Review Evidence Clip</h3>
                        <p className="text-[10px] text-white/40 uppercase font-medium">Supabase Secure Playback</p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-white/10 rounded-full transition-colors text-white/60 hover:text-white"
                    >
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Video Player */}
                <div className="aspect-video bg-black flex items-center justify-center">
                    {url ? (
                        <video
                            src={url}
                            controls
                            autoPlay
                            className="w-full h-full"
                        >
                            Your browser does not support the video tag.
                        </video>
                    ) : (
                        <div className="flex flex-col items-center gap-4 text-white/20">
                            <div className="w-12 h-12 rounded-full border-2 border-dashed border-white/20 animate-spin" />
                            <p className="text-sm">Loading evidence stream...</p>
                        </div>
                    )}
                </div>

                {/* Footer Status */}
                <div className="p-4 bg-black/40 flex justify-between items-center px-8 border-t border-white/5">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                        <span className="text-[10px] text-white/40 font-bold uppercase">PROCTOR_VALIDATED_MP4</span>
                    </div>
                    <span className="text-[10px] text-white/20 font-mono">{url.split('/').pop()}</span>
                </div>
            </div>
        </div>
    );
}
