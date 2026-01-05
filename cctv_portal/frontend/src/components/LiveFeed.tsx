"use client";

import React, { useState } from 'react';

export default function LiveFeed() {
    const [rtspUrl, setRtspUrl] = useState('../../test_exam_video.mp4');
    const [isStreaming, setIsStreaming] = useState(false);
    const [showAI, setShowAI] = useState(false);

    const startStream = async () => {
        try {
            await fetch(`http://localhost:8000/streams/start?rtsp_url=${encodeURIComponent(rtspUrl)}`, {
                method: 'POST'
            });
            setIsStreaming(true);
        } catch (error) {
            console.error("Failed to start stream", error);
        }
    };

    return (
        <div className="flex flex-col gap-4 p-6 bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 shadow-2xl">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-xl font-semibold text-white">Live Monitoring</h2>
                    <p className="text-[10px] text-white/40 uppercase tracking-tighter">Direct Camera Access</p>
                </div>
                <div className="flex gap-4 items-center">
                    {/* Mode Toggle */}
                    <div className="flex bg-black/40 p-1 rounded-lg border border-white/10">
                        <button
                            onClick={() => setShowAI(false)}
                            className={`px-3 py-1 text-[10px] font-bold rounded-md transition-all ${!showAI ? 'bg-blue-600 text-white shadow-lg' : 'text-white/40 hover:text-white'}`}
                        >
                            DIRECT FEED
                        </button>
                        <button
                            onClick={() => setShowAI(true)}
                            className={`px-3 py-1 text-[10px] font-bold rounded-md transition-all ${showAI ? 'bg-purple-600 text-white shadow-lg' : 'text-white/40 hover:text-white'}`}
                        >
                            AI DIAGNOSTICS
                        </button>
                    </div>

                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={rtspUrl}
                            onChange={(e) => setRtspUrl(e.target.value)}
                            className="bg-black/40 border border-white/20 rounded-lg px-3 py-1 text-sm text-white focus:outline-none focus:border-blue-500 w-48"
                            placeholder="RTSP URL"
                        />
                        <button
                            onClick={startStream}
                            className="bg-white/10 hover:bg-white/20 text-white px-4 py-1 rounded-lg text-sm transition-colors font-medium border border-white/10"
                        >
                            Connect
                        </button>
                    </div>
                </div>
            </div>

            <div className="max-h-[380px] aspect-[16/9] bg-black rounded-xl overflow-hidden relative border border-white/5 group shadow-inner mx-auto">
                {isStreaming ? (
                    <img
                        src={`http://localhost:8000/streams/video?rtsp_url=${encodeURIComponent(rtspUrl)}&mode=${showAI ? 'annotated' : 'raw'}`}
                        alt="Live Stream"
                        className="w-full h-full object-contain"
                    />
                ) : (
                    <div className="absolute inset-0 flex items-center justify-center text-white/30 flex-col gap-2">
                        <div className="w-10 h-10 rounded-full border-2 border-dashed border-white/20 animate-spin" />
                        <p className="text-xs font-medium uppercase tracking-widest">Feed Offline</p>
                    </div>
                )}

                {/* Status Indicator Overlay */}
                <div className="absolute top-4 left-4 flex gap-2">
                    <div className="bg-black/60 backdrop-blur-md px-2 py-1 rounded border border-white/10 flex items-center gap-2">
                        <div className={`w-1.5 h-1.5 rounded-full ${isStreaming ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
                        <span className="text-[10px] font-bold text-white uppercase">{isStreaming ? 'Live' : 'Offline'}</span>
                    </div>
                    {showAI && (
                        <div className="bg-purple-600/80 backdrop-blur-md px-2 py-1 rounded border border-white/10 flex items-center gap-2">
                            <span className="text-[10px] font-bold text-white uppercase tracking-tighter">AI Overlay Active</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
