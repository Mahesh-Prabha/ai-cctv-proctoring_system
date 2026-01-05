# AI CCTV Proctoring System

This repository contains the full CCTV Proctoring Portal, including the Backend (AI Analytics) and the Frontend (Dashboard).

## Project Structure

- **cctv_portal/backend**: FastAPI backend that handles video streams (RTSP/File), AI analysis (YOLO, tracking, face identification), and Supabase integration.
- **cctv_portal/frontend**: Next.js dashboard for real-time monitoring and violation review.

---

## 🚀 Deployment Guide

This project is designed for a **Split Deployment**:
1.  **Frontend** -> Vercel (Automatic, Scalable)
2.  **Backend** -> VPS or Render/Railway/AWS (GPU/CPU Intensive, Long-running)

### **A. Frontend (Vercel)**
1.  Push this repository to your GitHub.
2.  Open [Vercel](https://vercel.com) and click **"Add New Project"**.
3.  Select this repository.
4.  **Root Directory**: Set to `cctv_portal/frontend`.
5.  **Environment Variables**: Add the following:
    *   `NEXT_PUBLIC_SUPABASE_URL`: Your Supabase Project URL.
    *   `NEXT_PUBLIC_SUPABASE_ANON_KEY`: Your Supabase API Key.
    *   `NEXT_PUBLIC_API_URL`: The URL of your deployed Backend (e.g., `http://your-server-ip:8000`).

### **B. Backend (Docker / VPS)**
The backend is memory and CPU intensive. You should deploy it on a server with at least 4GB of RAM.

#### **Option 1: Using Render (Recommended)**
1.  Log in to [Render.com](https://render.com).
2.  Click **"New +"** -> **"Web Service"**.
3.  Connect your GitHub repository.
4.  **Service Type**: Select **"Docker"**.
5.  **Project Settings**:
    *   **Root Directory**: `cctv_portal/backend` (This is very important).
    *   **Instance Type**: Choose a plan with at least **4GB RAM** (Starter or above).
6.  **Environment Variables**: Add the following:
    *   `SUPABASE_URL`
    *   `SUPABASE_SERVICE_ROLE_KEY`
    *   `GEMINI_API_KEY`
    *   `PORT`: `8000`
7.  Click **"Create Web Service"**. Render will build the image and start the AI server.

#### **Option 2: Direct Setup (Ubuntu/Windows)**
1.  Ensure Python 3.11 and FFmpeg are installed.
2.  Follow the [Local Setup](#local-setup) steps.

---

## 🛠️ Local Setup

### 1. Backend
1.  `cd cctv_portal/backend`
2.  Create and activate a virtual environment.
3.  Install dependencies: `pip install -r requirements.txt`
4.  Configure `.env` file.
5.  Start server: `python -m uvicorn app.main:app --port 8000`

### 2. Frontend
1.  `cd cctv_portal/frontend`
2.  `npm install`
3.  Configure `.env.local`.
4.  Run: `npm run dev`

## 📋 Secrets Checklist
You will need to provide the following secrets during deployment:
- **Supabase**: `URL`, `ANON_KEY`, `SERVICE_ROLE_KEY`.
- **Gemini**: `GEMINI_API_KEY`.
- **System**: `NEXT_PUBLIC_API_URL` (points to the running backend).