# Deployment Guide — Indian Sign Language Recognition & Translation

This document explains the production deployment architecture for the **Indian Sign Language Recognition & Multilingual Voice Translator**.

---

## 1. Architecture Overview

Due to the continuous computer vision, MediaPipe hand landmark extraction, and TensorFlow neural network inference pipelines, the application is structured with a decoupled, high-performance architecture:

```
┌───────────────────────────────────────┐
│           Vercel Frontend             │
│   (Static Single Page Application)    │
│  - Hosted on Vercel Edge Global CDN   │
│  - Instant loading, zero server cost  │
│  - Interactive HUD & Indic Translator │
└──────────────────┬────────────────────┘
                   │
                   │ REST API / SSE / MJPEG Video Stream
                   ▼
┌───────────────────────────────────────┐
│        Python Backend Container       │
│  (Docker / Render / Railway / Local)  │
│  - Flask REST & SSE Server (Port 5001)│
│  - MediaPipe Hands Holistic Pipeline  │
│  - Dual-Model Fusion (CNN + Landmark) │
│  - Native TTS & Indic Translation     │
└───────────────────────────────────────┘
```

---

## 2. Deploying Frontend to Vercel (1-Click or CLI)

### Method A: Vercel CLI (1-Step Command)

Run from the project root:

```bash
vercel --prod
```

When prompted:
1. `Set up and deploy?` → **Y**
2. `Which scope do you want to deploy to?` → Select your personal account / team.
3. `Link to existing project?` → **N** (or select if updating an existing deployment).
4. `What's your project's name?` → `indian-sign-language-translator`
5. `In which directory is your code located?` → `./`

The deployment will automatically detect `vercel.json` and serve the production assets from `public/`.

### Method B: GitHub Vercel Integration

1. Go to [https://vercel.com/new](https://vercel.com/new).
2. Import your GitHub repository (`TechnoDream956/indian-sign-language-recognition`).
3. Set **Root Directory** to `./` and **Output Directory** to `public`.
4. Click **Deploy**.

---

## 3. Deploying Python Backend (Docker / Render / Railway)

### Method A: Render.com (Recommended Free/Standard Tier)

1. Fork or push this repository to GitHub.
2. Sign in to [https://dashboard.render.com](https://dashboard.render.com).
3. Click **New +** → **Blueprint** and select your repository (it will automatically read `render.yaml`).
4. Alternatively, click **New +** → **Web Service** → select **Docker** runtime.
5. Set:
   - **Environment**: `Docker`
   - **Port**: `5001`
   - **Health Check Path**: `/api/status`
6. Click **Create Web Service**. Render will build the container from `Dockerfile`.

### Method B: Railway.app

1. Go to [https://railway.app/new](https://railway.app/new).
2. Choose **Deploy from GitHub repo**.
3. Railway automatically detects the `Dockerfile` and deploys the container.
4. Set environment variable `PORT=5001`.

---

## 4. Connecting Frontend to Backend

1. In the top-right header of the web application, click the **⚙️ Backend** button.
2. Enter your backend server's URL (e.g. `https://isl-recognition-backend.onrender.com` or `http://localhost:5001`).
3. The application will store the URL in local browser storage and reconnect all vision, speech, and translation streams automatically!
