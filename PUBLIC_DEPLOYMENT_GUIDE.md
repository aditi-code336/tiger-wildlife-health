# Public Deployment Guide — Tiger Wildlife Health Diagnostic System

This guide outlines step-by-step methods to deploy the **Tiger Wildlife Health Anomaly Detection and Veterinary Diagnostic System** to a public internet URL accessible to evaluators, professors, reviewers, and field practitioners.

---

## Summary of Options

| Method | Speed | Cost | Persistence | Best Used For |
| :--- | :--- | :--- | :--- | :--- |
| **Render.com (Docker)** | 5 mins | Free | Permanent (24/7) | Official project submission & live portfolio link |
| **Hugging Face Spaces** | 5 mins | Free | Permanent (24/7) | Academic ML showcase & resume demonstration |
| **Instant SSH Tunnel** | 30 secs | Free | While terminal open | Live viva presentation, demo on mobile device |
| **Railway / Fly.io** | 5 mins | Free trial | Permanent | Enterprise cloud production container deployment |

---

## Step 0: Git Repository Status (Already Completed)

The local project repository has been initialized with a clean `.gitignore` and committed (commit `3c2e051`):
* All large raw PDFs and multi-gigabyte training photos are excluded to comply with cloud limits.
* All trained model weights (`models/photo_nn.npz`, `model.joblib`), pathology databases (`tiger_conditions_154.json`), backend endpoints, and frontend files are committed.
* Total clean repository footprint: **19.4 MB**.

---

## Option 1: Render.com (Permanent Free Cloud Hosting) — Recommended

Render provides free cloud hosting with automated SSL HTTPS certificates and continuous deployment directly from your Dockerfile.

### 1. Push Code to GitHub
1. Create a new repository on [github.com/new](https://github.com/new) (e.g., `tiger-wildlife-health`).
2. Run the following commands in your project terminal:
   ```bash
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/tiger-wildlife-health.git
   git branch -M main
   git push -u origin main
   ```

### 2. Deploy on Render
1. Log in to [render.com](https://render.com) (free signup using GitHub).
2. Click **New +** in the top navigation and select **Web Service**.
3. Choose **Build and deploy from a Git repository** and connect your `tiger-wildlife-health` repository.
4. Render will automatically detect the **`Dockerfile`**.
5. Set:
   * **Name:** `tiger-wildlife-health` (or your preferred name)
   * **Region:** Any (e.g., Singapore or Frankfurt)
   * **Instance Type:** **Free** ($0/month)
6. Click **Deploy Web Service**.
7. In 2 to 3 minutes, your service will be live at:
   ```
   https://tiger-wildlife-health.onrender.com
   ```

---

## Option 2: Hugging Face Spaces (Free Docker ML Space)

Hugging Face Spaces is an industry-standard platform for academic and machine learning portfolio deployments.

### Steps:
1. Create a free account at [huggingface.co](https://huggingface.co).
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
3. Fill in:
   * **Space name:** `tiger-health-diagnostics`
   * **License:** `mit` or `apache-2.0`
   * **Space SDK:** Select **Docker** -> **Blank**.
4. Clone the space repository locally or add it as a secondary git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<YOUR_HF_USERNAME>/tiger-health-diagnostics
   git push hf main
   ```
5. Hugging Face will automatically build the container and host it publicly at:
   ```
   https://huggingface.co/spaces/<YOUR_HF_USERNAME>/tiger-health-diagnostics
   ```

---

## Option 3: Instant Public Tunnel in 30 Seconds (Live Demo / Viva)

If you need a public URL immediately (e.g. to open the app on your mobile phone or share with your evaluator during a live presentation), you can create an instant secure HTTPS tunnel to your locally running Uvicorn server (port 8001) without creating any cloud accounts:

### Method A: Pinggy SSH Tunnel (Zero Installation, Zero Signup)
Open a new terminal window and run:
```bash
ssh -p 443 -R0:localhost:8001 a.pinggy.io
```
Pinggy will print an immediate public HTTPS link (e.g., `https://xyz.a.pinggy.link`). Anyone in the world can open that link in their browser.

### Method B: Cloudflare Tunnel (Free & Fast)
If you have `cloudflared` installed:
```bash
cloudflared tunnel --url http://localhost:8001
```
Cloudflare will provide a `https://<random-id>.trycloudflare.com` public URL.

---

## Option 4: Railway.app (One-Click Docker Cloud Deploy)

1. Sign in to [railway.app](https://railway.app) with GitHub.
2. Click **New Project** -> **Deploy from GitHub repo**.
3. Select your `tiger-wildlife-health` repository.
4. Under **Settings** -> **Networking**, click **Generate Domain**.
5. Railway provisions your container on an HTTPS domain (e.g., `tiger-wildlife-health.up.railway.app`).

---

## Verification After Public Deployment

Once your public URL is active:
1. Open the URL in your browser: confirm the 2-tab interface loads.
2. Test **Mode 1 (Photo)**: upload a tiger photo; verify the Top 5 differential diagnoses and probabilities appear.
3. Test **Renew Analysis**: click *Renew Analysis* to confirm the results clear and inputs reset.
4. Test **Clinical Reference**: verify that all 165 conditions load and can be searched.
