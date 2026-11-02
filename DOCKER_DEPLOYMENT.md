# Docker Deployment Guide — Tiger Wildlife Health Diagnostic System

This guide outlines how to build, containerize, and deploy the **Tiger Wildlife Health Anomaly Detection and Veterinary Diagnostic System** using Docker and Docker Compose.

---

## 1. Prerequisites

* **Docker Engine** (v20.10+ / Docker Desktop on Windows/macOS/Linux)
* **Docker Compose** (v2.0+)

---

## 2. Quick Start Deployment

### Option A: Using Docker Compose (Recommended)

From the project root directory (`d:\College\Semester 5-2026\FOML\Project`), execute:

```bash
docker compose up -d --build
```

To view real-time runtime logs:
```bash
docker compose logs -f
```

To stop the container:
```bash
docker compose down
```

---

### Option B: Using Standalone Docker CLI

#### 1. Build the Docker Image
```bash
docker build -t tiger-health-system:latest .
```

#### 2. Run the Container
```bash
docker run -d \
  --name tiger_diagnostics \
  -p 8001:8001 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  --restart unless-stopped \
  tiger-health-system:latest
```

---

## 3. Verifying the Deployment

### Health Check Endpoint
```bash
curl -f http://localhost:8001/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "model_version": "v2.0-pure-numpy-photo-nn",
  "database": "connected"
}
```

### Accessing the Web Interface
Open your browser and navigate to:
```
http://localhost:8001/
```
You will see the 2-tab interface:
1. **Tiger Injury Checker:** Photo, Clinical Signs, Voice Dispatch, WhatsApp/Radio Messages, and Combined Assessment, complete with the **Renew Analysis** action.
2. **Clinical Reference & Pathology Guide:** Full 165-condition catalog with real-time search, category badges, expandable veterinary protocols, and one-click "Assess in Checker".

---

## 4. Container Architecture & Optimization

* **Base Image:** `python:3.11-slim` for minimal container size and fast startup.
* **Volume Mounts:**
  * `/app/data`: Persists SQLite predictions database (`predictions.db`) and observation history.
  * `/app/models`: Persists trained NumPy neural network weights (`photo_nn.npz`).
* **Clean Build Context:** `.dockerignore` excludes large raw PDF documents (40+ MB) and intermediate cache files, keeping the image build lightweight and reproducible.
* **Healthcheck:** Built-in 30-second interval polling ensuring automatic restarts if the service becomes unresponsive.
