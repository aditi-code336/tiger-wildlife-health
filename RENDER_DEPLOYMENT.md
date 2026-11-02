# Deploying Tiger Wildlife Health Diagnostic System on Render

This guide provides the exact steps to deploy the application on [Render.com](https://render.com) for permanent 24/7 public access with automatic HTTPS.

---

## Prerequisites

1. Your code is pushed to your GitHub repository:
   ```
   https://github.com/aditi-code336/tiger-wildlife-health
   ```
2. A free account on [Render.com](https://render.com) (sign up with GitHub).

---

## Method 1: Automatic 1-Click Deploy via Blueprint (Fastest)

The project includes a ready-to-use [`render.yaml`](./render.yaml) configuration.

1. Go to your [Render Dashboard](https://dashboard.render.com/).
2. Click the **New +** button in the top navigation bar and select **Blueprint**.
3. Connect your GitHub repository: `aditi-code336/tiger-wildlife-health`.
4. Render will detect the `render.yaml` file automatically:
   * **Service Name**: `tiger-wildlife-health`
   * **Runtime**: Docker
   * **Plan**: Free
   * **Health Check**: `/health`
5. Click **Apply**.
6. Render will automatically build the Docker image and deploy your container. Once completed, your live URL will appear at the top:
   ```
   https://tiger-wildlife-health.onrender.com
   ```

---

## Method 2: Manual Web Service Creation (Alternative)

If you prefer configuring via the Render web UI:

1. In the [Render Dashboard](https://dashboard.render.com/), click **New +** > **Web Service**.
2. Select **Build and deploy from a Git repository**.
3. Choose or search for `aditi-code336/tiger-wildlife-health`.
4. Configure the settings:
   * **Name**: `tiger-wildlife-health` (or your chosen name)
   * **Region**: Choose the closest region (e.g., `Singapore`, `Frankfurt`, or `Oregon`)
   * **Branch**: `main`
   * **Root Directory**: *(leave blank)*
   * **Runtime**: **Docker**
   * **Instance Type**: **Free** ($0 / month)
5. Expand **Advanced Settings**:
   * **Health Check Path**: `/health`
   * **Environment Variables**:
     * `PORT` = `8001`
     * `PYTHONUNBUFFERED` = `1`
6. Click **Create Web Service**.

---

## Verification & Health Check

Once the deployment shows **Live**:

1. Open your live application URL:
   ```
   https://tiger-wildlife-health.onrender.com
   ```
2. Test the health endpoint:
   ```
   https://tiger-wildlife-health.onrender.com/health
   ```
   Expected response:
   ```json
   {
     "status": "healthy",
     "timestamp": "2026-11-02T...",
     "model_loaded": true,
     "database_backend": "SQLite fallback"
   }
   ```
3. Test interactive features:
   - **Clinical Differential Diagnostics** (Select symptoms, tiger reserve, and run diagnosis).
   - **Photo Health Assessment** (Upload tiger image to classify healthy vs injured).
   - **NTCA Tiger Reserve Explorer** (Browse reserves, population, veterinary contact info).

---

## Notes on Free Tier Behavior
* **Spin-down on Inactivity**: Free Web Services on Render sleep after 15 minutes of inactivity. When a new request arrives, Render spins the container back up in ~30–50 seconds.
* **Storage**: In-memory and local SQLite database (`data/predictions.db`) are bundled with the container image.
