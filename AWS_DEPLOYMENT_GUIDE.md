# AWS Public Deployment Guide — Tiger Wildlife Health Diagnostic System

This guide provides step-by-step instructions to deploy your containerized application to **Amazon Web Services (AWS)** using either **AWS App Runner** (recommended, fully managed) or **AWS EC2** (free-tier virtual machine).

---

## Prerequisites Already Prepared in Your Project
* **GitHub Repository:** [github.com/aditi-code336/tiger-wildlife-health](https://github.com/aditi-code336/tiger-wildlife-health) (already committed & pushed).
* **Dockerfile & Docker Compose:** Tested and ready on port `8001`.
* **Model Bundles:** Trained pure-NumPy neural network (`photo_nn.npz`) and clinical anomaly detector (`model.joblib`) committed.

---

## Method 1: AWS App Runner (Recommended — Fastest & Zero Server Maintenance)

**AWS App Runner** is AWS's modern managed service for containerized web applications. It connects directly to your GitHub repository, builds your `Dockerfile`, and provisions a permanent public HTTPS URL with automatic SSL and zero server management.

### Steps in the AWS Console:

1. **Log in to AWS:** Open the [AWS Management Console](https://console.aws.amazon.com/).
2. **Navigate to App Runner:** Search for **App Runner** in the top search bar (or go to `https://console.aws.amazon.com/apprunner`).
3. **Click Create Service:**
   * **Source:** Select **Source code repository**.
   * **Repository provider:** Click **Add new** to connect your GitHub account (`aditi-code336`).
   * **Repository:** Select **`aditi-code336/tiger-wildlife-health`**.
   * **Branch:** Select **`main`**.
   * **Deployment trigger:** Choose **Automatic** (deploys automatically when you push updates).
4. **Configure Build:**
   * Select **Use a Dockerfile**.
   * **Port:** Enter **`8001`**.
5. **Configure Service:**
   * **Service name:** `tiger-wildlife-health`
   * **Virtual CPU & Memory:** `1 vCPU, 2 GB` (plenty for our lightweight pure-NumPy model).
6. **Review and Create:** Click **Create & Deploy**.
7. In ~3 to 4 minutes, App Runner will provide your live public URL:
   ```
   https://<random-id>.<region>.awsapprunner.com
   ```

---

## Method 2: AWS EC2 (Free Tier Virtual Machine)

If you have an AWS Free Tier account and prefer running on an EC2 instance:

### Steps:

1. **Launch an EC2 Instance:**
   * Go to [EC2 Console](https://console.aws.amazon.com/ec2) &rarr; **Launch Instance**.
   * **Name:** `tiger-health-server`
   * **OS Image:** **Ubuntu 24.04 LTS** (Free tier eligible).
   * **Instance Type:** **`t2.micro`** or **`t3.micro`** (Free tier eligible).
   * **Key Pair:** Create or select an existing key pair (e.g. `vockey` or `tiger-key`).
2. **Configure Security Group (Firewall):**
   * Check **Allow SSH traffic from Anywhere**.
   * Check **Allow HTTP traffic from the internet**.
   * Click **Add Security Group Rule**:
     * **Type:** Custom TCP
     * **Port range:** **`8001`**
     * **Source:** `0.0.0.0/0` (Anywhere)
3. **Automate Setup via User Data:**
   * Scroll down to **Advanced Details** &rarr; find the **User Data** text box.
   * Paste the contents of [`aws_ec2_setup.sh`](file:///d:/College/Semester%205-2026/FOML/Project/aws_ec2_setup.sh):
     ```bash
     #!/bin/bash
     apt-get update -y
     apt-get install -y docker.io git docker-compose curl
     systemctl start docker
     systemctl enable docker
     cd /home/ubuntu
     git clone https://github.com/aditi-code336/tiger-wildlife-health.git
     cd tiger-wildlife-health
     docker-compose up -d --build
     ```
4. **Launch Instance:** Click **Launch Instance**.
5. In ~2 minutes, copy your instance's **Public IPv4 address** from the EC2 Console.
6. Open your browser:
   ```
   http://<YOUR_EC2_PUBLIC_IP>:8001/
   ```

---

## Method 3: AWS CLI Deployment (If You Prefer Command Line)

If you want to deploy directly from your local terminal using the installed `aws-cli/2.36.34`:

### 1. Configure AWS CLI Credentials
Open PowerShell and run:
```powershell
aws configure
```
Enter your credentials when prompted:
* **AWS Access Key ID:** `[Your Access Key from AWS IAM / Learner Lab]`
* **AWS Secret Access Key:** `[Your Secret Access Key]`
* **Default region name:** `us-east-1` (or `ap-south-1` for Mumbai)
* **Default output format:** `json`

### 2. Verify Your Login
```powershell
aws sts get-caller-identity
```

### 3. Deploy via Amazon Elastic Container Registry (ECR) & App Runner
Once logged in, run these commands to build and push your Docker image:
```powershell
# Create an ECR repository
aws ecr create-repository --repository-name tiger-wildlife-health

# Authenticate Docker to your Amazon ECR registry
$account_id = (aws sts get-caller-identity --query Account --output text)
$region = "us-east-1"
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin "$account_id.dkr.ecr.$region.amazonaws.com"

# Tag and push your Docker image
docker tag tiger-health-system:latest "$account_id.dkr.ecr.$region.amazonaws.com/tiger-wildlife-health:latest"
docker push "$account_id.dkr.ecr.$region.amazonaws.com/tiger-wildlife-health:latest"
```
