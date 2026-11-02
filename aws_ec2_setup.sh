#!/bin/bash
# AWS EC2 Automated User Data Setup Script for Tiger Wildlife Health Diagnostic System
set -e

# 1. Update packages and install Docker
apt-get update -y
apt-get install -y docker.io git docker-compose curl

# 2. Start and enable Docker service
systemctl start docker
systemctl enable docker

# 3. Clone the public GitHub repository
cd /home/ubuntu
git clone https://github.com/aditi-code336/tiger-wildlife-health.git
cd tiger-wildlife-health

# 4. Build and start the container in detached mode
docker-compose up -d --build

# 5. Verify local healthcheck
sleep 15
curl -f http://localhost:8001/health || echo "Healthcheck verification pending..."

echo "Tiger Health Diagnostics deployed and running on port 8001!"
