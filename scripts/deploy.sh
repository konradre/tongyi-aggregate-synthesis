#!/bin/bash
# Deploy research tool to VM
# Usage: ./scripts/deploy.sh [vm-ip]

VM_IP="${1:-192.168.1.119}"
VM_USER="comfy"
REMOTE_DIR="/home/comfy/research-tool"

echo "Deploying to $VM_USER@$VM_IP:$REMOTE_DIR"

# Sync files
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
    . "$VM_USER@$VM_IP:$REMOTE_DIR/"

# Build and start on VM
ssh "$VM_USER@$VM_IP" "cd $REMOTE_DIR && docker compose up -d --build"

echo "Deployed! API available at http://$VM_IP:8000"
echo "Docs at http://$VM_IP:8000/docs"
