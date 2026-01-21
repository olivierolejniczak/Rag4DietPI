# Installation Guide

This guide provides detailed installation instructions for deploying the RAG System on various platforms.

## Table of Contents

- [DietPi Installation](#dietpi-installation)
- [Debian/Ubuntu Installation](#debianubuntu-installation)
- [Raspberry Pi OS Installation](#raspberry-pi-os-installation)
- [Post-Installation Setup](#post-installation-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## DietPi Installation

DietPi provides the most optimized experience for this project.

### 1. Install Docker

```bash
sudo dietpi-software

# Navigate to:
# 1. Search Software
# 2. Search: docker
# 3. Select: [*] 134 Docker Compose
# 4. Confirm and Install
```

### 2. Run Setup Scripts

```bash
# Download scripts
cd /home/dietpi
mkdir rag-system && cd rag-system

# Copy setup scripts here, then run:
sudo bash setup-rag-core.sh
sudo bash setup-rag-ingest.sh
sudo bash setup-rag-query.sh

# Optional: Web UI
sudo bash setup-rag-webui.sh

# Optional: Backup utilities
sudo bash setup-rag-backup.sh
```

### 3. Verify Installation

```bash
./status.sh
```

---

## Debian/Ubuntu Installation

### 1. Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo systemctl enable docker
sudo systemctl start docker

# Add user to docker group (optional, avoids sudo)
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Adjust Data Directories

The scripts default to DietPi paths. For standard Debian/Ubuntu:

```bash
# Create data directories
sudo mkdir -p /opt/rag-data/{qdrant,searxng,backups}
sudo chmod 777 /opt/rag-data/*
```

Edit the scripts or set environment variables:

```bash
export QDRANT_DATA_DIR=/opt/rag-data/qdrant
export SEARXNG_DATA_DIR=/opt/rag-data/searxng
export BACKUP_DIR=/opt/rag-data/backups
```

### 3. Run Setup Scripts

```bash
cd /opt
sudo git clone https://github.com/yourusername/rag-system.git
cd rag-system

sudo bash setup-rag-core.sh
sudo bash setup-rag-ingest.sh
sudo bash setup-rag-query.sh
```

---

## Raspberry Pi OS Installation

### 1. Use 64-bit OS

The RAG system requires 64-bit OS for optimal performance:

```bash
# Verify architecture
uname -m
# Should output: aarch64
```

If you have 32-bit, reinstall with 64-bit Raspberry Pi OS.

### 2. Increase Swap (Critical for Pi 4 with 4GB)

```bash
# Check current swap
free -h

# Increase swap to 4GB
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set: CONF_SWAPSIZE=4096

sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 3. Install Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker pi
sudo systemctl enable docker
```

### 4. Run Setup

```bash
# Reboot after docker install
sudo reboot

# After reboot
cd ~
mkdir rag-system && cd rag-system
# Copy scripts and run setup
sudo bash setup-rag-core.sh
```

---

## Post-Installation Setup

### 1. Ingest Your Documents

```bash
# Create documents folder
mkdir -p ~/rag-system/documents

# Copy your files
cp /path/to/your/docs/* ~/rag-system/documents/

# Run ingestion
cd ~/rag-system
./ingest.sh ./documents/
```

### 2. Test the System

```bash
# Simple query
./query.sh "What documents do I have?"

# Check status
./status.sh
```

### 3. Configure for Your Needs

Edit `config.env` to customize:

```bash
nano config.env

# Key settings to adjust:
# - LLM_MODEL: Change if you have more/less RAM
# - CHUNK_SIZE: Larger for technical docs, smaller for conversations
# - DEFAULT_TOP_K: More results = more context but slower
```

### 4. Set Up Auto-Start (Optional)

Create a systemd service for Ollama (if not already configured):

```bash
sudo nano /etc/systemd/system/ollama.service
```

```ini
[Unit]
Description=Ollama LLM Server
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/ollama serve
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable ollama
sudo systemctl start ollama
```

### 5. Set Up Scheduled Backups (Optional)

```bash
# Install cron job for daily backups at 3 AM
./install-cron.sh
```

---

## Verification

### Check All Services

```bash
./status.sh
```

Expected output:
```
=== RAG System Status ===

=== System Profile ===
RAM: 8GB | CPUs: 4 @ 2400MHz
CPU Score: 9 | Arch: x86_64
Batch Size: 64 | Embedding: BAAI/bge-base-en-v1.5

=== Services ===
Docker: OK
Qdrant: OK (0 points)
Ollama: OK (qwen2.5:3b)
SearXNG: OK

=== Python Components ===
FastEmbed: OK
Qdrant Client: OK
Unstructured: OK
Spellcheck: OK (pyspellchecker)
FlashRank: OK
```

### Test Ingestion

```bash
# Create test document
echo "This is a test document about artificial intelligence and machine learning." > documents/test.txt

# Ingest
./ingest.sh documents/test.txt

# Query
./query.sh "What is the test document about?"
```

### Test Web Search

```bash
./query.sh --web-only "Current weather in Paris"
```

---

## Troubleshooting

### Docker Permission Denied

```bash
sudo usermod -aG docker $USER
newgrp docker
# Or logout and login again
```

### Qdrant Container Fails to Start

```bash
# Check logs
docker logs qdrant

# Common fix: permissions
sudo chmod 777 /mnt/dietpi_userdata/qdrant
# Or for Debian:
sudo chmod 777 /opt/rag-data/qdrant

# Recreate container
docker rm qdrant
sudo bash setup-rag-core.sh
```

### Ollama Out of Memory

```bash
# Check available RAM
free -h

# Add swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Use smaller model
nano config.env
# Change: LLM_MODEL=qwen2.5:1.5b
```

### SearXNG Not Returning JSON

```bash
# Check SearXNG logs
docker logs searxng

# Verify settings
cat /mnt/dietpi_userdata/searxng/settings.yml | grep formats

# Should show:
# formats:
#   - html
#   - json
```

### Python Import Errors

```bash
# Reinstall dependencies
pip3 install --upgrade --break-system-packages fastembed qdrant-client unstructured

# Check Python version
python3 --version
# Requires 3.9+
```

### Slow Performance

1. **Enable caching** in config.env:
   ```
   QDRANT_CACHE_ENABLED=true
   RESPONSE_CACHE_ENABLED=true
   ```

2. **Use quick mode** for simple queries:
   ```bash
   ./query.sh --mode quick "simple question"
   ```

3. **Reduce TOP_K** for faster retrieval:
   ```
   DEFAULT_TOP_K=3
   ```

---

## Next Steps

- [Read the full documentation](README.md)
- [Configure advanced features](FEATURES.md)
- [Set up the Web UI](WEBUI.md)
- [Learn about backup/restore](BACKUP.md)
