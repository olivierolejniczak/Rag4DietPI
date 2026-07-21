#!/bin/bash
# rollback-llama-swap.sh
# Rag4DietPI - Retour arriere de la migration llama-swap vers Ollama.
#
# Stoppe/desactive le service rag-llm, restaure config.env depuis la sauvegarde
# .pre-llamaswap.bak, puis reactive Ollama. Les binaires llama-swap/llama-server
# et les GGUF sont CONSERVES (suppression manuelle si souhaite).
#
# Usage : sudo bash rollback-llama-swap.sh [PROJECT_DIR]

set -euo pipefail

BLUE='\033[1;34m'
GREEN='\033[1;32m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $1" >&2; }
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

trap 'log_err "Echec inattendu (ligne $LINENO)."' ERR

PROJECT_DIR="${1:-$(pwd)}"
CONFIG_ENV="$PROJECT_DIR/config.env"
BACKUP_ENV="${CONFIG_ENV}.pre-llamaswap.bak"
SERVICE_NAME="rag-llm.service"

if [ "$(id -u)" -ne 0 ]; then
    log_err "Ce script doit etre lance en root."
    exit 1
fi

echo "============================================"
echo " Rag4DietPI - Rollback llama-swap -> Ollama"
echo "============================================"

# 1. Arret du backend llama-swap.
if systemctl list-unit-files 2>/dev/null | grep -q "^${SERVICE_NAME}"; then
    systemctl disable --now "$SERVICE_NAME" 2>/dev/null || log_warn "Service $SERVICE_NAME deja arrete."
    log_ok "Service $SERVICE_NAME stoppe et desactive."
else
    log_warn "Service $SERVICE_NAME absent."
fi

# 2. Restauration de config.env.
if [ -f "$BACKUP_ENV" ]; then
    cp "$CONFIG_ENV" "${CONFIG_ENV}.llamaswap.bak" 2>/dev/null || true
    cp "$BACKUP_ENV" "$CONFIG_ENV"
    log_ok "config.env restaure depuis $BACKUP_ENV."
else
    log_warn "Sauvegarde $BACKUP_ENV introuvable — config.env non restaure."
    log_warn "Retirez manuellement LLM_API_BASE / LLM_MODEL_* si presents."
fi

# 3. Reactivation d'Ollama.
if command -v ollama >/dev/null 2>&1; then
    if systemctl list-unit-files 2>/dev/null | grep -q '^ollama'; then
        systemctl enable --now ollama
        log_ok "Service ollama reactive."
    else
        nohup ollama serve > /var/log/ollama.log 2>&1 &
        log_ok "Ollama relance manuellement (nohup)."
    fi
else
    log_err "Ollama n'est plus installe (desinstalle lors de la migration)."
    log_err "Reinstallez-le : curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

echo ""
log_ok "Rollback termine. Verifiez : ./status.sh"
log_info "Les binaires llama-swap/llama-server et les GGUF ($PROJECT_DIR/models) sont conserves."
