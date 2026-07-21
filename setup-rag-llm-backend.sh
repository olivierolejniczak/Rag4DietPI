#!/bin/bash
# setup-rag-llm-backend.sh
# Rag4DietPI - Migration du backend LLM : Ollama -> llama-swap + llama-server
#
# Installe llama-swap (proxy hot-swap, binaire Go statique) et llama-server
# (llama.cpp, compile avec optimisations CPU natives), telecharge les modeles
# GGUF selon le profil materiel detecte, genere llama-swap.yaml + le service
# systemd rag-llm.service, stoppe/desactive Ollama, puis adapte config.env.
#
# Contraintes :
#   - Offline-first : les seuls telechargements ont lieu ICI, a l'installation,
#     avec verification de checksum (SHA256 GitHub + OID LFS HuggingFace).
#   - Idempotent : relancable sans dommage.
#   - Reversible : sauvegarde config.env + rollback-llama-swap.sh reactive Ollama.
#
# Usage : sudo bash setup-rag-llm-backend.sh [PROJECT_DIR]

set -euo pipefail

# ============================================================================
# Couleurs / journalisation
# ============================================================================
BLUE='\033[1;34m'
GREEN='\033[1;32m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $1" >&2; }
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Les avertissements non bloquants sont regroupes dans un recapitulatif final.
WARNINGS=()
warn() { log_warn "$1"; WARNINGS+=("$1"); }

# Le trap ERR n'affiche la ligne fautive que pour un echec REELLEMENT non gere
# (set -e). Les commandes tolerees utilisent "|| true".
trap 'log_err "Echec inattendu (ligne $LINENO). Rien n'"'"'a ete desactive de force ; relancez apres correction."' ERR

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR="${1:-$(pwd)}"
MODELS_DIR="$PROJECT_DIR/models"
YAML_PATH="$PROJECT_DIR/llama-swap.yaml"
CONFIG_ENV="$PROJECT_DIR/config.env"
SERVICE_NAME="rag-llm.service"
LISTEN_ADDR="127.0.0.1:11434"   # on reutilise le port d'Ollama pour minimiser les changements
SVC_USER="ragsvc"
NPROC="$(nproc)"

# Depots HuggingFace des GGUF (variante Q4_K_M pour les LLM, f16 pour l'embed)
REPO_Q15="Qwen/Qwen2.5-1.5B-Instruct-GGUF"
REPO_Q3="Qwen/Qwen2.5-3B-Instruct-GGUF"
REPO_Q7="Qwen/Qwen2.5-7B-Instruct-GGUF"
REPO_EMB="CompendiumLabs/bge-base-en-v1.5-gguf"

FILE_Q15="qwen2.5-1.5b-instruct-q4_k_m.gguf"
FILE_Q3="qwen2.5-3b-instruct-q4_k_m.gguf"
FILE_Q7="qwen2.5-7b-instruct-q4_k_m.gguf"
# bge-base-en-v1.5 en f16 : meme famille/dimension (768) que FastEmbed -> pas de
# reindexation. FastEmbed reste la source primaire ; llama-swap ne sert les
# embeddings qu'en repli (voir lib/llm_helper.py).
FILE_EMB="bge-base-en-v1.5-f16.gguf"

echo "============================================"
echo " Rag4DietPI - Backend LLM : Ollama -> llama-swap"
echo "============================================"
echo ""

# ============================================================================
# Pre-requis
# ============================================================================
if [ "$(id -u)" -ne 0 ]; then
    log_err "Ce script doit etre lance en root (systemctl, /usr/local/bin, useradd)."
    exit 1
fi

if [ ! -f "$CONFIG_ENV" ]; then
    log_err "config.env introuvable dans $PROJECT_DIR — lancez d'abord setup-rag-core.sh."
    exit 1
fi

# Detection d'architecture -> nom GOARCH utilise par les releases
ARCH_RAW="$(uname -m)"
case "$ARCH_RAW" in
    x86_64)  GOARCH="amd64" ;;
    aarch64) GOARCH="arm64" ;;
    *)
        log_err "Architecture non supportee : $ARCH_RAW (x86_64 / aarch64 uniquement)."
        exit 1
        ;;
esac
log_info "Architecture : $ARCH_RAW ($GOARCH), $NPROC threads CPU."

# ============================================================================
# Helpers de telechargement verifie
# ============================================================================

# download_verified <url> <dest> <sha256|"">
# Telecharge dans un fichier .part, verifie le SHA256 si fourni, puis renomme.
download_verified() {
    local url="$1" dest="$2" expected="$3" actual
    curl -fSL --retry 3 --retry-delay 2 -o "${dest}.part" "$url"
    if [ -n "$expected" ]; then
        actual="$(sha256sum "${dest}.part" | awk '{print $1}')"
        if [ "$actual" != "$expected" ]; then
            rm -f "${dest}.part"
            log_err "Checksum invalide pour $(basename "$dest") (attendu ${expected:0:12}…, obtenu ${actual:0:12}…)."
            return 1
        fi
    fi
    mv "${dest}.part" "$dest"
}

# hf_download <repo> <fichier> <dest>
# Telecharge un GGUF depuis HuggingFace en verifiant l'OID SHA256 du pointeur LFS.
hf_download() {
    local repo="$1" file="$2" dest="$3" pointer expected
    if [ -f "$dest" ]; then
        log_ok "Deja present : $(basename "$dest")"
        return 0
    fi
    log_info "Telechargement de $file ($repo)…"
    pointer="$(curl -fsSL "https://huggingface.co/${repo}/raw/main/${file}" 2>/dev/null || true)"
    expected="$(printf '%s\n' "$pointer" | sed -n 's/^oid sha256:\([0-9a-f]\{64\}\).*/\1/p')"
    if [ -z "$expected" ]; then
        warn "SHA256 LFS indisponible pour $file — telechargement sans verification de checksum."
    fi
    download_verified "https://huggingface.co/${repo}/resolve/main/${file}?download=true" "$dest" "$expected"
    log_ok "$file"
}

# ============================================================================
# Phase A - Binaires (llama-swap + llama-server)
# ============================================================================
install_llama_swap() {
    if command -v llama-swap >/dev/null 2>&1; then
        log_ok "llama-swap deja installe : $(llama-swap --version 2>/dev/null | head -1 || echo present)."
        return 0
    fi
    log_info "Installation de llama-swap (derniere release mostlygeek/llama-swap)…"
    local api urls asset_url sums_url tmp asset_name expected actual
    api="$(curl -fsSL "https://api.github.com/repos/mostlygeek/llama-swap/releases/latest")"
    urls="$(printf '%s' "$api" | grep -oE '"browser_download_url": *"[^"]+"' | cut -d'"' -f4)"
    # Actif attendu : *_linux_<amd64|arm64>.tar.gz
    asset_url="$(printf '%s\n' "$urls" | grep -E "linux_${GOARCH}\.tar\.gz$" | head -1)"
    sums_url="$(printf '%s\n' "$urls" | grep -iE 'checksums?\.txt$' | head -1)"
    if [ -z "$asset_url" ]; then
        log_err "Aucun binaire llama-swap linux_${GOARCH} trouve dans la derniere release."
        return 1
    fi
    tmp="$(mktemp -d)"
    asset_name="$(basename "$asset_url")"
    curl -fSL --retry 3 -o "$tmp/$asset_name" "$asset_url"
    # Verification du checksum publie (fichier checksums.txt : "<sha256>  <nom>")
    if [ -n "$sums_url" ]; then
        curl -fsSL -o "$tmp/checksums.txt" "$sums_url"
        expected="$(awk -v f="$asset_name" '$2==f || $2=="*"f {print $1}' "$tmp/checksums.txt" | head -1)"
        if [ -n "$expected" ]; then
            actual="$(sha256sum "$tmp/$asset_name" | awk '{print $1}')"
            if [ "$actual" != "$expected" ]; then
                rm -rf "$tmp"
                log_err "Checksum llama-swap invalide."
                return 1
            fi
            log_ok "Checksum llama-swap verifie."
        else
            warn "Nom d'actif absent de checksums.txt — llama-swap non verifie."
        fi
    else
        warn "checksums.txt introuvable dans la release — llama-swap non verifie."
    fi
    tar -xzf "$tmp/$asset_name" -C "$tmp"
    install -m 755 "$(find "$tmp" -type f -name llama-swap | head -1)" /usr/local/bin/llama-swap
    rm -rf "$tmp"
    log_ok "llama-swap installe dans /usr/local/bin."
}

install_llama_server() {
    if command -v llama-server >/dev/null 2>&1 && llama-server --version >/dev/null 2>&1; then
        log_ok "llama-server deja present."
        return 0
    fi
    # Compilation depuis les sources : -DGGML_NATIVE=ON active les optimisations
    # CPU (AVX/AVX2/NEON) du processeur local, ce que les binaires prebuilds ne
    # garantissent pas. C'est le chemin recommande sur CPU contraint.
    log_info "Compilation de llama-server (llama.cpp) — peut durer plusieurs minutes…"
    DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential cmake git libcurl4-openssl-dev >/dev/null
    local src="/usr/local/src/llama.cpp"
    if [ ! -d "$src/.git" ]; then
        git clone --depth 1 https://github.com/ggml-org/llama.cpp "$src"
    else
        git -C "$src" pull --ff-only || warn "Mise a jour des sources llama.cpp ignoree (etat local conserve)."
    fi
    cmake -S "$src" -B "$src/build" -DGGML_NATIVE=ON -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$src/build" --config Release -j"$NPROC" --target llama-server >/dev/null
    install -m 755 "$src/build/bin/llama-server" /usr/local/bin/llama-server
    log_ok "llama-server compile et installe dans /usr/local/bin."
}

install_llama_swap
install_llama_server

# ============================================================================
# Phase B - Modeles GGUF selon le profil RAM
# ============================================================================
RAM_KB="$(awk '/MemTotal/{print $2}' /proc/meminfo)"
RAM_GB=$(( RAM_KB / 1024 / 1024 ))
log_info "RAM detectee : ${RAM_GB} Go — selection des modeles."

mkdir -p "$MODELS_DIR"

# Contextes dimensionnes pour CPU/RAM contraints.
if [ "$RAM_GB" -lt 8 ]; then
    # <8 Go : un seul modele (1.5b) ; les 3 tiers pointent dessus avec des
    # contextes croissants (jamais un modele que le profil ne peut charger).
    hf_download "$REPO_Q15" "$FILE_Q15" "$MODELS_DIR/$FILE_Q15"
    M_QUICK="$FILE_Q15"; M_DEFAULT="$FILE_Q15"; M_DEEP="$FILE_Q15"
    CTX_QUICK=2048; CTX_DEFAULT=4096; CTX_DEEP=4096
elif [ "$RAM_GB" -lt 16 ]; then
    # 8-16 Go : 1.5b (quick) + 3b (default/deep).
    hf_download "$REPO_Q15" "$FILE_Q15" "$MODELS_DIR/$FILE_Q15"
    hf_download "$REPO_Q3"  "$FILE_Q3"  "$MODELS_DIR/$FILE_Q3"
    M_QUICK="$FILE_Q15"; M_DEFAULT="$FILE_Q3"; M_DEEP="$FILE_Q3"
    CTX_QUICK=4096; CTX_DEFAULT=8192; CTX_DEEP=8192
else
    # 16 Go+ : les trois tiers.
    hf_download "$REPO_Q15" "$FILE_Q15" "$MODELS_DIR/$FILE_Q15"
    hf_download "$REPO_Q3"  "$FILE_Q3"  "$MODELS_DIR/$FILE_Q3"
    hf_download "$REPO_Q7"  "$FILE_Q7"  "$MODELS_DIR/$FILE_Q7"
    M_QUICK="$FILE_Q15"; M_DEFAULT="$FILE_Q3"; M_DEEP="$FILE_Q7"
    CTX_QUICK=4096; CTX_DEFAULT=8192; CTX_DEEP=8192
fi

# Modele d'embedding (repli ; FastEmbed reste primaire).
hf_download "$REPO_EMB" "$FILE_EMB" "$MODELS_DIR/$FILE_EMB"

# ============================================================================
# Phase C - Configuration llama-swap
# ============================================================================
# Schema llama-swap : models: { "<nom>": { cmd: "<ligne>", ttl: <s> } }.
# ${PORT} est une MACRO substituee par llama-swap (laissee litterale ici).
# ttl = duree d'inactivite (s) avant dechargement automatique du modele.
# NOTE : le schema exact (cle "cmd"/"ttl") doit etre valide contre la version de
# llama-swap reellement installee — voir le controle de demarrage en fin de script.
log_info "Generation de $YAML_PATH…"
cat > "$YAML_PATH" <<YAML
# Configuration llama-swap - generee par setup-rag-llm-backend.sh
# Profil : ${RAM_GB} Go RAM, ${NPROC} threads. Ne pas editer a la main :
# relancez setup-rag-llm-backend.sh pour regenerer.
models:
  "rag-quick":
    cmd: /usr/local/bin/llama-server -m ${MODELS_DIR}/${M_QUICK} -c ${CTX_QUICK} --host 127.0.0.1 --port \${PORT} -t ${NPROC}
    ttl: 180
  "rag-default":
    cmd: /usr/local/bin/llama-server -m ${MODELS_DIR}/${M_DEFAULT} -c ${CTX_DEFAULT} --host 127.0.0.1 --port \${PORT} -t ${NPROC}
    ttl: 180
  "rag-deep":
    cmd: /usr/local/bin/llama-server -m ${MODELS_DIR}/${M_DEEP} -c ${CTX_DEEP} --host 127.0.0.1 --port \${PORT} -t ${NPROC}
    ttl: 300
  "rag-embed":
    cmd: /usr/local/bin/llama-server -m ${MODELS_DIR}/${FILE_EMB} --embeddings --pooling cls -c 512 --host 127.0.0.1 --port \${PORT} -t ${NPROC}
    ttl: 600
YAML
log_ok "llama-swap.yaml genere (tiers : quick=${M_QUICK}, default=${M_DEFAULT}, deep=${M_DEEP})."

# ============================================================================
# Phase D - Utilisateur de service + unite systemd
# ============================================================================
if ! id "$SVC_USER" >/dev/null 2>&1; then
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SVC_USER"
    log_ok "Utilisateur systeme $SVC_USER cree."
fi

# Droits : ragsvc doit lire les modeles et la config, sans droit d'ecriture ailleurs.
chown -R "$SVC_USER:$SVC_USER" "$MODELS_DIR"
chmod 750 "$MODELS_DIR"
chown root:"$SVC_USER" "$YAML_PATH"
chmod 640 "$YAML_PATH"

log_info "Installation du service $SERVICE_NAME…"
cat > "/etc/systemd/system/$SERVICE_NAME" <<SERVICE
[Unit]
Description=Rag4DietPI - Backend LLM (llama-swap + llama.cpp)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SVC_USER}
Group=${SVC_USER}
ExecStart=/usr/local/bin/llama-swap --config ${YAML_PATH} --listen ${LISTEN_ADDR}
Restart=on-failure
RestartSec=3

# Durcissement
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${PROJECT_DIR}
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictSUIDSGID=true
RestrictNamespaces=true
LockPersonality=true
SystemCallArchitectures=native
CapabilityBoundingSet=

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
log_ok "Unite $SERVICE_NAME installee."

# ============================================================================
# Phase E - Bascule Ollama -> llama-swap
# ============================================================================
# Sauvegarde de config.env avant modification.
if [ ! -f "${CONFIG_ENV}.pre-llamaswap.bak" ]; then
    cp "$CONFIG_ENV" "${CONFIG_ENV}.pre-llamaswap.bak"
    log_ok "Sauvegarde : ${CONFIG_ENV}.pre-llamaswap.bak"
fi

# set_env <cle> <valeur> : met a jour ou ajoute une cle dans config.env.
set_env() {
    local key="$1" val="$2"
    if grep -qE "^${key}=" "$CONFIG_ENV"; then
        sed -i "s|^${key}=.*|${key}=${val}|" "$CONFIG_ENV"
    else
        printf '%s=%s\n' "$key" "$val" >> "$CONFIG_ENV"
    fi
}

# API OpenAI-compatible exposee par llama-swap sur l'ancien port d'Ollama.
set_env "LLM_API_BASE"          "http://127.0.0.1:11434/v1"
set_env "LLM_MODEL_QUICK"       "rag-quick"
set_env "LLM_MODEL_DEFAULT"     "rag-default"
set_env "LLM_MODEL_DEEP"        "rag-deep"
set_env "LLM_EMBED_MODEL"       "rag-embed"
# Le 1er appel apres un swap charge le modele (30-120 s sur CPU) : timeout large.
set_env "LLM_FIRST_CALL_TIMEOUT" "180"
# OLLAMA_HOST est conserve tel quel pour permettre le rollback.
log_ok "config.env mis a jour (LLM_API_BASE + mapping des tiers)."

# Arret et desactivation d'Ollama (NON desinstalle automatiquement).
if systemctl list-unit-files 2>/dev/null | grep -q '^ollama'; then
    systemctl disable --now ollama 2>/dev/null || warn "Impossible de desactiver le service ollama."
    log_ok "Service ollama stoppe et desactive."
else
    warn "Aucun service systemd ollama detecte (Ollama lance manuellement ?). Arretez-le a la main si necessaire."
fi

# Demarrage du nouveau backend.
systemctl enable --now "$SERVICE_NAME"

# ============================================================================
# Phase F - Verification
# ============================================================================
log_info "Attente du proxy llama-swap sur ${LISTEN_ADDR}…"
OK=false
for _ in $(seq 1 30); do
    if curl -sf "http://${LISTEN_ADDR}/v1/models" >/dev/null 2>&1; then
        OK=true
        break
    fi
    sleep 1
done
if [ "$OK" = true ]; then
    log_ok "llama-swap repond. Modeles exposes :"
    curl -sf "http://${LISTEN_ADDR}/v1/models" | python3 -c 'import sys,json;[print("  -",m["id"]) for m in json.load(sys.stdin).get("data",[])]' 2>/dev/null || true
else
    warn "llama-swap ne repond pas encore. Verifiez : journalctl -u ${SERVICE_NAME} -n 50"
    warn "Si l'erreur porte sur le format du YAML, alignez llama-swap.yaml sur le schema de la version installee (llama-swap --help)."
fi

# ============================================================================
# Desinstallation optionnelle d'Ollama (interactive, defaut Non)
# ============================================================================
if command -v ollama >/dev/null 2>&1; then
    echo ""
    read -r -p "Desinstaller completement Ollama et ses modeles ? [o/N] " REP || REP="N"
    case "$REP" in
        [oO]|[oO][uU][iI])
            rm -f /usr/local/bin/ollama /usr/bin/ollama 2>/dev/null || true
            rm -rf /usr/share/ollama 2>/dev/null || true
            rm -f /etc/systemd/system/ollama.service 2>/dev/null || true
            systemctl daemon-reload
            warn "Ollama desinstalle. Le rollback ne pourra pas reactiver Ollama sans reinstallation."
            log_ok "Ollama desinstalle."
            ;;
        *)
            log_info "Ollama conserve (stoppe/desactive). rollback-llama-swap.sh pourra le reactiver."
            ;;
    esac
fi

# ============================================================================
# Recapitulatif
# ============================================================================
echo ""
echo "============================================"
log_ok "Migration terminee."
echo "  Backend      : llama-swap + llama-server (${LISTEN_ADDR})"
echo "  Modeles      : $MODELS_DIR"
echo "  Config YAML  : $YAML_PATH"
echo "  Service      : systemctl status ${SERVICE_NAME}"
echo "  Rollback     : bash rollback-llama-swap.sh"
if [ "${#WARNINGS[@]}" -gt 0 ]; then
    echo ""
    log_warn "Avertissements (${#WARNINGS[@]}) :"
    for w in "${WARNINGS[@]}"; do echo "  - $w"; done
fi
echo "============================================"
