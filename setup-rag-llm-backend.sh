#!/bin/bash
# setup-rag-llm-backend.sh
# Rag4DietPI - Migration du backend LLM : Ollama -> llama-swap + llama-server
#
# Installe llama-swap (proxy hot-swap, binaire Go statique) et llama-server
# (llama.cpp, compile avec optimisations CPU natives), telecharge les modeles
# GGUF selon le profil materiel detecte, genere llama-swap.yaml + le service
# systemd rag-llm.service, desinstalle Ollama, puis adapte config.env.
#
# Contraintes :
#   - Offline-first : les seuls telechargements ont lieu ICI, a l'installation,
#     avec verification de checksum (SHA256 GitHub + OID LFS HuggingFace).
#   - Idempotent : relancable sans dommage.
#   - Une sauvegarde config.env.pre-llamaswap.bak est conservee par securite.
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
CONFIG_ENV="$PROJECT_DIR/config.env"
# Les actifs du service (modeles GGUF + YAML) vivent HORS de PROJECT_DIR, dans un
# emplacement systeme appartenant a ragsvc. Indispensable quand PROJECT_DIR est
# sous /home : sinon le service (User=ragsvc + ProtectHome) ne peut ni traverser
# le home ni lire le YAML ("permission denied").
LLM_STATE_DIR="/var/lib/rag-llm"
MODELS_DIR="$LLM_STATE_DIR/models"
YAML_PATH="$LLM_STATE_DIR/llama-swap.yaml"
SERVICE_NAME="rag-llm.service"
LISTEN_ADDR="0.0.0.0:11434"     # expose sur le LAN (UI + API) ; reutilise le port d'Ollama
SVC_USER="ragsvc"
NPROC="$(nproc)"

# Depots HuggingFace des GGUF (variante Q4_K_M pour les LLM, f16 pour l'embed)
REPO_Q15="Qwen/Qwen2.5-1.5B-Instruct-GGUF"
REPO_Q3="Qwen/Qwen2.5-3B-Instruct-GGUF"
REPO_Q7="Qwen/Qwen2.5-7B-Instruct-GGUF"
REPO_EMB="CompendiumLabs/bge-base-en-v1.5-gguf"

FILE_Q15="qwen2.5-1.5b-instruct-q4_k_m.gguf"
FILE_Q3="qwen2.5-3b-instruct-q4_k_m.gguf"
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

# hf_download_q4km <repo> <dest_dir>
# Telecharge TOUS les fichiers Q4_K_M .gguf d'un depot (gere le cas des modeles
# scindes en shards, ex. 7B = *-00001-of-00002.gguf) et renvoie le 1er shard,
# celui a passer a llama-server (-m) qui charge automatiquement les suivants.
hf_download_q4km() {
    local repo="$1" dir="$2" files f first
    files="$(curl -fsSL "https://huggingface.co/api/models/${repo}" 2>/dev/null \
        | python3 -c 'import sys,json; d=json.load(sys.stdin); [print(s["rfilename"]) for s in d.get("siblings",[]) if s["rfilename"].lower().endswith(".gguf") and "q4_k_m" in s["rfilename"].lower()]' \
        | sort)"
    if [ -z "$files" ]; then
        log_err "Aucun fichier Q4_K_M trouve dans $repo"
        return 1
    fi
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        hf_download "$repo" "$f" "$dir/$f" >&2
    done <<EOF
$files
EOF
    first="$(printf '%s\n' "$files" | head -1)"
    printf '%s' "$first"
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

# ----------------------------------------------------------------------------
# Selection des tiers par empreinte memoire estimee (plutot qu'un seuil fixe).
#
# On choisit, pour chaque tier, le plus gros modele dont l'empreinte crete tient
# dans le budget RAM disponible :
#   budget = RAM_totale - RESERVED
#   empreinte(modele) ~= poids Q4_K_M resident + buffers calcul (~600 Mo)
#                        + cache KV(ctx)
# Le cache KV fp16 = 2(K+V) * n_layers * n_kv_heads * head_dim * ctx * 2 octets
# est faible ici (GQA) : ~0,45 Go pour le 7B @8k, negligeable devant les poids.
# On pre-calcule donc une empreinte crete par modele@ctx (en Mo).
# RESERVED couvre Docker+Qdrant+SearXNG (~0,5 Go), FastEmbed+Splade+process de
# requete (~1,3 Go), OS + marge (~1,7 Go).
# ----------------------------------------------------------------------------
RAM_MB=$(( RAM_KB / 1024 ))
RESERVED_MB="${LLM_RESERVED_MB:-3500}"       # surchargeable via config.env
BUDGET_MB=$(( RAM_MB - RESERVED_MB ))
[ "$BUDGET_MB" -lt 0 ] && BUDGET_MB=0

NEED_Q15=1800   # 1.5b Q4_K_M @ 4k  (~1,1 Go poids + 0,6 buffers + ~0,1 KV)
NEED_Q3=2900    # 3b   Q4_K_M @ 8k  (~2,0 Go poids + 0,6 buffers + ~0,3 KV)
NEED_Q7=5850    # 7b   Q4_K_M @ 8k  (~4,8 Go poids + 0,6 buffers + ~0,45 KV)

log_info "RAM ${RAM_GB} Go — budget LLM estime : ${BUDGET_MB} Mo (reserve ${RESERVED_MB} Mo)."

# quick : toujours le 1.5b (le plus rapide) ; contexte reduit si RAM tres serree.
hf_download "$REPO_Q15" "$FILE_Q15" "$MODELS_DIR/$FILE_Q15"
M_QUICK="$FILE_Q15"
if [ "$BUDGET_MB" -ge "$NEED_Q15" ]; then CTX_QUICK=4096; else CTX_QUICK=2048; fi

# default : 3b s'il tient dans le budget, sinon repli sur le 1.5b.
if [ "$BUDGET_MB" -ge "$NEED_Q3" ]; then
    hf_download "$REPO_Q3" "$FILE_Q3" "$MODELS_DIR/$FILE_Q3"
    M_DEFAULT="$FILE_Q3"; CTX_DEFAULT=8192
else
    M_DEFAULT="$FILE_Q15"; CTX_DEFAULT="$CTX_QUICK"
fi

# deep : le plus gros modele dont l'empreinte tient (7b > 3b > 1.5b).
if [ "$BUDGET_MB" -ge "$NEED_Q7" ]; then
    # Le 7B est scinde en shards : on telecharge tout et on pointe sur le 1er.
    Q7_MAIN="$(hf_download_q4km "$REPO_Q7" "$MODELS_DIR")"
    M_DEEP="$Q7_MAIN"; CTX_DEEP=8192
elif [ "$BUDGET_MB" -ge "$NEED_Q3" ]; then
    hf_download "$REPO_Q3" "$FILE_Q3" "$MODELS_DIR/$FILE_Q3"  # no-op si deja present
    M_DEEP="$FILE_Q3"; CTX_DEEP=8192
else
    M_DEEP="$FILE_Q15"; CTX_DEEP="$CTX_QUICK"
fi

# Etiquette lisible (taille) pour le journal, ex. "3b" / "7b".
_sz() { echo "$1" | grep -oE '[0-9.]+b' | head -1; }
log_info "Tiers retenus : quick=$(_sz "$M_QUICK")(@${CTX_QUICK}) default=$(_sz "$M_DEFAULT")(@${CTX_DEFAULT}) deep=$(_sz "$M_DEEP")(@${CTX_DEEP})."

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

# rag-embed est un repli dormant (FastEmbed tourne en local et reste primaire ;
# rag-embed n'est sollicite que si FastEmbed devient indisponible). L'isoler
# dans son propre groupe evite que ce repli rare n'evince un modele de chat
# actif, et inversement.
groups:
  chat:
    swap: true
    exclusive: true
    members:
      - "rag-quick"
      - "rag-default"
      - "rag-deep"
  embed:
    swap: false
    exclusive: false
    members:
      - "rag-embed"
YAML
log_ok "llama-swap.yaml genere (tiers : quick=${M_QUICK}, default=${M_DEFAULT}, deep=${M_DEEP})."

# ============================================================================
# Phase D - Utilisateur de service + unite systemd
# ============================================================================
if ! id "$SVC_USER" >/dev/null 2>&1; then
    useradd --system --no-create-home --shell /usr/sbin/nologin "$SVC_USER"
    log_ok "Utilisateur systeme $SVC_USER cree."
fi

# Droits : tout l'etat du service (modeles + YAML) appartient a ragsvc.
chown -R "$SVC_USER:$SVC_USER" "$LLM_STATE_DIR"
chmod 750 "$LLM_STATE_DIR" "$MODELS_DIR"
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
ReadWritePaths=${LLM_STATE_DIR}
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
log_ok "config.env mis a jour (LLM_API_BASE + mapping des tiers)."

# Arret ET desinstallation complete d'Ollama : le backend est desormais
# llama-swap, Ollama n'a plus de raison d'etre.
if systemctl list-unit-files 2>/dev/null | grep -q '^ollama'; then
    systemctl disable --now ollama 2>/dev/null || true
fi
# Ollama peut aussi tourner sans unite systemd (ex. 'ollama serve' manuel).
pkill -f 'ollama serve' 2>/dev/null || true
pkill -x ollama 2>/dev/null || true
# Suppression des binaires, bibliotheques, modeles et utilisateur/groupe Ollama.
rm -f /usr/local/bin/ollama /usr/bin/ollama 2>/dev/null || true
rm -rf /usr/local/lib/ollama /usr/share/ollama /root/.ollama 2>/dev/null || true
rm -f /etc/systemd/system/ollama.service /etc/systemd/system/*/ollama.service 2>/dev/null || true
systemctl daemon-reload 2>/dev/null || true
id ollama >/dev/null 2>&1 && userdel ollama 2>/dev/null || true
getent group ollama >/dev/null 2>&1 && groupdel ollama 2>/dev/null || true
log_ok "Ollama arrete et desinstalle."
# On stoppe d'abord une eventuelle instance existante du service pour liberer le
# port (les relances doivent repartir sur la config regeneree).
systemctl stop "$SERVICE_NAME" 2>/dev/null || true
# Le port 11434 doit etre libre avant de demarrer llama-swap.
for _ in $(seq 1 10); do
    ss -ltn 2>/dev/null | grep -q ':11434 ' || break
    sleep 1
done
if ss -ltn 2>/dev/null | grep -q ':11434 '; then
    warn "Le port 11434 est encore occupe — llama-swap risque de ne pas demarrer (ss -ltnp | grep 11434)."
fi

# Demarrage du backend. On utilise restart (pas 'enable --now') pour que la
# config regeneree soit bien prise en compte meme si le service tournait deja.
systemctl enable "$SERVICE_NAME" >/dev/null 2>&1 || true
systemctl restart "$SERVICE_NAME"

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
# On verifie que le port expose bien NOS modeles rag-*, et pas un autre backend
# (ex. Ollama residuel) qui repondrait sur le meme port.
if [ "$OK" = true ]; then
    MODELS_JSON="$(curl -sf "http://${LISTEN_ADDR}/v1/models" 2>/dev/null)"
    if printf '%s' "$MODELS_JSON" | grep -q '"rag-'; then
        log_ok "llama-swap repond. Modeles exposes :"
        printf '%s' "$MODELS_JSON" | python3 -c 'import sys,json;[print("  -",m["id"]) for m in json.load(sys.stdin).get("data",[])]' 2>/dev/null || true
    else
        OK=false
        warn "Le port 11434 repond mais n'expose PAS les modeles rag-* (backend residuel ?)."
        warn "Verifiez : ss -ltnp | grep 11434 ; journalctl -u ${SERVICE_NAME} -n 30"
    fi
fi
if [ "$OK" != true ]; then
    warn "Backend llama-swap non operationnel. Verifiez : journalctl -u ${SERVICE_NAME} -n 50"
    warn "Si l'erreur porte sur le format du YAML, alignez llama-swap.yaml sur le schema de la version installee (llama-swap --help)."
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
if [ "${#WARNINGS[@]}" -gt 0 ]; then
    echo ""
    log_warn "Avertissements (${#WARNINGS[@]}) :"
    for w in "${WARNINGS[@]}"; do echo "  - $w"; done
fi
echo "============================================"
