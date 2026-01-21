#!/bin/bash
# setup-rag-backup.sh
# RAG System - Backup/Restore Utilities
# Full backup of scripts, settings, cache, Qdrant data
# Plain ASCII output

set -e

log_ok() { echo "[OK] $1"; }
log_err() { echo "[ERROR] $1" >&2; }
log_info() { echo "[INFO] $1"; }

PROJECT_DIR="${1:-$(pwd)}"
BACKUP_DIR="${BACKUP_DIR:-/mnt/dietpi_userdata/rag-backups}"
QDRANT_DATA_DIR="${QDRANT_DATA_DIR:-/mnt/dietpi_userdata/qdrant}"

echo "============================================"
echo " RAG System - Backup/Restore Setup"
echo "============================================"
echo ""

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

[ -f "./config.env" ] && source ./config.env

# ============================================================================
# Create backup script
# ============================================================================
log_info "Creating backup.sh..."
cat > "$PROJECT_DIR/backup.sh" << 'EOFBACKUP'
#!/bin/bash
# RAG System - Backup Utility
# Backs up: scripts, config, lib, cache, Qdrant data, tracking

set -e

log_ok() { echo "[OK] $1"; }
log_err() { echo "[ERROR] $1" >&2; }
log_info() { echo "[INFO] $1"; }

cd "$(dirname "$0")"
source ./config.env 2>/dev/null || true

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/mnt/dietpi_userdata/rag-backups}"
QDRANT_DATA_DIR="${QDRANT_DATA_DIR:-/mnt/dietpi_userdata/qdrant}"
SEARXNG_DATA_DIR="${SEARXNG_DATA_DIR:-/mnt/dietpi_userdata/searxng}"
PROJECT_DIR="$(pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="rag-backup-${TIMESTAMP}"

# Parse arguments
FULL_BACKUP=false
DATA_ONLY=false
CONFIG_ONLY=false
COMPRESS=true

show_help() {
    echo "Usage: ./backup.sh [options]"
    echo ""
    echo "Options:"
    echo "  --full         Full backup (all components)"
    echo "  --data         Data only (Qdrant, cache, tracking)"
    echo "  --config       Config only (scripts, settings, lib)"
    echo "  --no-compress  Skip compression"
    echo "  --output DIR   Custom backup directory"
    echo "  --name NAME    Custom backup name"
    echo "  --list         List existing backups"
    echo "  --help         Show this help"
    echo ""
    echo "Default: Full backup with compression"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --full) FULL_BACKUP=true; shift ;;
        --data) DATA_ONLY=true; shift ;;
        --config) CONFIG_ONLY=true; shift ;;
        --no-compress) COMPRESS=false; shift ;;
        --output) BACKUP_DIR="$2"; shift 2 ;;
        --name) BACKUP_NAME="$2"; shift 2 ;;
        --list)
            echo "Existing backups in $BACKUP_DIR:"
            ls -lh "$BACKUP_DIR"/*.tar.gz 2>/dev/null || echo "  No backups found"
            exit 0 ;;
        --help|-h) show_help; exit 0 ;;
        *) shift ;;
    esac
done

# Default to full backup
if ! $DATA_ONLY && ! $CONFIG_ONLY; then
    FULL_BACKUP=true
fi

echo "============================================"
echo " RAG System - Backup"
echo "============================================"
echo ""
echo "Project:   $PROJECT_DIR"
echo "Output:    $BACKUP_DIR"
echo "Name:      $BACKUP_NAME"
echo "Compress:  $COMPRESS"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"
mkdir -p "$BACKUP_PATH"

# ============================================================================
# Backup Config (scripts, settings, lib)
# ============================================================================
if $FULL_BACKUP || $CONFIG_ONLY; then
    log_info "Backing up configuration..."
    
    # Scripts
    mkdir -p "$BACKUP_PATH/scripts"
    cp -v *.sh "$BACKUP_PATH/scripts/" 2>/dev/null || true
    cp -v *.env "$BACKUP_PATH/scripts/" 2>/dev/null || true
    cp -v *.md "$BACKUP_PATH/scripts/" 2>/dev/null || true
    log_ok "Scripts backed up"
    
    # Python modules
    if [ -d "lib" ]; then
        cp -r lib "$BACKUP_PATH/"
        log_ok "Python modules backed up ($(ls lib/*.py 2>/dev/null | wc -l) files)"
    fi
    
    # Web UI
    if [ -d "webui" ]; then
        cp -r webui "$BACKUP_PATH/"
        log_ok "Web UI backed up"
    fi
fi

# ============================================================================
# Backup Data (Qdrant, cache, tracking)
# ============================================================================
if $FULL_BACKUP || $DATA_ONLY; then
    log_info "Backing up data..."
    
    # Cache
    if [ -d "cache" ]; then
        cp -r cache "$BACKUP_PATH/"
        CACHE_SIZE=$(du -sh cache 2>/dev/null | cut -f1)
        log_ok "Cache backed up ($CACHE_SIZE)"
    fi
    
    # Tracking
    if [ -d ".ingest_tracking" ]; then
        cp -r .ingest_tracking "$BACKUP_PATH/"
        TRACK_COUNT=$(ls .ingest_tracking/*.json 2>/dev/null | wc -l)
        log_ok "Tracking data backed up ($TRACK_COUNT files)"
    fi
    
    # Qdrant data (stop container for consistency)
    if [ -d "$QDRANT_DATA_DIR" ]; then
        log_info "Backing up Qdrant data..."
        
        # Check if Qdrant is running
        QDRANT_RUNNING=false
        if docker ps --format '{{.Names}}' | grep -q "^qdrant$"; then
            QDRANT_RUNNING=true
            log_info "Stopping Qdrant for consistent backup..."
            docker stop qdrant
            sleep 2
        fi
        
        mkdir -p "$BACKUP_PATH/qdrant"
        cp -r "$QDRANT_DATA_DIR"/* "$BACKUP_PATH/qdrant/" 2>/dev/null || true
        QDRANT_SIZE=$(du -sh "$QDRANT_DATA_DIR" 2>/dev/null | cut -f1)
        log_ok "Qdrant data backed up ($QDRANT_SIZE)"
        
        # Restart Qdrant if it was running
        if $QDRANT_RUNNING; then
            log_info "Restarting Qdrant..."
            docker start qdrant
            sleep 3
        fi
    fi
    
    # Documents (optional, can be large)
    if [ -d "documents" ]; then
        DOC_SIZE=$(du -sh documents 2>/dev/null | cut -f1)
        read -p "Backup documents folder ($DOC_SIZE)? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cp -r documents "$BACKUP_PATH/"
            log_ok "Documents backed up"
        else
            log_info "Documents skipped"
        fi
    fi
fi

# ============================================================================
# Create manifest
# ============================================================================
log_info "Creating manifest..."
cat > "$BACKUP_PATH/MANIFEST.txt" << EOFMANIFEST
RAG System Backup
=====================
Created: $(date)
Hostname: $(hostname)
Project: $PROJECT_DIR

Contents:
$(ls -la "$BACKUP_PATH")

Backup Type:
  Full: $FULL_BACKUP
  Data Only: $DATA_ONLY
  Config Only: $CONFIG_ONLY

Sizes:
$(du -sh "$BACKUP_PATH"/* 2>/dev/null)

Total: $(du -sh "$BACKUP_PATH" | cut -f1)
EOFMANIFEST
log_ok "Manifest created"

# ============================================================================
# Compress
# ============================================================================
if $COMPRESS; then
    log_info "Compressing backup..."
    cd "$BACKUP_DIR"
    tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"
    rm -rf "$BACKUP_NAME"
    FINAL_SIZE=$(du -sh "${BACKUP_NAME}.tar.gz" | cut -f1)
    log_ok "Compressed: ${BACKUP_NAME}.tar.gz ($FINAL_SIZE)"
    BACKUP_FILE="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"
else
    FINAL_SIZE=$(du -sh "$BACKUP_PATH" | cut -f1)
    BACKUP_FILE="$BACKUP_PATH"
fi

echo ""
echo "============================================"
echo " Backup Complete"
echo "============================================"
echo ""
echo "Location: $BACKUP_FILE"
echo "Size:     $FINAL_SIZE"
echo ""
echo "Restore with: ./restore.sh $BACKUP_FILE"
EOFBACKUP
chmod +x "$PROJECT_DIR/backup.sh"
log_ok "backup.sh"

# ============================================================================
# Create restore script
# ============================================================================
log_info "Creating restore.sh..."
cat > "$PROJECT_DIR/restore.sh" << 'EOFRESTORE'
#!/bin/bash
# RAG System - Restore Utility
# Restores: scripts, config, lib, cache, Qdrant data, tracking

set -e

log_ok() { echo "[OK] $1"; }
log_err() { echo "[ERROR] $1" >&2; }
log_info() { echo "[INFO] $1"; }

cd "$(dirname "$0")"

# Configuration
QDRANT_DATA_DIR="${QDRANT_DATA_DIR:-/mnt/dietpi_userdata/qdrant}"
PROJECT_DIR="$(pwd)"

show_help() {
    echo "Usage: ./restore.sh <backup-file> [options]"
    echo ""
    echo "Options:"
    echo "  --data         Restore data only"
    echo "  --config       Restore config only"
    echo "  --force        Overwrite without confirmation"
    echo "  --dry-run      Show what would be restored"
    echo "  --help         Show this help"
    echo ""
    echo "Examples:"
    echo "  ./restore.sh /path/to/rag-backup-20240101_120000.tar.gz"
    echo "  ./restore.sh /path/to/backup --data --force"
}

if [ $# -lt 1 ]; then
    show_help
    exit 1
fi

BACKUP_SOURCE="$1"
shift

DATA_ONLY=false
CONFIG_ONLY=false
FORCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --data) DATA_ONLY=true; shift ;;
        --config) CONFIG_ONLY=true; shift ;;
        --force) FORCE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help|-h) show_help; exit 0 ;;
        *) shift ;;
    esac
done

# Verify backup exists
if [ ! -e "$BACKUP_SOURCE" ]; then
    log_err "Backup not found: $BACKUP_SOURCE"
    exit 1
fi

echo "============================================"
echo " RAG System - Restore"
echo "============================================"
echo ""
echo "Source:  $BACKUP_SOURCE"
echo "Target:  $PROJECT_DIR"
echo "Dry run: $DRY_RUN"
echo ""

# Extract if compressed
TEMP_DIR=""
if [[ "$BACKUP_SOURCE" == *.tar.gz ]]; then
    log_info "Extracting backup..."
    TEMP_DIR=$(mktemp -d)
    tar -xzf "$BACKUP_SOURCE" -C "$TEMP_DIR"
    BACKUP_DIR=$(ls "$TEMP_DIR")
    BACKUP_PATH="$TEMP_DIR/$BACKUP_DIR"
else
    BACKUP_PATH="$BACKUP_SOURCE"
fi

# Show manifest
if [ -f "$BACKUP_PATH/MANIFEST.txt" ]; then
    echo "=== Backup Manifest ==="
    cat "$BACKUP_PATH/MANIFEST.txt"
    echo "======================="
    echo ""
fi

# Confirm restore
if ! $FORCE && ! $DRY_RUN; then
    read -p "Restore this backup? This may overwrite existing data. (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Restore cancelled"
        [ -n "$TEMP_DIR" ] && rm -rf "$TEMP_DIR"
        exit 0
    fi
fi

# ============================================================================
# Restore Config
# ============================================================================
if ! $DATA_ONLY; then
    log_info "Restoring configuration..."
    
    # Scripts
    if [ -d "$BACKUP_PATH/scripts" ]; then
        if $DRY_RUN; then
            echo "  Would restore: scripts/*.sh, *.env, *.md"
        else
            cp -v "$BACKUP_PATH/scripts"/*.sh "$PROJECT_DIR/" 2>/dev/null || true
            cp -v "$BACKUP_PATH/scripts"/*.env "$PROJECT_DIR/" 2>/dev/null || true
            cp -v "$BACKUP_PATH/scripts"/*.md "$PROJECT_DIR/" 2>/dev/null || true
            chmod +x "$PROJECT_DIR"/*.sh 2>/dev/null || true
            log_ok "Scripts restored"
        fi
    fi
    
    # Python modules
    if [ -d "$BACKUP_PATH/lib" ]; then
        if $DRY_RUN; then
            echo "  Would restore: lib/ ($(ls "$BACKUP_PATH/lib"/*.py 2>/dev/null | wc -l) files)"
        else
            mkdir -p "$PROJECT_DIR/lib"
            cp -r "$BACKUP_PATH/lib"/* "$PROJECT_DIR/lib/"
            log_ok "Python modules restored"
        fi
    fi
    
    # Web UI
    if [ -d "$BACKUP_PATH/webui" ]; then
        if $DRY_RUN; then
            echo "  Would restore: webui/"
        else
            mkdir -p "$PROJECT_DIR/webui"
            cp -r "$BACKUP_PATH/webui"/* "$PROJECT_DIR/webui/"
            log_ok "Web UI restored"
        fi
    fi
fi

# ============================================================================
# Restore Data
# ============================================================================
if ! $CONFIG_ONLY; then
    log_info "Restoring data..."
    
    # Cache
    if [ -d "$BACKUP_PATH/cache" ]; then
        if $DRY_RUN; then
            echo "  Would restore: cache/"
        else
            mkdir -p "$PROJECT_DIR/cache"
            cp -r "$BACKUP_PATH/cache"/* "$PROJECT_DIR/cache/" 2>/dev/null || true
            log_ok "Cache restored"
        fi
    fi
    
    # Tracking
    if [ -d "$BACKUP_PATH/.ingest_tracking" ]; then
        if $DRY_RUN; then
            echo "  Would restore: .ingest_tracking/"
        else
            mkdir -p "$PROJECT_DIR/.ingest_tracking"
            cp -r "$BACKUP_PATH/.ingest_tracking"/* "$PROJECT_DIR/.ingest_tracking/" 2>/dev/null || true
            log_ok "Tracking data restored"
        fi
    fi
    
    # Qdrant data
    if [ -d "$BACKUP_PATH/qdrant" ]; then
        if $DRY_RUN; then
            echo "  Would restore: Qdrant data to $QDRANT_DATA_DIR"
        else
            log_info "Restoring Qdrant data..."
            
            # Stop Qdrant if running
            if docker ps --format '{{.Names}}' | grep -q "^qdrant$"; then
                log_info "Stopping Qdrant..."
                docker stop qdrant
                sleep 2
            fi
            
            # Restore data
            mkdir -p "$QDRANT_DATA_DIR"
            rm -rf "$QDRANT_DATA_DIR"/*
            cp -r "$BACKUP_PATH/qdrant"/* "$QDRANT_DATA_DIR/"
            chmod -R 777 "$QDRANT_DATA_DIR"
            log_ok "Qdrant data restored"
            
            # Restart Qdrant
            if docker ps -a --format '{{.Names}}' | grep -q "^qdrant$"; then
                log_info "Starting Qdrant..."
                docker start qdrant
                sleep 5
            fi
        fi
    fi
    
    # Documents
    if [ -d "$BACKUP_PATH/documents" ]; then
        if $DRY_RUN; then
            echo "  Would restore: documents/"
        else
            mkdir -p "$PROJECT_DIR/documents"
            cp -r "$BACKUP_PATH/documents"/* "$PROJECT_DIR/documents/" 2>/dev/null || true
            log_ok "Documents restored"
        fi
    fi
fi

# Cleanup temp
if [ -n "$TEMP_DIR" ]; then
    rm -rf "$TEMP_DIR"
fi

if $DRY_RUN; then
    echo ""
    echo "Dry run complete. No changes made."
else
    echo ""
    echo "============================================"
    echo " Restore Complete"
    echo "============================================"
    echo ""
    echo "Verify with: ./status.sh"
fi
EOFRESTORE
chmod +x "$PROJECT_DIR/restore.sh"
log_ok "restore.sh"

# ============================================================================
# Create scheduled backup script
# ============================================================================
log_info "Creating scheduled-backup.sh..."
cat > "$PROJECT_DIR/scheduled-backup.sh" << 'EOFSCHEDULED'
#!/bin/bash
# RAG System - Scheduled Backup
# For use with cron

cd "$(dirname "$0")"
source ./config.env 2>/dev/null || true

BACKUP_DIR="${BACKUP_DIR:-/mnt/dietpi_userdata/rag-backups}"
KEEP_DAYS="${BACKUP_KEEP_DAYS:-7}"
LOG_FILE="$BACKUP_DIR/backup.log"

mkdir -p "$BACKUP_DIR"

echo "$(date): Starting scheduled backup" >> "$LOG_FILE"

# Run backup
./backup.sh --full >> "$LOG_FILE" 2>&1

# Cleanup old backups
if [ "$KEEP_DAYS" -gt 0 ]; then
    find "$BACKUP_DIR" -name "rag-backup-*.tar.gz" -mtime +$KEEP_DAYS -delete
    echo "$(date): Cleaned up backups older than $KEEP_DAYS days" >> "$LOG_FILE"
fi

echo "$(date): Backup complete" >> "$LOG_FILE"
EOFSCHEDULED
chmod +x "$PROJECT_DIR/scheduled-backup.sh"
log_ok "scheduled-backup.sh"

# ============================================================================
# Create cron installer
# ============================================================================
log_info "Creating install-cron.sh..."
cat > "$PROJECT_DIR/install-cron.sh" << EOFCRON
#!/bin/bash
# Install daily backup cron job
PROJECT_DIR="$PROJECT_DIR"

CRON_LINE="0 3 * * * \$PROJECT_DIR/scheduled-backup.sh"

echo "Installing cron job for daily backup at 3:00 AM..."
(crontab -l 2>/dev/null | grep -v "scheduled-backup.sh"; echo "\$CRON_LINE") | crontab -

echo "[OK] Cron job installed"
echo ""
echo "View with: crontab -l"
echo "Remove with: crontab -e"
EOFCRON
chmod +x "$PROJECT_DIR/install-cron.sh"
log_ok "install-cron.sh"

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo ""
echo "============================================"
echo " Backup/Restore Setup Complete"
echo "============================================"
echo ""
echo "Backup directory: $BACKUP_DIR"
echo ""
echo "Usage:"
echo "  ./backup.sh                    # Full backup"
echo "  ./backup.sh --data             # Data only"
echo "  ./backup.sh --config           # Config only"
echo "  ./backup.sh --list             # List backups"
echo ""
echo "  ./restore.sh <backup.tar.gz>   # Restore backup"
echo "  ./restore.sh <backup> --dry-run"
echo ""
echo "Scheduled backups:"
echo "  ./install-cron.sh              # Daily at 3 AM"
echo "  ./scheduled-backup.sh          # Manual run"

# ASSERTION: plain_ascii=true, full_backup=true, restore=true
