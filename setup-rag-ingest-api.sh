#!/bin/bash
# setup-rag-ingest-api.sh
# RAG System - LAN Ingest API Setup
#
# Sibling generator to setup-rag-ingest.sh. Writes lib/ingest_api.py and
# installs rag-ingest-api.service (systemd) only -- never touches
# setup-rag-ingest.sh or ingest.sh. The API is a thin wrapper: every ingest
# job it runs shells out to the real ./ingest.sh, so the CLI and the LAN API
# can never drift apart.
#
# Test-lab component: binds 0.0.0.0, no authentication, no TLS, no rate
# limiting by design (see README "Remote ingestion (LAN API)"). Only run
# this on a trusted LAN.
set -e

log_ok()   { echo "[OK] $1"; }
log_err()  { echo "[ERROR] $1" >&2; }
log_info() { echo "[INFO] $1"; }

if [ "$(id -u)" -ne 0 ]; then
    log_err "This script must be run as root (installs a systemd unit)."
    exit 1
fi

PROJECT_DIR="${1:-$(pwd)}"

echo "============================================"
echo " RAG System - Ingest API Setup"
echo "============================================"
echo ""

mkdir -p "$PROJECT_DIR"/lib "$PROJECT_DIR"/logs
cd "$PROJECT_DIR"

[ -f "./config.env" ] && { set -a; source ./config.env; set +a; }

if [ ! -x "./ingest.sh" ]; then
    log_err "./ingest.sh not found or not executable - run setup-rag-ingest.sh first."
    exit 1
fi

# ----------------------------------------------------------------------------
# Required dependencies: fastapi + uvicorn. Unlike pydantic-ai in
# setup-rag-agentic-query.sh, there's no fallback mode here - without these
# the API can't run at all, so a failed install is fatal.
# ----------------------------------------------------------------------------
echo "Installing fastapi + uvicorn..."
PIP_FLAGS="--break-system-packages --root-user-action=ignore"
pip3 install --help 2>&1 | grep -q "break-system-packages" || PIP_FLAGS=""
pip3 install --quiet fastapi "uvicorn[standard]" python-multipart $PIP_FLAGS 2>/dev/null \
    && log_ok "fastapi + uvicorn installed" \
    || { log_err "fastapi/uvicorn required for the ingest API"; exit 1; }

log_info "Creating lib/ingest_api.py..."
cat > "$PROJECT_DIR/lib/ingest_api.py" << 'EOFPY'
"""LAN-reachable HTTP API for triggering ./ingest.sh remotely.

Test-lab component: intentionally unauthenticated, no TLS, no rate limiting
(see README "Remote ingestion (LAN API)"). Only run this on a trusted LAN.

Every job shells out to the real ./ingest.sh (subprocess), so this module
NEVER reimplements ingestion logic - the CLI and the API always behave
identically. Jobs are tracked in-memory only: a service restart loses job
history, which is acceptable for a lab tool.
"""
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _load_env_value(key: str, default: str) -> str:
    """Read KEY=value from config.env, same default ingest.sh falls back to."""
    config_path = PROJECT_DIR / "config.env"
    if config_path.exists():
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return default


DOCUMENTS_DIR = Path(_load_env_value("DOCUMENTS_DIR", "./documents")).expanduser().resolve()
API_PORT = int(_load_env_value("INGEST_API_PORT", "8090"))
UPLOAD_STAGING_DIR = DOCUMENTS_DIR / "_api_uploads"

app = FastAPI(title="Rag4DietPI Ingest API")

_jobs_lock = threading.Lock()
_jobs = {}  # job_id -> {"proc": Popen, "cmd": [...], "log_path": str, "started_at": float}
_JOB_HISTORY_LIMIT = 200


class IngestRequest(BaseModel):
    path: Optional[str] = None
    url: Optional[str] = None
    force: bool = False
    recreate: bool = False


def _resolve_target_path(subpath: str) -> Path:
    """Resolve `subpath` under DOCUMENTS_DIR, rejecting anything that escapes it."""
    if os.path.isabs(subpath):
        raise HTTPException(status_code=400, detail="path must be relative to the documents directory")
    candidate = (DOCUMENTS_DIR / subpath).resolve()
    try:
        candidate.relative_to(DOCUMENTS_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"path escapes the documents directory: {subpath}")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"path not found: {candidate}")
    return candidate


def _start_job(args: list) -> str:
    job_id = uuid.uuid4().hex[:12]
    log_path = LOG_DIR / f"ingest-{job_id}.log"
    cmd = [str(PROJECT_DIR / "ingest.sh")] + args
    with open(log_path, "wb") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            shell=False,
        )
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "cmd": cmd,
            "proc": proc,
            "log_path": str(log_path),
            "started_at": time.time(),
        }
        # Cap in-memory history so a long-lived service can't grow unbounded.
        if len(_jobs) > _JOB_HISTORY_LIMIT:
            oldest = min(_jobs, key=lambda k: _jobs[k]["started_at"])
            _jobs.pop(oldest, None)
    return job_id


def _job_status(job: dict) -> str:
    returncode = job["proc"].poll()
    if returncode is None:
        return "running"
    return "done" if returncode == 0 else "failed"


@app.get("/")
def health():
    return {"status": "ok", "documents_dir": str(DOCUMENTS_DIR)}


@app.post("/ingest")
def ingest(req: IngestRequest):
    if bool(req.path) == bool(req.url):
        raise HTTPException(status_code=400, detail="provide exactly one of 'path' or 'url'")
    args = []
    if req.force:
        args.append("--force")
    if req.recreate:
        args.append("--recreate")
    if req.url:
        args += ["--url", req.url]
    else:
        args.append(str(_resolve_target_path(req.path)))
    job_id = _start_job(args)
    return {"job_id": job_id, "status": "queued"}


@app.post("/ingest/upload")
async def ingest_upload(files: list[UploadFile] = File(...), force: bool = False, recreate: bool = False):
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")
    job_id = uuid.uuid4().hex[:12]
    staging_dir = UPLOAD_STAGING_DIR / job_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = staging_dir / Path(f.filename).name  # strip any path components
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
    args = []
    if force:
        args.append("--force")
    if recreate:
        args.append("--recreate")
    args.append(str(staging_dir))
    real_job_id = _start_job(args)
    return {"job_id": real_job_id, "status": "queued", "staged_files": [f.filename for f in files]}


@app.get("/status/{job_id}")
def status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    log_tail = ""
    log_path = Path(job["log_path"])
    if log_path.exists():
        log_tail = log_path.read_bytes()[-8000:].decode("utf-8", errors="replace")
    return {
        "job_id": job_id,
        "status": _job_status(job),
        "started_at": job["started_at"],
        "cmd": job["cmd"],
        "log_tail": log_tail,
    }


@app.get("/jobs")
def jobs():
    with _jobs_lock:
        items = sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True)
        return [
            {"job_id": j["job_id"], "status": _job_status(j), "started_at": j["started_at"], "cmd": j["cmd"]}
            for j in items
        ]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
EOFPY
chmod 644 "$PROJECT_DIR/lib/ingest_api.py"
log_ok "lib/ingest_api.py"

# ----------------------------------------------------------------------------
# config.env: append INGEST_API_PORT if not already present (idempotent).
# ----------------------------------------------------------------------------
if [ -f "./config.env" ] && ! grep -q '^INGEST_API_PORT=' ./config.env; then
    log_info "Adding INGEST_API_PORT to config.env..."
    cat >> ./config.env << 'EOFCFG'

# Ingest API (setup-rag-ingest-api.sh only). LAN-reachable HTTP wrapper
# around ./ingest.sh -- test-lab component, no auth/TLS/rate-limiting.
INGEST_API_PORT=8090
EOFCFG
    log_ok "config.env updated"
fi

# ----------------------------------------------------------------------------
# systemd unit
# ----------------------------------------------------------------------------
SERVICE_NAME="rag-ingest-api.service"
INGEST_API_PORT="${INGEST_API_PORT:-8090}"

log_info "Installing $SERVICE_NAME..."
cat > "/etc/systemd/system/$SERVICE_NAME" << SERVICE
[Unit]
Description=Rag4DietPI - LAN Ingest API (test-lab, no auth/TLS)
After=network.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_DIR}
ExecStart=/usr/bin/python3 ${PROJECT_DIR}/lib/ingest_api.py
Restart=on-failure
RestartSec=3

# Hardening (root is required: ingest.sh writes under DOCUMENTS_DIR, which
# defaults to /root/documents, so ProtectHome is intentionally left off).
NoNewPrivileges=true
PrivateTmp=true
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictSUIDSGID=true
RestrictNamespaces=true
LockPersonality=true
SystemCallArchitectures=native

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable "$SERVICE_NAME" >/dev/null 2>&1 || true
systemctl restart "$SERVICE_NAME"
log_ok "$SERVICE_NAME installed and started"

# ============================================================================
# Verification
# ============================================================================
echo ""
echo "=== Verification ==="
[ -f "$PROJECT_DIR/lib/ingest_api.py" ] && log_ok "lib/ingest_api.py" || log_err "lib/ingest_api.py"
python3 -c "import ast; ast.parse(open('$PROJECT_DIR/lib/ingest_api.py').read())" 2>/dev/null \
    && log_ok "lib/ingest_api.py (valid Python syntax)" || log_err "lib/ingest_api.py has a syntax error"
systemctl is-active --quiet "$SERVICE_NAME" \
    && log_ok "$SERVICE_NAME is active" \
    || log_err "$SERVICE_NAME is not active - check: systemctl status $SERVICE_NAME"
sleep 1
curl -sf "http://127.0.0.1:${INGEST_API_PORT}/" >/dev/null \
    && log_ok "API responding on :${INGEST_API_PORT}" \
    || log_err "API not responding on :${INGEST_API_PORT} - check: journalctl -u $SERVICE_NAME"

echo ""
echo "============================================"
echo " Ingest API Setup Complete"
echo "============================================"
echo ""
echo "TEST LAB ONLY: no authentication, no TLS, no rate limiting."
echo "Anyone on the LAN who can reach this host can trigger ingestion."
echo ""
echo "Endpoints (from any LAN device):"
echo "  POST http://<host>:${INGEST_API_PORT}/ingest"
echo "       {\"path\": \"subdir/under/documents\"}  or  {\"url\": \"https://example.com\"}"
echo "       optional: \"force\": true, \"recreate\": true"
echo "  POST http://<host>:${INGEST_API_PORT}/ingest/upload   (multipart file upload)"
echo "  GET  http://<host>:${INGEST_API_PORT}/status/{job_id}"
echo "  GET  http://<host>:${INGEST_API_PORT}/jobs"
echo ""
echo "Config (config.env):"
echo "  INGEST_API_PORT=8090"
echo ""
echo "Service:"
echo "  systemctl status $SERVICE_NAME"
echo "  journalctl -u $SERVICE_NAME -f"
