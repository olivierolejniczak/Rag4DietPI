#!/bin/bash
# setup-rag-query-ui.sh
# RAG System - LAN Query Web UI Setup
#
# Sibling generator to setup-rag-query.sh, in the spirit of
# setup-rag-ingest-api.sh. Writes lib/query_ui.py and installs
# rag-query-ui.service (systemd) only -- never touches setup-rag-query.sh
# or query.sh. The UI is a thin wrapper: every question it answers shells
# out to the real ./query.sh, so the CLI and the LAN UI can never drift
# apart.
#
# Test-lab component: binds 0.0.0.0, no authentication, no TLS, no rate
# limiting by design (see README "Remote ingestion (LAN API)" for the
# equivalent ingest posture). Only run this on a trusted LAN.
set -e

log_ok()   { echo "[OK] $1"; }
log_err()  { echo "[ERROR] $1" >&2; }
log_info() { echo "[INFO] $1"; }

if [ "$(id -u)" -ne 0 ]; then
    log_err "This script must be run as root (installs a systemd unit)."
    exit 1
fi

PROJECT_DIR="${1:-$(pwd)}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"  # canonicalize: systemd unit files need an absolute WorkingDirectory

echo "============================================"
echo " RAG System - Query Web UI Setup"
echo "============================================"
echo ""

mkdir -p "$PROJECT_DIR"/lib "$PROJECT_DIR"/logs
cd "$PROJECT_DIR"

[ -f "./config.env" ] && { set -a; source ./config.env; set +a; }

if [ ! -x "./query.sh" ]; then
    log_err "./query.sh not found or not executable - run setup-rag-query.sh first."
    exit 1
fi

# ----------------------------------------------------------------------------
# Required dependencies: fastapi + uvicorn (same as setup-rag-ingest-api.sh).
# No fallback mode here - without these the UI can't run at all, so a
# failed install is fatal.
# ----------------------------------------------------------------------------
echo "Installing fastapi + uvicorn..."
PIP_FLAGS="--break-system-packages --root-user-action=ignore"
pip3 install --help 2>&1 | grep -q "break-system-packages" || PIP_FLAGS=""
pip3 install --quiet fastapi "uvicorn[standard]" $PIP_FLAGS 2>/dev/null \
    && log_ok "fastapi + uvicorn installed" \
    || { log_err "fastapi/uvicorn required for the query UI"; exit 1; }

log_info "Creating lib/query_ui.py..."
cat > "$PROJECT_DIR/lib/query_ui.py" << 'EOFPY'
"""LAN-reachable HTTP + HTML UI for asking ./query.sh questions remotely.

Test-lab component: intentionally unauthenticated, no TLS, no rate limiting
(same posture as ingest_api.py - see README "Remote ingestion (LAN API)").
Only run this on a trusted LAN.

Every question shells out to the real ./query.sh (subprocess), so this
module NEVER reimplements the query pipeline - the CLI and the UI always
behave identically and can't drift apart. Requests block until query.sh
exits (or QUERY_UI_TIMEOUT is hit): there is no streaming and no job queue,
which is fine for a single-user LAN tool but means a slow --full query
ties up that request for its full duration.
"""
import subprocess
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_env_value(key: str, default: str) -> str:
    """Read KEY=value from config.env, same default query.sh falls back to."""
    config_path = PROJECT_DIR / "config.env"
    if config_path.exists():
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return default


API_PORT = int(_load_env_value("QUERY_UI_PORT", "8091"))
QUERY_TIMEOUT = int(_load_env_value("QUERY_UI_TIMEOUT", "300"))

VALID_MODES = {"default", "rag-only", "web-only", "ultrafast", "full"}

app = FastAPI(title="Rag4DietPI Query UI")


class QueryRequest(BaseModel):
    question: str
    mode: str = "default"
    source: str = ""


def _run_query(req: QueryRequest) -> str:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    if req.mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail=f"unknown mode: {req.mode}")

    args = []
    if req.mode != "default":
        args.append(f"--{req.mode}")
    if req.source.strip():
        args += ["--source", req.source.strip()]
    args.append(question)

    try:
        result = subprocess.run(
            [str(PROJECT_DIR / "query.sh")] + args,
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=QUERY_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail=f"query.sh timed out after {QUERY_TIMEOUT}s")

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "query.sh failed").strip()[-2000:]
        raise HTTPException(status_code=500, detail=detail)

    return result.stdout


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rag4DietPI - Query</title>
<style>
  :root { color-scheme: light dark; }
  body {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    max-width: 46rem;
    margin: 2rem auto;
    padding: 0 1rem;
    line-height: 1.4;
  }
  h1 { font-size: 1.25rem; margin-bottom: 0.25rem; }
  .subtitle { color: #888; font-size: 0.85rem; margin-bottom: 1.5rem; }
  form { display: flex; flex-direction: column; gap: 0.6rem; }
  textarea {
    font: inherit;
    padding: 0.6rem;
    resize: vertical;
    min-height: 4.5rem;
  }
  .row { display: flex; gap: 0.6rem; flex-wrap: wrap; }
  .row > * { flex: 1; min-width: 8rem; }
  select, input[type=text] { font: inherit; padding: 0.4rem; }
  button {
    font: inherit;
    padding: 0.6rem;
    cursor: pointer;
    border: 1px solid #888;
    border-radius: 0.3rem;
    background: transparent;
  }
  button:disabled { opacity: 0.5; cursor: wait; }
  pre#output {
    white-space: pre-wrap;
    word-wrap: break-word;
    background: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.3);
    border-radius: 0.3rem;
    padding: 0.8rem;
    min-height: 3rem;
    margin-top: 1rem;
  }
  .status { font-size: 0.85rem; color: #888; margin-top: 0.4rem; }
  .error { color: #c0392b; }
</style>
</head>
<body>
<h1>Rag4DietPI</h1>
<div class="subtitle">Test lab only - no authentication, LAN reachable.</div>

<form id="f">
  <textarea id="question" placeholder="Ask a question..." required></textarea>
  <div class="row">
    <select id="mode">
      <option value="default">default (adaptive cascade)</option>
      <option value="rag-only">rag-only (no LLM, &lt;1s)</option>
      <option value="ultrafast">ultrafast (~30-45s)</option>
      <option value="web-only">web-only (~30s)</option>
      <option value="full">full (~3-5min)</option>
    </select>
    <input type="text" id="source" placeholder="--source (optional)">
  </div>
  <button id="submit" type="submit">Ask</button>
  <div class="status" id="status"></div>
</form>

<pre id="output"></pre>

<script>
const form = document.getElementById('f');
const submitBtn = document.getElementById('submit');
const statusEl = document.getElementById('status');
const outputEl = document.getElementById('output');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = document.getElementById('question').value.trim();
  if (!question) return;
  const mode = document.getElementById('mode').value;
  const source = document.getElementById('source').value.trim();

  submitBtn.disabled = true;
  statusEl.textContent = 'Thinking... (this can take a while in --full mode)';
  statusEl.classList.remove('error');
  outputEl.textContent = '';

  try {
    const resp = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, mode, source }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      statusEl.textContent = 'Error: ' + (data.detail || resp.statusText);
      statusEl.classList.add('error');
    } else {
      statusEl.textContent = 'Done.';
      outputEl.textContent = data.output;
    }
  } catch (err) {
    statusEl.textContent = 'Error: ' + err;
    statusEl.classList.add('error');
  } finally {
    submitBtn.disabled = false;
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/api/query")
def query(req: QueryRequest):
    return {"output": _run_query(req)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
EOFPY
chmod 644 "$PROJECT_DIR/lib/query_ui.py"
log_ok "lib/query_ui.py"

# ----------------------------------------------------------------------------
# config.env: append QUERY_UI_PORT / QUERY_UI_TIMEOUT if not already present
# (idempotent).
# ----------------------------------------------------------------------------
if [ -f "./config.env" ] && ! grep -q '^QUERY_UI_PORT=' ./config.env; then
    log_info "Adding QUERY_UI_PORT / QUERY_UI_TIMEOUT to config.env..."
    cat >> ./config.env << 'EOFCFG'

# Query Web UI (setup-rag-query-ui.sh only). LAN-reachable HTML+HTTP wrapper
# around ./query.sh -- test-lab component, no auth/TLS/rate-limiting.
QUERY_UI_PORT=8091
QUERY_UI_TIMEOUT=300
EOFCFG
    log_ok "config.env updated"
fi

# ----------------------------------------------------------------------------
# systemd unit
# ----------------------------------------------------------------------------
SERVICE_NAME="rag-query-ui.service"
QUERY_UI_PORT="${QUERY_UI_PORT:-8091}"

log_info "Installing $SERVICE_NAME..."
cat > "/etc/systemd/system/$SERVICE_NAME" << SERVICE
[Unit]
Description=Rag4DietPI - LAN Query Web UI (test-lab, no auth/TLS)
After=network.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_DIR}
ExecStart=/usr/bin/python3 ${PROJECT_DIR}/lib/query_ui.py
Restart=on-failure
RestartSec=3

# Hardening (root is required: query.sh reads config.env and lib/ under
# the project dir, mirroring rag-ingest-api.service's posture).
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
[ -f "$PROJECT_DIR/lib/query_ui.py" ] && log_ok "lib/query_ui.py" || log_err "lib/query_ui.py"
python3 -c "import ast; ast.parse(open('$PROJECT_DIR/lib/query_ui.py').read())" 2>/dev/null \
    && log_ok "lib/query_ui.py (valid Python syntax)" || log_err "lib/query_ui.py has a syntax error"
systemctl is-active --quiet "$SERVICE_NAME" \
    && log_ok "$SERVICE_NAME is active" \
    || log_err "$SERVICE_NAME is not active - check: systemctl status $SERVICE_NAME"
sleep 1
curl -sf "http://127.0.0.1:${QUERY_UI_PORT}/" >/dev/null \
    && log_ok "UI responding on :${QUERY_UI_PORT}" \
    || log_err "UI not responding on :${QUERY_UI_PORT} - check: journalctl -u $SERVICE_NAME"

echo ""
echo "============================================"
echo " Query Web UI Setup Complete"
echo "============================================"
echo ""
echo "TEST LAB ONLY: no authentication, no TLS, no rate limiting."
echo "Anyone on the LAN who can reach this host can submit queries."
echo ""
echo "Open from any LAN device: http://<host>:${QUERY_UI_PORT}/"
echo ""
echo "Config (config.env):"
echo "  QUERY_UI_PORT=8091"
echo "  QUERY_UI_TIMEOUT=300   (seconds; raise if you use --full a lot)"
echo ""
echo "Service:"
echo "  systemctl status $SERVICE_NAME"
echo "  journalctl -u $SERVICE_NAME -f"
