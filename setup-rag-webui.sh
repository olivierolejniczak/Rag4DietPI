#!/bin/bash
# setup-rag-webui.sh
# RAG System - Simple Web UI
# ChatGPT-style conversation interface
# LAN accessible, no authentication
# Plain ASCII output

set -e

log_ok() { echo "[OK] $1"; }
log_err() { echo "[ERROR] $1" >&2; }
log_info() { echo "[INFO] $1"; }

PROJECT_DIR="${1:-$(pwd)}"
WEBUI_PORT="${WEBUI_PORT:-5000}"

echo "============================================"
echo " RAG System - Web UI Setup"
echo "============================================"
echo ""

cd "$PROJECT_DIR"
[ -f "./config.env" ] && source ./config.env

mkdir -p "$PROJECT_DIR"/{webui,lib}

# ============================================================================
# Install dependencies
# ============================================================================
log_info "Installing web dependencies..."
pip3 install flask flask-cors --break-system-packages --quiet 2>/dev/null || \
pip3 install flask flask-cors --quiet 2>/dev/null || true

python3 -c "from flask import Flask" 2>/dev/null && log_ok "Flask" || { log_err "Flask install failed"; exit 1; }

# ============================================================================
# Create Flask backend
# ============================================================================
log_info "Creating web backend..."
cat > "$PROJECT_DIR/webui/app.py" << 'EOFPY'
"""RAG Web UI Backend - Flask API
Simple ChatGPT-style interface for RAG queries
"""

import os
import sys
import json
import time
import uuid
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

# Add lib to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))

app = Flask(__name__, static_folder='static')
CORS(app)

# Conversation storage (in-memory, per session)
conversations = {}

def get_config():
    """Load configuration"""
    return {
        "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        "qdrant_host": os.environ.get("QDRANT_HOST", "http://localhost:6333"),
        "collection": os.environ.get("COLLECTION_NAME", "documents"),
        "model": os.environ.get("LLM_MODEL", "qwen2.5:3b"),
        "top_k": int(os.environ.get("TOP_K", "5")),
        "searxng_url": os.environ.get("SEARXNG_URL", "http://localhost:8085/search"),
    }

def query_rag(question, mode="default", conversation_id=None):
    """Execute RAG query"""
    try:
        # Import RAG modules
        from hybrid_search import hybrid_search
        from llm_helper import llm_generate
        from generation import generate_answer
    except ImportError as e:
        return {"error": f"RAG modules not found: {e}", "answer": None}
    
    config = get_config()
    result = {
        "question": question,
        "mode": mode,
        "chunks": [],
        "answer": None,
        "sources": [],
        "timing": {},
    }
    
    start = time.time()
    
    # Web-only mode
    if mode == "web-only":
        try:
            from web_only_query import web_only_query
            web_result = web_only_query(question)
            result["answer"] = web_result.get("answer", "No web results found.")
            result["sources"] = web_result.get("sources", [])
            result["timing"]["total"] = round(time.time() - start, 2)
            return result
        except Exception as e:
            result["error"] = str(e)
            return result
    
    # RAG search
    search_start = time.time()
    try:
        chunks = hybrid_search(question, top_k=config["top_k"])
        result["chunks"] = chunks[:5] if chunks else []
        result["timing"]["search"] = round(time.time() - search_start, 2)
    except Exception as e:
        result["error"] = f"Search failed: {e}"
        return result
    
    # RAG-only mode - return chunks without LLM
    if mode == "rag-only":
        if chunks:
            result["answer"] = "\n\n".join([
                f"**{c.get('filename', 'Source')}**\n{c.get('text', '')[:500]}..."
                for c in chunks[:3]
            ])
        else:
            result["answer"] = "No relevant documents found."
        result["timing"]["total"] = round(time.time() - start, 2)
        return result
    
    # Generate answer with LLM
    llm_start = time.time()
    try:
        # Get conversation context
        memory_context = ""
        if conversation_id and conversation_id in conversations:
            history = conversations[conversation_id][-3:]  # Last 3 turns
            memory_context = "\n".join([
                f"User: {h['question']}\nAssistant: {h['answer'][:300]}"
                for h in history if h.get('answer')
            ])
        
        answer = generate_answer(
            question, 
            chunks,
            memory_context=memory_context,
            config={"citations": mode == "full", "num_predict": 800}
        )
        result["answer"] = answer
        result["timing"]["llm"] = round(time.time() - llm_start, 2)
    except Exception as e:
        result["error"] = f"Generation failed: {e}"
        result["answer"] = "I encountered an error generating the response."
    
    # Extract sources
    result["sources"] = list(set([
        c.get("filename", "Unknown") for c in chunks[:5] if c.get("filename")
    ]))
    
    result["timing"]["total"] = round(time.time() - start, 2)
    return result

@app.route('/')
def index():
    """Serve main page"""
    return send_from_directory('static', 'index.html')

@app.route('/api/query', methods=['POST'])
def api_query():
    """Handle query request"""
    data = request.get_json() or {}
    question = data.get('question', '').strip()
    mode = data.get('mode', 'default')
    conversation_id = data.get('conversation_id')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Create conversation if needed
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Execute query
    result = query_rag(question, mode, conversation_id)
    result["conversation_id"] = conversation_id
    
    # Store in conversation history
    if result.get("answer"):
        conversations[conversation_id].append({
            "question": question,
            "answer": result["answer"],
            "timestamp": time.strftime("%H:%M:%S"),
        })
        # Limit history
        conversations[conversation_id] = conversations[conversation_id][-10:]
    
    return jsonify(result)

@app.route('/api/query/stream', methods=['POST'])
def api_query_stream():
    """Streaming query response"""
    data = request.get_json() or {}
    question = data.get('question', '').strip()
    mode = data.get('mode', 'default')
    conversation_id = data.get('conversation_id')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    def generate():
        # Send searching status
        yield f"data: {json.dumps({'status': 'searching'})}\n\n"
        
        result = query_rag(question, mode, conversation_id)
        
        # Send chunks found
        if result.get("chunks"):
            yield f"data: {json.dumps({'status': 'found', 'count': len(result['chunks'])})}\n\n"
        
        # Send generating status
        if mode not in ["rag-only"]:
            yield f"data: {json.dumps({'status': 'generating'})}\n\n"
        
        # Send final result
        yield f"data: {json.dumps({'status': 'done', 'result': result})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/conversations', methods=['GET'])
def api_conversations():
    """List active conversations"""
    return jsonify({
        "count": len(conversations),
        "conversations": [
            {"id": cid, "turns": len(turns)}
            for cid, turns in conversations.items()
        ]
    })

@app.route('/api/conversation/<conversation_id>', methods=['GET'])
def api_conversation(conversation_id):
    """Get conversation history"""
    if conversation_id not in conversations:
        return jsonify({"error": "Conversation not found"}), 404
    return jsonify({"history": conversations[conversation_id]})

@app.route('/api/conversation/<conversation_id>', methods=['DELETE'])
def api_conversation_delete(conversation_id):
    """Clear conversation"""
    if conversation_id in conversations:
        del conversations[conversation_id]
    return jsonify({"status": "cleared"})

@app.route('/api/status', methods=['GET'])
def api_status():
    """System status"""
    import requests
    config = get_config()
    status = {
        "qdrant": False,
        "ollama": False,
        "model": config["model"],
        "collection": config["collection"],
    }
    
    try:
        r = requests.get(f"{config['qdrant_host']}/collections/{config['collection']}", timeout=3)
        if r.status_code == 200:
            data = r.json()
            status["qdrant"] = True
            status["points"] = data.get("result", {}).get("points_count", 0)
    except:
        pass
    
    try:
        r = requests.get(f"{config['ollama_host']}/api/tags", timeout=3)
        if r.status_code == 200:
            status["ollama"] = True
    except:
        pass
    
    return jsonify(status)

if __name__ == '__main__':
    port = int(os.environ.get('WEBUI_PORT', 5000))
    print(f"Starting RAG Web UI on http://0.0.0.0:{port}")
    print(f"LAN access: http://<your-ip>:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
EOFPY
log_ok "app.py"

# ============================================================================
# Create HTML frontend
# ============================================================================
log_info "Creating web frontend..."
mkdir -p "$PROJECT_DIR/webui/static"

cat > "$PROJECT_DIR/webui/static/index.html" << 'EOFHTML'
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Assistant</title>
    <style>
        :root {
            --bg-primary: #343541;
            --bg-secondary: #444654;
            --bg-input: #40414f;
            --text-primary: #ececf1;
            --text-secondary: #c5c5d2;
            --accent: #10a37f;
            --accent-hover: #1a7f64;
            --border: #565869;
            --user-bg: #343541;
            --assistant-bg: #444654;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        /* Header */
        .header {
            background: var(--bg-secondary);
            padding: 12px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
        }
        
        .header h1 {
            font-size: 1.1rem;
            font-weight: 600;
        }
        
        .header-controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .mode-select {
            background: var(--bg-input);
            color: var(--text-primary);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
        }
        
        .btn-clear {
            background: transparent;
            color: var(--text-secondary);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .btn-clear:hover {
            background: var(--bg-input);
            color: var(--text-primary);
        }
        
        /* Chat container */
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px 0;
        }
        
        .chat-messages {
            max-width: 800px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        /* Messages */
        .message {
            padding: 20px;
            margin-bottom: 0;
            border-bottom: 1px solid var(--border);
        }
        
        .message.user {
            background: var(--user-bg);
        }
        
        .message.assistant {
            background: var(--assistant-bg);
        }
        
        .message-content {
            max-width: 800px;
            margin: 0 auto;
            display: flex;
            gap: 16px;
        }
        
        .message-avatar {
            width: 30px;
            height: 30px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            flex-shrink: 0;
        }
        
        .message.user .message-avatar {
            background: #5436da;
        }
        
        .message.assistant .message-avatar {
            background: var(--accent);
        }
        
        .message-text {
            flex: 1;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
        }
        
        .message-text p {
            margin-bottom: 12px;
        }
        
        .message-text p:last-child {
            margin-bottom: 0;
        }
        
        .message-text code {
            background: var(--bg-input);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Consolas', monospace;
            font-size: 0.9em;
        }
        
        .message-text pre {
            background: var(--bg-input);
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 12px 0;
        }
        
        .message-text pre code {
            background: none;
            padding: 0;
        }
        
        /* Sources */
        .sources {
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid var(--border);
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .sources-label {
            font-weight: 600;
            margin-bottom: 6px;
        }
        
        .source-tag {
            display: inline-block;
            background: var(--bg-input);
            padding: 3px 8px;
            border-radius: 4px;
            margin: 2px 4px 2px 0;
            font-size: 0.8rem;
        }
        
        /* Timing */
        .timing {
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 8px;
            opacity: 0.7;
        }
        
        /* Status indicator */
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 20px;
            color: var(--text-secondary);
            font-style: italic;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent);
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        
        /* Input area */
        .input-container {
            background: var(--bg-primary);
            padding: 20px;
            border-top: 1px solid var(--border);
        }
        
        .input-wrapper {
            max-width: 800px;
            margin: 0 auto;
            position: relative;
        }
        
        .input-box {
            display: flex;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            transition: border-color 0.2s;
        }
        
        .input-box:focus-within {
            border-color: var(--accent);
        }
        
        #question-input {
            flex: 1;
            background: transparent;
            border: none;
            padding: 14px 16px;
            color: var(--text-primary);
            font-size: 1rem;
            resize: none;
            min-height: 52px;
            max-height: 200px;
            line-height: 1.5;
        }
        
        #question-input:focus {
            outline: none;
        }
        
        #question-input::placeholder {
            color: var(--text-secondary);
        }
        
        .send-btn {
            background: var(--accent);
            color: white;
            border: none;
            padding: 0 16px;
            cursor: pointer;
            transition: background 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .send-btn:hover:not(:disabled) {
            background: var(--accent-hover);
        }
        
        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .send-btn svg {
            width: 20px;
            height: 20px;
        }
        
        /* Footer hint */
        .input-hint {
            text-align: center;
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 8px;
            opacity: 0.7;
        }
        
        /* Welcome screen */
        .welcome {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-secondary);
        }
        
        .welcome h2 {
            font-size: 1.5rem;
            color: var(--text-primary);
            margin-bottom: 12px;
        }
        
        .welcome p {
            margin-bottom: 30px;
        }
        
        .suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            max-width: 600px;
            margin: 0 auto;
        }
        
        .suggestion {
            background: var(--bg-input);
            border: 1px solid var(--border);
            padding: 10px 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9rem;
        }
        
        .suggestion:hover {
            background: var(--bg-secondary);
            border-color: var(--accent);
        }
        
        /* Responsive */
        @media (max-width: 600px) {
            .header {
                flex-direction: column;
                gap: 10px;
            }
            
            .message-content {
                gap: 10px;
            }
            
            .message-avatar {
                width: 26px;
                height: 26px;
                font-size: 12px;
            }
            
            .suggestions {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>RAG Assistant </h1>
        <div class="header-controls">
            <select id="mode-select" class="mode-select">
                <option value="default">Standard</option>
                <option value="rag-only">RAG Only (fast)</option>
                <option value="web-only">Web Search</option>
                <option value="full">Full (detailed)</option>
            </select>
            <button class="btn-clear" onclick="clearConversation()">Nouvelle conversation</button>
        </div>
    </div>
    
    <div class="chat-container" id="chat-container">
        <div class="chat-messages" id="chat-messages">
            <div class="welcome" id="welcome">
                <h2>Comment puis-je vous aider ?</h2>
                <p>Posez une question sur vos documents ou recherchez sur le web.</p>
                <div class="suggestions">
                    <div class="suggestion" onclick="askQuestion('Quels sont les documents disponibles ?')">
                        Quels documents sont disponibles ?
                    </div>
                    <div class="suggestion" onclick="askQuestion('Resume les points importants')">
                        Resume les points importants
                    </div>
                    <div class="suggestion" onclick="askQuestion('Explique-moi le dernier rapport')">
                        Explique le dernier rapport
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="input-container">
        <div class="input-wrapper">
            <div class="input-box">
                <textarea 
                    id="question-input" 
                    placeholder="Posez votre question..."
                    rows="1"
                    onkeydown="handleKeyDown(event)"
                    oninput="autoResize(this)"
                ></textarea>
                <button class="send-btn" id="send-btn" onclick="sendQuestion()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                    </svg>
                </button>
            </div>
            <div class="input-hint">Entree pour envoyer, Shift+Entree pour nouvelle ligne</div>
        </div>
    </div>

    <script>
        let conversationId = null;
        let isProcessing = false;
        
        function autoResize(el) {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 200) + 'px';
        }
        
        function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendQuestion();
            }
        }
        
        function askQuestion(text) {
            document.getElementById('question-input').value = text;
            sendQuestion();
        }
        
        function clearConversation() {
            if (conversationId) {
                fetch(`/api/conversation/${conversationId}`, { method: 'DELETE' });
            }
            conversationId = null;
            document.getElementById('chat-messages').innerHTML = `
                <div class="welcome" id="welcome">
                    <h2>Comment puis-je vous aider ?</h2>
                    <p>Posez une question sur vos documents ou recherchez sur le web.</p>
                    <div class="suggestions">
                        <div class="suggestion" onclick="askQuestion('Quels sont les documents disponibles ?')">
                            Quels documents sont disponibles ?
                        </div>
                        <div class="suggestion" onclick="askQuestion('Resume les points importants')">
                            Resume les points importants
                        </div>
                        <div class="suggestion" onclick="askQuestion('Explique-moi le dernier rapport')">
                            Explique le dernier rapport
                        </div>
                    </div>
                </div>
            `;
        }
        
        function formatText(text) {
            if (!text) return '';
            // Basic markdown-like formatting
            return text
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`([^`]+)`/g, '<code>$1</code>')
                .replace(/\n/g, '<br>');
        }
        
        function addMessage(role, content, sources = [], timing = null) {
            const welcome = document.getElementById('welcome');
            if (welcome) welcome.remove();
            
            const messages = document.getElementById('chat-messages');
            const avatar = role === 'user' ? '👤' : '🤖';
            
            let sourcesHtml = '';
            if (sources && sources.length > 0) {
                sourcesHtml = `
                    <div class="sources">
                        <div class="sources-label">Sources:</div>
                        ${sources.map(s => `<span class="source-tag">${s}</span>`).join('')}
                    </div>
                `;
            }
            
            let timingHtml = '';
            if (timing) {
                timingHtml = `<div class="timing">Temps: ${timing.total || 0}s</div>`;
            }
            
            const messageHtml = `
                <div class="message ${role}">
                    <div class="message-content">
                        <div class="message-avatar">${avatar}</div>
                        <div class="message-text">
                            ${formatText(content)}
                            ${sourcesHtml}
                            ${timingHtml}
                        </div>
                    </div>
                </div>
            `;
            
            messages.insertAdjacentHTML('beforeend', messageHtml);
            scrollToBottom();
        }
        
        function addStatus(text) {
            const messages = document.getElementById('chat-messages');
            const statusId = 'status-' + Date.now();
            messages.insertAdjacentHTML('beforeend', `
                <div class="message assistant" id="${statusId}">
                    <div class="message-content">
                        <div class="message-avatar">🤖</div>
                        <div class="status">
                            <div class="status-dot"></div>
                            <span>${text}</span>
                        </div>
                    </div>
                </div>
            `);
            scrollToBottom();
            return statusId;
        }
        
        function removeStatus(statusId) {
            const el = document.getElementById(statusId);
            if (el) el.remove();
        }
        
        function scrollToBottom() {
            const container = document.getElementById('chat-container');
            container.scrollTop = container.scrollHeight;
        }
        
        async function sendQuestion() {
            if (isProcessing) return;
            
            const input = document.getElementById('question-input');
            const question = input.value.trim();
            if (!question) return;
            
            const mode = document.getElementById('mode-select').value;
            
            // Clear input
            input.value = '';
            input.style.height = 'auto';
            
            // Add user message
            addMessage('user', question);
            
            // Show status
            isProcessing = true;
            document.getElementById('send-btn').disabled = true;
            
            let statusText = 'Recherche en cours...';
            if (mode === 'web-only') statusText = 'Recherche web...';
            if (mode === 'rag-only') statusText = 'Recherche documents...';
            const statusId = addStatus(statusText);
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question,
                        mode,
                        conversation_id: conversationId
                    })
                });
                
                const data = await response.json();
                removeStatus(statusId);
                
                if (data.error && !data.answer) {
                    addMessage('assistant', `Erreur: ${data.error}`);
                } else {
                    conversationId = data.conversation_id;
                    addMessage('assistant', data.answer || 'Pas de reponse.', data.sources, data.timing);
                }
            } catch (error) {
                removeStatus(statusId);
                addMessage('assistant', `Erreur de connexion: ${error.message}`);
            }
            
            isProcessing = false;
            document.getElementById('send-btn').disabled = false;
            input.focus();
        }
        
        // Check status on load
        fetch('/api/status')
            .then(r => r.json())
            .then(status => {
                if (!status.qdrant) {
                    console.warn('Qdrant not available');
                }
                if (!status.ollama) {
                    console.warn('Ollama not available');
                }
            })
            .catch(console.error);
    </script>
</body>
</html>
EOFHTML
log_ok "index.html"

# ============================================================================
# Create launcher script
# ============================================================================
log_info "Creating launcher..."
cat > "$PROJECT_DIR/webui.sh" << EOFLAUNCHER
#!/bin/bash
# RAG Web UI Launcher
cd "\$(dirname "\$0")"
source ./config.env 2>/dev/null || true

export WEBUI_PORT=\${WEBUI_PORT:-$WEBUI_PORT}
export OLLAMA_HOST=\${OLLAMA_HOST:-http://localhost:11434}
export QDRANT_HOST=\${QDRANT_HOST:-http://localhost:6333}
export COLLECTION_NAME=\${COLLECTION_NAME:-documents}
export LLM_MODEL=\${LLM_MODEL:-qwen2.5:3b}
export SEARXNG_URL=\${SEARXNG_URL:-http://localhost:8085/search}

# Get LAN IP
LAN_IP=\$(hostname -I 2>/dev/null | awk '{print \$1}')
[ -z "\$LAN_IP" ] && LAN_IP="localhost"

echo "============================================"
echo " RAG Web UI"
echo "============================================"
echo ""
echo "Local:  http://localhost:\$WEBUI_PORT"
echo "LAN:    http://\$LAN_IP:\$WEBUI_PORT"
echo ""
echo "Model:  \$LLM_MODEL"
echo "Qdrant: \$QDRANT_HOST"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================"
echo ""

cd webui
python3 app.py
EOFLAUNCHER
chmod +x "$PROJECT_DIR/webui.sh"
log_ok "webui.sh"

# ============================================================================
# Create systemd service (optional)
# ============================================================================
log_info "Creating systemd service file..."
cat > "$PROJECT_DIR/webui/rag-webui.service" << EOFSVC
[Unit]
Description=RAG Web UI
After=network.target docker.service

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_DIR
ExecStart=/usr/bin/python3 $PROJECT_DIR/webui/app.py
Restart=on-failure
RestartSec=5
Environment=WEBUI_PORT=$WEBUI_PORT
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOFSVC
log_ok "rag-webui.service"

# ============================================================================
# Verification
# ============================================================================
echo ""
echo "=== Verification ==="
python3 -c "from flask import Flask" 2>/dev/null && log_ok "Flask" || log_err "Flask"
[ -f "$PROJECT_DIR/webui/app.py" ] && log_ok "app.py" || log_err "app.py"
[ -f "$PROJECT_DIR/webui/static/index.html" ] && log_ok "index.html" || log_err "index.html"
[ -f "$PROJECT_DIR/webui.sh" ] && log_ok "webui.sh" || log_err "webui.sh"

# Get LAN IP
LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$LAN_IP" ] && LAN_IP="<your-ip>"

echo ""
echo "============================================"
echo " Web UI Setup Complete"
echo "============================================"
echo ""
echo "Start the web UI:"
echo "  ./webui.sh"
echo ""
echo "Access from:"
echo "  Local:  http://localhost:$WEBUI_PORT"
echo "  LAN:    http://$LAN_IP:$WEBUI_PORT"
echo ""
echo "Install as service (auto-start):"
echo "  sudo cp webui/rag-webui.service /etc/systemd/system/"
echo "  sudo systemctl enable rag-webui"
echo "  sudo systemctl start rag-webui"
echo ""
echo "Features:"
echo "  - ChatGPT-style conversation"
echo "  - Standard / RAG-only / Web / Full modes"
echo "  - Conversation history"
echo "  - Source citations"
echo "  - Mobile-friendly"

# ASSERTION: plain_ascii=true, no_auth=true, lan_accessible=true
