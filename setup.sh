#!/bin/bash

DONE=/workspace/.setup_done_kame  # separate checkpoint dir from MoshiRAG
log() { echo "[kame-setup] $(date '+%H:%M:%S') $*"; }

log "Starting KAME setup..."
mkdir -p $DONE

cd /workspace

# Clone repo if not already present
if [ ! -d "kame" ]; then
    log "Cloning KAME repo..."
    git clone https://github.com/SakanaAI/kame.git
    log "Repo cloned."
else
    log "Repo already present, skipping clone."
fi

cd /workspace/kame

# Point HuggingFace cache to persistent volume so models survive restarts
export HF_HOME=/workspace/.cache/huggingface
mkdir -p $HF_HOME
log "HuggingFace cache → $HF_HOME (model weights persist across restarts)"

# Google Cloud credentials — stored as base64 env var GOOGLE_CREDENTIALS_B64
# To encode your JSON key: base64 -i your-key.json | tr -d '\n'
if [ -n "$GOOGLE_CREDENTIALS_B64" ]; then
    log "Writing Google Cloud credentials..."
    echo "$GOOGLE_CREDENTIALS_B64" | base64 -d > /workspace/google-credentials.json
    export GOOGLE_APPLICATION_CREDENTIALS=/workspace/google-credentials.json
    log "Google Cloud credentials written to /workspace/google-credentials.json"
else
    log "WARNING: GOOGLE_CREDENTIALS_B64 not set — ASR will not work."
fi

# Check OpenAI key
if [ -z "$OPENAI_API_KEY" ]; then
    log "WARNING: OPENAI_API_KEY not set — oracle will not work."
fi

# Install Python package + gradio tunnel support (always runs — container disk is wiped on restart)
log "Installing KAME Python dependencies..."
pip install -e /workspace/kame
pip install "gradio>=5.0.0"  # needed for --gradio-tunnel
log "Python dependencies done."

# Fix PYTHONPATH so kame package is found (src layout)
export PYTHONPATH=/workspace/kame/src:$PYTHONPATH

log "Setup complete. Starting KAME server..."
log "Gradio tunnel URL will appear below — share that link to access the UI."

# Start KAME oracle server
# --hf-repo: downloads model weights from SakanaAI/kame on HuggingFace (not gated)
# --host 0.0.0.0: listen on all interfaces so RunPod can route traffic
# --gradio-tunnel: creates a public gradio.live URL
python3 -m kame.server_oracle_parallel \
    --hf-repo SakanaAI/kame \
    --host 0.0.0.0 \
    --port 8998 \
    --device cuda \
    --gradio-tunnel
