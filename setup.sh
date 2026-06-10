#!/bin/bash

DONE=/workspace/.setup_done_kame  # separate checkpoint dir from MoshiRAG
log() { echo "[kame-setup] $(date '+%H:%M:%S') $*"; }

log "Starting KAME setup..."
mkdir -p $DONE

cd /workspace

# Clone repo if not already present
if [ ! -d "kame" ]; then
    log "Cloning KAME repo..."
    git clone https://github.com/flhkl1/kame_cdr.git
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
ASR_FLAG=""
if [ -n "$GOOGLE_CREDENTIALS_B64" ]; then
    log "Writing Google Cloud credentials..."
    echo "$GOOGLE_CREDENTIALS_B64" | base64 -d > /workspace/google-credentials.json
    export GOOGLE_APPLICATION_CREDENTIALS=/workspace/google-credentials.json
    log "Google Cloud credentials written."
else
    log "WARNING: GOOGLE_CREDENTIALS_B64 not set — starting without ASR (speech-to-text disabled)."
    ASR_FLAG="--no-enable-asr"
fi

# Check OpenAI key
if [ -z "$OPENAI_API_KEY" ]; then
    log "WARNING: OPENAI_API_KEY not set — oracle LLM will not work."
fi

# Install Python package (always runs — container disk is wiped on restart)
log "Installing KAME Python dependencies..."
pip install -e /workspace/kame
log "Python dependencies done."

# Fix PYTHONPATH so kame package is found (src layout)
export PYTHONPATH=/workspace/kame/src:$PYTHONPATH

log "Setup complete. Starting KAME server on port 8998... (ASR_FLAG=$ASR_FLAG)"

# Start KAME oracle server
python3 -m kame.server_oracle_parallel \
    --hf-repo SakanaAI/kame \
    --host 0.0.0.0 \
    --port 8998 \
    --device cuda \
    $ASR_FLAG
