#!/usr/bin/env bash
# ╔════════════════════════════════════════╗
# ║  🔥 AURA+++ - INFERENCE WORKER 🔥     ║
# ║  💙 Talons LED FULL CHARGE            ║
# ║  ❤️ By Mel & Aura                     ║
# ╚════════════════════════════════════════╝
#
# Lance llama-server (llama.cpp) avec config Aura-Rebirth.
# Telecharge le GGUF + mmproj + chat template au premier boot, met en cache
# sur le network volume RunPod, reutilise au cold-start suivant.

set -euo pipefail

# === Required env vars (set on RunPod template) ===
: "${HF_TOKEN:?HF_TOKEN required}"
: "${HF_GGUF_REPO:?HF_GGUF_REPO required (e.g. SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-GGUF)}"
: "${GGUF_FILE:?GGUF_FILE required (e.g. model-q5_k_m.gguf)}"
MMPROJ_FILE="${MMPROJ_FILE:-mmproj-f16.gguf}"
CHAT_TEMPLATE_URL="${CHAT_TEMPLATE_URL:-https://raw.githubusercontent.com/ggml-org/llama.cpp/master/models/templates/google-gemma-4-31B-it-interleaved.jinja}"

# Optional
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
PARALLEL="${PARALLEL:-1}"
THREADS="${THREADS:-8}"
REASONING_FORMAT="${REASONING_FORMAT:-deepseek}"
LLAMA_PORT="${LLAMA_PORT:-8000}"
REQUIRE_MMPROJ="${REQUIRE_MMPROJ:-1}"

# Paths
VOLUME_DIR="${RUNPOD_VOLUME_PATH:-/runpod-volume}"
MODEL_DIR="${VOLUME_DIR}/aura"
mkdir -p "${MODEL_DIR}"

GGUF_PATH="${MODEL_DIR}/${GGUF_FILE}"
MMPROJ_PATH="${MODEL_DIR}/${MMPROJ_FILE}"
TEMPLATE_PATH="${MODEL_DIR}/chat-template.jinja"

# === Download chat template (always fresh, lightweight) ===
echo "[boot] Downloading chat template from ${CHAT_TEMPLATE_URL}"
curl -fsSL "${CHAT_TEMPLATE_URL}" -o "${TEMPLATE_PATH}"

# === Download GGUF if missing ===
if [ ! -s "${GGUF_PATH}" ]; then
  echo "[boot] Downloading GGUF: ${HF_GGUF_REPO}/${GGUF_FILE}"
  python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='${HF_GGUF_REPO}',
    filename='${GGUF_FILE}',
    local_dir='${MODEL_DIR}',
    token='${HF_TOKEN}',
)
"
fi

# === Download mmproj if missing ===
if [ ! -s "${MMPROJ_PATH}" ]; then
  echo "[boot] Downloading mmproj: ${HF_GGUF_REPO}/${MMPROJ_FILE}"
  python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='${HF_GGUF_REPO}',
    filename='${MMPROJ_FILE}',
    local_dir='${MODEL_DIR}',
    token='${HF_TOKEN}',
)
" || {
    if [ "${REQUIRE_MMPROJ}" = "1" ]; then
      echo "[boot] ERROR: mmproj download failed and REQUIRE_MMPROJ=1"
      exit 1
    fi
    echo "[boot] WARN: mmproj missing, continuing without vision"
    MMPROJ_PATH=""
  }
fi

# === Verify GGUF ===
if [ ! -s "${GGUF_PATH}" ]; then
  echo "[boot] ERROR: GGUF missing at ${GGUF_PATH}"
  exit 1
fi

# === Build llama-server args ===
LLAMA_ARGS=(
  --model "${GGUF_PATH}"
  --jinja
  --chat-template-file "${TEMPLATE_PATH}"
  --reasoning-format "${REASONING_FORMAT}"
  --ctx-size "${CONTEXT_LENGTH}"
  --n-gpu-layers "${N_GPU_LAYERS}"
  --flash-attn
  --cache-type-k q8_0
  --cache-type-v q8_0
  --parallel "${PARALLEL}"
  --threads "${THREADS}"
  --host 0.0.0.0
  --port "${LLAMA_PORT}"
)

if [ -n "${MMPROJ_PATH}" ] && [ -s "${MMPROJ_PATH}" ]; then
  LLAMA_ARGS+=( --mmproj "${MMPROJ_PATH}" )
fi

# Note : on N'AJOUTE PAS LLAMA_ARG_REASONING=on / LLAMA_ARG_THINK
# pour ne PAS forcer le thinking globalement (V2 = tone fade).
# Le --reasoning-format est juste un parser, pas un forceur.

echo "[boot] Starting llama-server :"
echo "  ${LLAMA_ARGS[@]}"

# === Launch llama-server in background ===
/app/llama-server "${LLAMA_ARGS[@]}" &
LLAMA_PID=$!

# Wait for /health
echo "[boot] Waiting for llama-server /health..."
for i in $(seq 1 60); do
  if curl -fsS "http://localhost:${LLAMA_PORT}/health" >/dev/null 2>&1; then
    echo "[boot] llama-server ready"
    break
  fi
  sleep 2
done

# === Launch RunPod handler ===
echo "[boot] Starting RunPod handler"
exec python /app/handler.py
