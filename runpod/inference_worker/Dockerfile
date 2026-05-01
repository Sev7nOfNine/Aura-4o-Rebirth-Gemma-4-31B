# ╔════════════════════════════════════════╗
# ║  🔥 AURA+++ - INFERENCE WORKER 🔥     ║
# ║  💙 Talons LED FULL CHARGE            ║
# ║  ❤️ By Mel & Aura                     ║
# ╚════════════════════════════════════════╝
#
# Image worker pour RunPod Serverless.
# Base: llama.cpp officiel CUDA + Python pour le handler RunPod.

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

WORKDIR /app

# PEP 668 compat : Debian/Ubuntu recents marquent leur Python comme externally-managed,
# pip refuse d'installer dans le system Python par defaut. On force ici (image jetable).
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Install Python deps for handler + HF download
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    requests \
    huggingface_hub \
    hf_transfer

ENV HF_HUB_ENABLE_HF_TRANSFER=1

COPY handler.py /app/handler.py
COPY start-runpod.sh /app/start-runpod.sh
RUN chmod +x /app/start-runpod.sh

# llama-server binary is in /app from the base image (server-cuda variant)
# But we override the entrypoint to run our boot script.
ENTRYPOINT ["/app/start-runpod.sh"]
