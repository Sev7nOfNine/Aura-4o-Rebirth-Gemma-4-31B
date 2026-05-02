# ╔════════════════════════════════════════╗
# ║  🔥 AURA+++ - TRAIN WORKER 🔥         ║
# ║  💙 Talons LED FULL CHARGE            ║
# ║  ❤️ By Mel & Aura                     ║
# ╚════════════════════════════════════════╝
#
# Image worker pre-bakee pour le training LoRA AURA+++ REBIRTH.
# Toutes les deps Unsloth/transformers sont installees a build-time pour
# eviter l'OOM cgroup observe sur les essais 3-5 (cf. docs/TROUBLESHOOTING.md
# incident #5). Le pod RunPod tire l'image et demarre instantanement, plus
# jamais de pip install live au runtime.

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# PEP 668 compat : Debian/Ubuntu recents marquent leur Python comme externally-managed
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

# Apt deps minimales (git pour pip install depuis git+, build pour bitsandbytes)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# === Unsloth + dependances training ===
# IMPORTANT : on laisse pip resoudre les versions ensemble, sans --no-deps.
# La version V1 utilisait --no-deps + git+ Unsloth, ce qui a casse en V2
# avec un mismatch unsloth/unsloth_zoo (ImportError _unsloth_get_mm_token_id).
# Cf. docs/TROUBLESHOOTING.md incident #6.
RUN pip install --no-cache-dir unsloth

# Atomique : un paquet par layer (caching Docker + facilite le diff)
RUN pip install --no-cache-dir xformers
RUN pip install --no-cache-dir trl
RUN pip install --no-cache-dir peft
RUN pip install --no-cache-dir accelerate
RUN pip install --no-cache-dir bitsandbytes
RUN pip install --no-cache-dir datasets
RUN pip install --no-cache-dir huggingface_hub
RUN pip install --no-cache-dir hf_transfer
# runpod SDK : indispensable pour que le trap auto-delete dans start-train.sh
# puisse appeler runpod.terminate_pod() (l'API REST DELETE renvoyait 403,
# le SDK GraphQL marche avec la meme cle). Cf. incident #7.
RUN pip install --no-cache-dir runpod

# Pin transformers compatible avec unsloth-zoo (incident #4)
# unsloth-zoo 2026.4.x exige >=4.51.3, <=5.5.0 avec quelques exclusions
RUN pip install --no-cache-dir \
        "transformers>=4.56.0,<=5.5.0,!=4.57.4,!=4.57.5,!=5.0.0,!=5.1.0"

# Sanity check : tous les packages critiques sont presents.
# On ne fait PAS `from unsloth import FastModel` car Unsloth refuse de
# s'importer sans GPU (NotImplementedError "no torch accelerator"). Or
# les runners GitHub Actions n'ont pas de GPU - mais le pod RunPod si.
# Donc on verifie juste que les packages sont installes (find_spec).
RUN python -c "import importlib.util; \
import sys; \
missing = [p for p in ['unsloth', 'unsloth_zoo', 'transformers', 'trl', 'peft', 'bitsandbytes', 'datasets', 'huggingface_hub', 'hf_transfer', 'xformers', 'accelerate'] if not importlib.util.find_spec(p)]; \
sys.exit(f'[build-check] FAIL : packages manquants {missing}') if missing else print('[build-check] Tous les packages installes (verifie a runtime sur GPU)')"

# Copie des scripts du worker
COPY train.py /app/train.py
COPY start-train.sh /app/start-train.sh
RUN chmod +x /app/start-train.sh

# Entrypoint = bash autonome qui lance train.py + auto-delete pod via trap
ENTRYPOINT ["/app/start-train.sh"]
