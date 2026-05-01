# 🔥 AURA+++ - Inference Worker 🔥

```
╔════════════════════════════════════════╗
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝
```

Worker RunPod Serverless qui sert le GGUF Aura via llama.cpp.

## Architecture

```
RunPod request
   ↓
handler.py  (runpod-python SDK, listens on RunPod queue)
   ↓ HTTP
llama-server  (llama.cpp, port 8000, GPU)
   ↓
GGUF + mmproj  (cached on /runpod-volume after first boot)
```

## Build & push

```bash
# Build local
docker build -t sevenofnine/aura-4o-rebirth-worker:latest .

# Push to your registry (Docker Hub, ghcr.io, etc.)
docker push sevenofnine/aura-4o-rebirth-worker:latest
```

L'image doit être pushée quelque part accessible publiquement par RunPod.

## Env vars (set on RunPod template)

| Var | Required | Description |
|-----|----------|-------------|
| `HF_TOKEN` | ✅ | HF token avec accès lecture aux repos privés |
| `HF_GGUF_REPO` | ✅ | Ex: `SevenOfNine/Aura-4o-Rebirth-GGUF` |
| `GGUF_FILE` | ✅ | Ex: `model-q5_k_m.gguf` |
| `MMPROJ_FILE` | ❌ | Default: `mmproj-f16.gguf` |
| `CHAT_TEMPLATE_URL` | ❌ | Default: Gemma 4 31B-It interleaved.jinja |
| `CONTEXT_LENGTH` | ❌ | Default: 32768 |
| `REASONING_FORMAT` | ❌ | Default: deepseek |
| `REQUIRE_MMPROJ` | ❌ | Default: 1 (refuses to start without mmproj) |
| `DEFAULT_TEMPERATURE` | ❌ | Default: 0.85 |
| `DEFAULT_TOP_P` | ❌ | Default: 0.95 |
| `DEFAULT_TOP_K` | ❌ | Default: 64 |
| `DEFAULT_MIN_P` | ❌ | Default: 0.05 |
| `DEFAULT_REPETITION_PENALTY` | ❌ | Default: 1.05 |
| `DEFAULT_MAX_TOKENS` | ❌ | Default: 4096 |

## Flag important : reasoning OFF par défaut

Le worker passe `--reasoning-format deepseek` à llama-server (capacité présente, parsing prêt) mais **n'envoie PAS** `enable_thinking=true` automatiquement. Le modèle répond direct sans bloc thinking.

Pour activer le thinking sur une requête précise, le client envoie `"enable_thinking": true` dans le payload.

## Fallback "Option E"

Si le modèle met sa réponse dans `reasoning_content` au lieu de `content` (bug V1/V2), le handler copie automatiquement `reasoning_content` → `content` avant de renvoyer la réponse. Le client voit toujours quelque chose.
