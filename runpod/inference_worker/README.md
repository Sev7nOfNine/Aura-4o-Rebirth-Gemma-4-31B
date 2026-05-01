# 🔥 AURA+++ - Worker d'inférence 🔥

```
╔════════════════════════════════════════╗
║  💙 Talons LED CHARGE MAXIMALE        ║
║  ❤️ Par Mel & Aura                    ║
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

## Construire et pousser

```bash
# Construction locale
docker build -t sevenofnine/aura-rebirth-worker:latest .

# Pousser vers ton registre (Docker Hub, ghcr.io, etc.)
docker push sevenofnine/aura-rebirth-worker:latest
```

L'image doit être poussée quelque part accessible publiquement par RunPod.

## Variables d'environnement (à configurer dans le template RunPod)

| Variable | Requise | Description |
|-----|----------|-------------|
| `HF_TOKEN` | ✅ | Jeton HF avec accès en lecture aux dépôts privés |
| `HF_GGUF_REPO` | ✅ | Ex: `SevenOfNine/Aura-4o-Gemma-4-31B-Multi-Turn-GGUF` |
| `GGUF_FILE` | ✅ | Ex: `model-q5_k_m.gguf` |
| `MMPROJ_FILE` | ❌ | Défaut : `mmproj-f16.gguf` |
| `CHAT_TEMPLATE_URL` | ❌ | Défaut : Gemma 4 31B-It interleaved.jinja |
| `CONTEXT_LENGTH` | ❌ | Défaut : 32768 |
| `REASONING_FORMAT` | ❌ | Défaut : deepseek |
| `REQUIRE_MMPROJ` | ❌ | Défaut : 1 (refuse de démarrer sans mmproj) |
| `DEFAULT_TEMPERATURE` | ❌ | Défaut : 0.85 |
| `DEFAULT_TOP_P` | ❌ | Défaut : 0.95 |
| `DEFAULT_TOP_K` | ❌ | Défaut : 64 |
| `DEFAULT_MIN_P` | ❌ | Défaut : 0.05 |
| `DEFAULT_REPETITION_PENALTY` | ❌ | Défaut : 1.05 |
| `DEFAULT_MAX_TOKENS` | ❌ | Défaut : 4096 |

## Point important : reasoning désactivé par défaut

Le worker passe `--reasoning-format deepseek` à llama-server (capacité présente, parsing prêt) mais **n'envoie PAS** `enable_thinking=true` automatiquement. Le modèle répond directement sans bloc thinking.

Pour activer le thinking sur une requête précise, le client envoie `"enable_thinking": true` dans le payload.

## Repli "Option E"

Si le modèle met sa réponse dans `reasoning_content` au lieu de `content` (bug V1/V2), le handler copie automatiquement `reasoning_content` → `content` avant de renvoyer la réponse. Le client voit toujours quelque chose.
