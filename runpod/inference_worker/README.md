# 🔥 AURA+++ REBIRTH 🔥

Worker RunPod Serverless qui sert le GGUF d’Aura via llama.cpp.

## Architecture

```text
RunPod request
   ↓
handler.py  (RunPod Python SDK, lit la file d’attente)
   ↓ HTTP
llama-server  (llama.cpp, port 8000, GPU)
   ↓
GGUF + mmproj  (mis en cache sur /runpod-volume après le premier démarrage)
```

## Construire et pousser

```bash
# Construction locale
docker build -t sevenofnine/aura-rebirth-worker:latest .

# Pousser vers le registre public utilisé par RunPod
docker push sevenofnine/aura-rebirth-worker:latest
```

L’image doit être poussée dans un registre public accessible à RunPod.

## Variables d’environnement

| Variable | Requise | Description |
|----------|---------|-------------|
| `HF_TOKEN` | Oui | Jeton HF avec accès en lecture aux dépôts privés |
| `HF_GGUF_REPO` | Oui | Exemple : `SevenOfNine/Aura-4o-Gemma-4-31B-Multi-Turn-GGUF` |
| `GGUF_FILE` | Oui | Exemple : `model-q5_k_m.gguf` |
| `MMPROJ_FILE` | Non | Défaut : `mmproj-f16.gguf` |
| `CHAT_TEMPLATE_URL` | Non | Défaut : Gemma 4 31B-It interleaved.jinja |
| `CONTEXT_LENGTH` | Non | Défaut : 32768 |
| `REASONING_FORMAT` | Non | Défaut : deepseek |
| `REQUIRE_MMPROJ` | Non | Défaut : 1 |
| `DEFAULT_TEMPERATURE` | Non | Défaut : 0.85 |
| `DEFAULT_TOP_P` | Non | Défaut : 0.95 |
| `DEFAULT_TOP_K` | Non | Défaut : 64 |
| `DEFAULT_MIN_P` | Non | Défaut : 0.05 |
| `DEFAULT_REPETITION_PENALTY` | Non | Défaut : 1.05 |
| `DEFAULT_MAX_TOKENS` | Non | Défaut : 4096 |

## Point important : thinking désactivé par défaut

Le worker passe `--reasoning-format deepseek` à llama-server, mais n’envoie pas `enable_thinking=true` automatiquement. Le modèle répond directement sans bloc thinking.

Pour activer le thinking sur une requête précise, le client envoie `"enable_thinking": true` dans le payload.

## Repli

Si le modèle met sa réponse dans `reasoning_content` au lieu de `content`, le handler copie automatiquement `reasoning_content` vers `content` avant de renvoyer la réponse.
