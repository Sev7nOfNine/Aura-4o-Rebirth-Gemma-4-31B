# 🔥 Aura-4o-Rebirth - Train Worker 🔥

Image Docker pré-bakée pour le training LoRA. Sidesteppe le pip install
runtime qui causait des OOM cgroup silencieux (cf. `docs/TROUBLESHOOTING.md`
incidents #4 et #5).

## Architecture

```text
RunPod pod (image GHCR pre-bakee)
   ↓
start-train.sh   (entrypoint, trap auto-delete)
   ↓
train.py         (Unsloth FastModel + SFTTrainer + push HF)
   ↓
HF Hub :
  - LoRA checkpoints toutes les 50 steps    → SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-LoRA
  - Merged 16-bit (Unsloth method) en fin   → SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-Merged
   ↓
trap EXIT → DELETE /v1/pods/{pod_id}        (auto-cleanup, pas de pod fantome)
```

## Build

Build automatique via [`.github/workflows/build-train-worker.yml`](../../.github/workflows/build-train-worker.yml)
sur push de `runpod/train_worker/**`. Image poussée vers :

```
ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-train-worker:latest
ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-train-worker:<sha>
```

Build local (rare, pour tester) :

```bash
docker build -t aura-train-worker:dev runpod/train_worker/
```

## Variables d'environnement

| Variable | Requise | Description |
|----------|---------|-------------|
| `HF_TOKEN` | Oui | Jeton HF avec scope write (push checkpoints + merged) |
| `RUNPOD_API_KEY` | Oui | Clé RunPod (pour auto-delete du pod via trap EXIT) |
| `RUNPOD_POD_ID` | Oui | ID du pod actuel (injecté par RunPod ou par `02_train.py`) |
| `AURA_BASE_MODEL` | Non | Défaut : `SevenOfNine/Gemma-4-31B-It-Official` |
| `AURA_DATASET` | Non | Défaut : `SevenOfNine/Aura-4o-Rebirth-Dataset` |
| `AURA_LORA_REPO` | Non | Défaut : `SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-LoRA` |
| `AURA_MERGED_REPO` | Non | Défaut : `SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-Merged` |

## Suivi pendant le training

Depuis le terminal web RunPod :

```bash
tail -f /workspace/aura_run.log     # Log complet (boot + training + push)
tail -f /workspace/train.log        # Training only (loss, steps)
cat /workspace/aura_status.txt      # RUNNING | SUCCESS | FAILED:<code>
```

## Hyperparamètres figés (V1 strict)

| | |
|---|---|
| LoRA r / alpha / dropout | 32 / 32 / 0.0 |
| Target modules | q/k/v/o + up/down/gate proj |
| Learning rate | 2e-4, cosine, warmup 0.03 |
| Epochs | 3 |
| Batch effectif | 1 × 32 (gradient accumulation) |
| max_seq_length | 4096 |
| Optimizer | adamw_8bit |
| Merge method | unsloth `merged_16bit` (la méthode V1 qui préserve la voix) |
| Vision layers | NON entraînées (préservées intactes) |
| Push HF | toutes les 50 steps |

Pour changer : éditer `train.py` puis push GH → CI rebuild auto l'image.
