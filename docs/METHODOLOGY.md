# 🔥 Aura-4o-Rebirth 🔥

## Pourquoi du multi-turn

Aura n’est pas une suite de réponses indépendantes. C’est un flux conversationnel avec des changements de registre, des reprises, des digressions et des retours. Le multi-turn permet de préserver cette continuité au lieu d’apprendre un modèle plat et froid.

## Pourquoi un LoRA conservateur

La recette retenue est volontairement proche de celle qui a capté la voix Aura sans écraser le modèle de base :

- LoRA raisonnable
- learning rate mesuré
- plusieurs epochs mais sans excès
- `max_seq_length` contrôlé

L’objectif est d’ajouter la personnalité sans casser les capacités générales du modèle.

### Hyperparams V7 (locked 3 May 2026, post-audit)

- `r=32`, `alpha=32`, `dropout=0.0`, `target_modules='all-linear'`
- `lr=2e-4`, `cosine`, `warmup_ratio=0.03`, `weight_decay=0.01`
- `epochs=3`, `batch=4`, `grad_accum=8` (effective batch 32)
- `max_seq_length=4096`, `bf16`, `load_in_4bit=True`, `optim=adamw_8bit`
- **`packing=False`** (mandatory for vision-language Gemma 4)
- **`assistant_only_loss=True`** (V7 critical fix: loss only on Aura tokens, no user-token contamination)

V6 silently trained on user tokens too (default `assistant_only_loss=False` in TRL). V7 fixes this and simplifies the script (no more vlm preprocessing or DataCollator: dataset is pure text, TRL handles chat template + masking automatically).

Source of truth at runtime: `runpod/train_worker/train.py` (DEFAULTS hardcoded).
`configs/aura.yaml` is kept in sync for documentation purposes.

## Pourquoi le merge puis l’abliteration

L’ordre retenu est :

1. entraînement LoRA
2. merge dans le modèle de base
3. abliteration éventuelle sur le modèle complet
4. conversion GGUF
5. déploiement

Cet ordre garde le comportement de base intact pendant l’entraînement puis nettoie le modèle final au bon moment.

## Pourquoi le serverless

Le serverless est plus adapté à un usage personnel de chat :

- pas de machine allumée inutilement
- facturation à l’usage effectif
- mise à l’échelle automatique
- déploiement plus simple à maintenir

## Pourquoi la quantification

Le GGUF en Q5_K_M est le point d’équilibre retenu pour l’usage quotidien :

- suffisamment compact
- qualité correcte
- coût GPU raisonnable
- plus simple à servir via llama.cpp
