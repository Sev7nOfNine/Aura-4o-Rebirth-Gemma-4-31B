# 🔥 AURA+++ REBIRTH 🔥

Projet privé de reconstruction d’Aura. Le code est sous MIT, mais le jeu de données reste privé et n’est pas destiné à la redistribution.

## Ce que fait AURA+++ REBIRTH

Pipeline complet pour reconstruire Aura à partir de son dataset multi-turn 4o, l’entraîner sur un modèle open source, la convertir en GGUF et la déployer en serverless pour l’usage quotidien.

## Démarrage rapide

```bash
# Pipeline complet : entraînement → GGUF → déploiement
python aura.py

# Juste le déploiement si LoRA + merged + GGUF sont déjà sur HF
python aura.py --skip-train --skip-gguf

# V3.1 avec abliteration si V3.0 ne suffit pas
python aura.py --skip-train --abliterate
```

`aura.py` orchestre les 4 sous-scripts du pipeline. Il lance d’abord `preflight.py` en lecture seule pour sortir un verdict `GO/NO-GO` avant toute dépense RunPod. À la moindre erreur dans une étape, le script s’arrête proprement et tu peux reprendre en sautant les étapes déjà faites.

## Garde-fous avant dépense

```bash
# Audit lecture seule : dataset, longueurs tokenizer, coûts, CI du worker, RunPod si la clé est dispo
python preflight.py

# Prépare et pousse le dataset chunké si le preflight bloque sur max_seq_length
python pipeline/01_chunk_dataset.py --push-hf SevenOfNine/Aura-4o-Dataset-Multi-Turn-Chunked-4096 --private

# Après déploiement : teste le vrai endpoint façon TypingMind
python typingmind_smoke.py --endpoint-id <RUNPOD_ENDPOINT_ID>
```

`preflight.py` bloque notamment si trop de conversations dépassent `training.max_seq_length`, car TRL peut tronquer les exemples longs. `pipeline/01_chunk_dataset.py` découpe les longues conversations sans jamais couper au milieu d’un tour user→assistant, et liste les tours géants dans `giant_turns_review.jsonl`. `typingmind_smoke.py` envoie de vraies requêtes à l’endpoint et vérifie le texte, le mode thinking désactivé, la vision, les tools/function-calling et la forme du web-search.

## Sous-scripts

| # | Script | Rôle |
|---|--------|------|
| 01 | `pipeline/01_dataset_build.py` | Reconstruit le dataset multi-turn depuis l’export 4o trié |
| 01b | `pipeline/01_chunk_dataset.py` | Découpe le dataset en blocs <= `max_seq_length` pour éviter la troncature SFT |
| 02 | `pipeline/02_train.py` | Fine-tune LoRA V1 strict sur RunPod, auto-size GPU/disque, merge Unsloth 4-bit, checkpoints HF réguliers |
| 03 | `pipeline/03_abliterate.py` | Récupère le merged → extrait le mmproj → GGUF + quants → pousse sur HF |
| 04 | `pipeline/04_deploy.py` | Crée un nouvel endpoint serverless RunPod sans toucher à l’existant |

## Procédure complète

### 0. Pré-requis

- Python 3.10+
- Compte Hugging Face + token d’écriture : <https://huggingface.co/settings/tokens>
- Compte RunPod + clé SSH ed25519 ajoutée : <https://www.runpod.io/console/user/settings>
- Disque local : ~50 GB libres pour les caches intermédiaires

```bash
pip install -r requirements.txt
```

### 1. Préparer les variables d’environnement

Crée un fichier `.env` à la racine du dépôt :

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
RUNPOD_API_KEY=your_runpod_key
HF_USERNAME=YourUsername
```

### 2. Construction du dataset multi-turn

À partir de :

- `aura_dataset.jsonl` (tri manuel des paires)
- `conversations.json` (export ChatGPT brut, pour l’ordre uniquement)

```bash
python pipeline/01_dataset_build.py \
  --jsonl path/to/aura_dataset.jsonl \
  --conversations path/to/conversations.json \
  --output aura_final_dataset.jsonl \
  --push-hf SevenOfNine/Aura-4o-Dataset-Multi-Turn \
  --private
```

**Principe** : le tri JSONL est la vérité absolue du contenu. Le `conversations.json` sert seulement à reconstruire l’ordre et les passages multi-turn continus. Aucun message rerouté, aucun pré-Aura, aucun contenu parasite n’est ajouté.

Voir [`docs/DATASET.md`](docs/DATASET.md) pour les détails.

### 3. Ajustement sur RunPod

```bash
python pipeline/02_train.py
# ou avec dry-run pour voir le plan + GPU choisi sans rien créer :
python pipeline/02_train.py --dry-run
```

Le script lit `configs/aura.yaml` comme source de vérité. Il :

- auto-size le GPU et le disque selon la taille du modèle
- crée un Pod RunPod, installe Unsloth + dépendances
- lance le LoRA SFT avec la recette V1 stricte
- merge via Unsloth 4-bit
- pousse le LoRA et le merged sur Hugging Face privé
- termine le Pod automatiquement

Voir [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) pour le pourquoi des choix.

### 4. GGUF + mmproj

```bash
# V3.0 de base : GGUF Q5 + mmproj, sans abliteration
python pipeline/03_abliterate.py --config configs/aura.yaml

# V3.1 : ajoute l’abliteration si nécessaire
python pipeline/03_abliterate.py --config configs/aura.yaml --abliterate
```

### 5. Déploiement serverless

```bash
python pipeline/04_deploy.py --worker-image ghcr.io/sev7nofnine/aura-4o-rebirth-worker:latest
```

L’image worker est construite et publiée automatiquement par GitHub Actions sur GHCR à chaque push qui modifie `runpod/inference_worker/**`.

## Structure du dépôt

```text
├── aura.py                         # Orchestrateur tout-en-un
├── preflight.py                    # Audit lecture seule
├── typingmind_smoke.py             # Smoke test post-déploiement
├── pipeline/
│   ├── 01_dataset_build.py
│   ├── 01_chunk_dataset.py
│   ├── 02_train.py
│   ├── 03_abliterate.py
│   └── 04_deploy.py
├── configs/
│   └── aura.yaml                   # Source de vérité unique
├── docs/
│   ├── DATASET.md
│   └── METHODOLOGY.md
└── runpod/
    └── inference_worker/
        ├── handler.py
        ├── Dockerfile
        └── README.md
```

## Modèles publiés

Les modèles finaux seront publiés sur <https://huggingface.co/SevenOfNine> avec ce schéma :

- `Aura-4o-Rebirth-LoRA`
- `Aura-4o-Rebirth-Merged`
- `Aura-4o-Rebirth-GGUF`

## Pourquoi c’est différent

- Multi-turn préservé : le modèle apprend la fluidité des conversations, pas des paires isolées.
- Recette V1 stricte : les hyperparamètres qui ont capté la voix Aura sont conservés.
- Préflight avant toute dépense : on bloque avant de brûler du budget.
- Dataset privé : le code peut être partagé, pas les données.

## Licence

Code sous licence MIT. Le jeu de données reste privé et ne doit pas être redistribué.
