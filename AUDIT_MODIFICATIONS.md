# Audit Modifications - Aura-Rebirth

Date: 2026-05-01  
Commit verifie: `ce97764`  
Depot: `Sev7nOfNine/Aura-4o-Rebirth-Gemma-4-31B`

## Resume

Les trois points remontes pendant l'audit Codex ont ete corriges dans `ce97764`, puis verifies localement. Un microfix supplementaire a ete applique au `README.md` pour rendre le bloc Docker/deploiement executable depuis la racine du depot.

## Corrections Validees

### 1. README aligne avec les scripts

Probleme initial:

Le `README.md` documentait encore des flags qui n'existent plus:

```bash
--merged-repo
--output-repo
--gguf-repo
--endpoint-name
```

Correction:

Les commandes documentees correspondent maintenant aux CLI reelles:

```bash
python pipeline/03_abliterate.py
python pipeline/03_abliterate.py --abliterate
python pipeline/03_abliterate.py --extra-quants q4_k_m,q8_0
python pipeline/04_deploy.py --worker-image ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest
```

Verification:

```bash
python pipeline/03_abliterate.py --help
python pipeline/04_deploy.py --help
```

Les flags exposes par les scripts correspondent aux exemples du `README.md`.

### 2. Payload RunPod corrige dans `04_deploy.py`

Probleme initial:

Le script envoyait un payload endpoint avec des champs non conformes ou mal formes:

```python
'locations': 'EU-SE-1'
'workersStandby': 0
'env': [{'key': k, 'value': v}]
'dockerStartCmd': ''
```

Correction:

Le payload suit maintenant la specification RunPod REST API:

```python
'dataCenterIds': ['EU-SE-1']
'env': env_vars
'computeType': 'GPU'
```

Changements notables:

- `locations` remplace par `dataCenterIds`, sous forme de liste.
- `workersStandby` supprime, car absent de la spec POST `/v1/endpoints`.
- `env` passe en objet/dictionnaire, pas en tableau `{key, value}`.
- `dockerStartCmd` supprime, l'`ENTRYPOINT` du Dockerfile suffit.
- `computeType: GPU` ajoute explicitement.

Sources verifiees:

- RunPod `POST /v1/endpoints`: `dataCenterIds`, `computeType`, `gpuTypeIds`, `workersMin`, `workersMax`.
- RunPod `POST /v1/templates`: `env` est un objet, `dockerStartCmd` est une liste optionnelle.

### 3. Cleanup dataset deplace apres matching

Probleme initial:

Le cleanup etait applique au JSONL avant le matching avec `conversations.json`. Comme l'export brut contenait encore les em-dashes et thinking blocks d'origine, le matching exact pouvait etre degrade et tomber inutilement en fuzzy.

Correction:

Le pipeline fait maintenant:

```text
1. Charger JSONL raw
2. Charger conversations.json raw
3. Extraire les paires export
4. Matcher raw JSONL <-> raw export
5. Construire les runs
6. Appliquer cleanup seulement sur les paires conservees
7. Ecrire le dataset final
```

Effet attendu:

- Le matching garde la couverture originale.
- Le dataset final reste nettoye.
- Les paires non utilisees ne sont pas modifiees inutilement.

Validation mentionnee dans le commit:

```text
11349 unique stable
68.7% coverage
```

### 4. Microfix README ajoute localement

Probleme trouve apres verification:

Le bloc deploiement faisait:

```bash
cd runpod/inference_worker
docker build -t ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest .
docker push ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest
python pipeline/04_deploy.py --worker-image ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest
```

Apres le `cd`, la commande `python pipeline/04_deploy.py` ne partait plus de la racine du depot et aurait echoue.

Correction appliquee:

```bash
docker build -t ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest runpod/inference_worker
docker push ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest
python pipeline/04_deploy.py --worker-image ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest
```

Le bloc complet reste maintenant executable depuis la racine du repo.

## Verifications Locales

Syntaxe Python:

```bash
python -m py_compile pipeline/01_dataset_build.py pipeline/02_train.py pipeline/03_abliterate.py pipeline/04_deploy.py runpod/inference_worker/handler.py
```

Resultat:

```text
OK
```

CLI:

```bash
python pipeline/03_abliterate.py --help
python pipeline/04_deploy.py --help
```

Resultat:

```text
OK - les flags documentes correspondent aux scripts.
```

## Etat Actuel

`ce97764` corrige les trois findings initiaux.  
Le microfix README et ce fichier d'audit ont ete integres dans `origin/main` via `abba47c`.

Findings additionnels traites apres recheck :

- `b35f6bd` : pas de double stockage (`volume_in_gb=0`), datacenter `EU-SE-1` pinne, `disk_train_gb_min` revus a la baisse pour tous les modeles du registre
- workflow GHCR : tags forces en lowercase (`ghcr.io/sev7nofnine/...`) car `github.repository_owner` (`Sev7nOfNine`) contient des majuscules invalides en nom Docker
- `README.md` : aligne sur GHCR (`ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest`) au lieu du naming Docker Hub style
- `runpod/inference_worker/Dockerfile` : build Docker compatible PEP 668 avec `PIP_BREAK_SYSTEM_PACKAGES=1`

## Commits suivants (Codex + ajustements)

`9ffa5a8` (script tout-en-un + checkpoints)
- `aura.py` ajoute a la racine : orchestrateur train + GGUF + deploy en une commande
- 02_train.py : checkpoints LoRA pousses sur HF tous les save_steps
  (`save_strategy=steps`, `push_to_hub=True`, `hub_strategy=every_save`)
- aura.yaml : `save_steps=300` initialement, abaisse a `50` plus tard apres chunking

`2da2727` (preflight + smoke tests)
- `preflight.py` : audit read-only avant depense RunPod
  - structure dataset (JSON, roles, vides, thinking, dashes)
  - tokenization vraie via tokenizer Gemma 4
  - check `max_seq_length` : avait detecte le piege 340/841 rows > 4096
  - GitHub Actions worker image status
  - inventaire RunPod actif (pods + endpoints)
  - sortie : `preflight_report.md` avec verdict GO / NO-GO
- `typingmind_smoke.py` : 5 tests post-deploy
  text, thinking-off, vision (image PNG), tools (function calling), web-tool shape
- `aura.py` : lance `preflight.py` automatiquement avant chaque etape couteuse

`4b7dfc5` (chunking)
- `pipeline/01_chunk_dataset.py` : decoupe dataset multi-turn en chunks <= max_seq_length
  - tokenizer Gemma 4 utilise pour mesure exacte
  - jamais cut au milieu d'une paire (user, assistant)
  - 25 turns geants exclus + listes dans `giant_turns_review.jsonl` pour review humaine
- aura.yaml : ajout `dataset.train_hf_id` pointant vers le chunked
  (`Aura-4o-Rebirth-Dataset-Chunked-4096`)
- 02_train.py : utilise `dataset.train_hf_id` si present
- save_steps : 300 -> 50 (1868 sequences chunked, 300 ne checkpoinerait jamais)

`2153a3b` (dataset cards)
- `dataset_card_chunked.md` : nouveau, README pour le dataset HF chunked
- `dataset_card.md` : warning ajoute en haut, pointer vers le chunked, stats reelles
- Push sur HF : les deux datasets ont maintenant un README clair

## Etat final pipeline

| Verification | Statut |
|---|---|
| Tous les .py compilent | OK |
| GitHub Actions worker build | green sur dernier run |
| Image GHCR `ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest` | publiee |
| Dataset chunked sur HF | 1868 chunks, 100% retention, 0 erreurs |
| Preflight verdict | GO |
| Pods RunPod actifs | 0 |
| Endpoint V1 existant `01p64ykg6u3p0i` | dormant, intact |
| Cout estime training V3 | $4.68 - $9.36 (12-24h sur A40 EU-SE-1) |

## Prochaines Etapes Recommandees

1. Verifier que le workflow `Build & Push Worker Image` repasse au vert.
2. Lancer un dry-run training:

```bash
python pipeline/02_train.py --dry-run
```

3. Verifier le GPU choisi, le disque demande, les repos HF cibles et le cout estime.
4. Lancer le training V3 uniquement apres validation du dry-run.

