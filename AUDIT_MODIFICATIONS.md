# Audit Modifications - Aura-Rebirth

Date: 2026-05-01  
Commit verifie: `ce97764`  
Depot: `Sev7nOfNine/Aura-Rebirth`

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
python pipeline/04_deploy.py --worker-image sev7nofnine/aura-rebirth-worker:latest
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
docker build -t sev7nofnine/aura-rebirth-worker:latest .
docker push sev7nofnine/aura-rebirth-worker:latest
python pipeline/04_deploy.py --worker-image sev7nofnine/aura-rebirth-worker:latest
```

Apres le `cd`, la commande `python pipeline/04_deploy.py` ne partait plus de la racine du depot et aurait echoue.

Correction appliquee:

```bash
docker build -t sev7nofnine/aura-rebirth-worker:latest runpod/inference_worker
docker push sev7nofnine/aura-rebirth-worker:latest
python pipeline/04_deploy.py --worker-image sev7nofnine/aura-rebirth-worker:latest
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
Le microfix README est present localement et doit encore etre commit/push si on veut le garder dans l'historique distant.

Fichier modifie localement apres `ce97764`:

```text
README.md
```

Nouveau fichier de suivi:

```text
AUDIT_MODIFICATIONS.md
```

## Prochaines Etapes Recommandees

1. Commit/push le microfix README et ce fichier d'audit.
2. Lancer un dry-run training:

```bash
python pipeline/02_train.py --dry-run
```

3. Verifier le GPU choisi, le disque demande, les repos HF cibles et le cout estime.
4. Lancer le training V3 uniquement apres validation du dry-run.

