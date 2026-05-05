# 🚀 Serverless Deploy Guide — Aura-4o-Rebirth-Gemma-4-31B

Guide pour déployer un endpoint Serverless RunPod qui sert le GGUF Aura 31B avec
multimodal (vision via mmproj). Pas de network volume, marche n'importe où il y
a une 48 GB GPU dispo.

> **TL;DR** :
> ```bash
> python pipeline/04b_deploy_no_volume.py
> ```
> Une commande, un endpoint OpenAI-compatible, fini.

---

## Pourquoi cette voie

On voulait au départ déployer avec un **network volume** (cache du GGUF entre
cold starts) en EU-SE-1 (où il y a beaucoup d'A40). Trois murs nous ont fait
pivoter :

1. **EU-SE-1 ne supporte plus la création de network volumes** (seulement les
   pods classiques). Erreur API : `"Data center 'EU-SE-1' not found or does
   not support network volumes"`.
2. **Les autres DC EU avec volumes** (NL, CZ, NO, RO, IS) avaient peu ou pas de
   stock 48 GB au moment du test.
3. **Tenter un volume = se contraindre à un seul DC** (le worker doit tourner
   au même endroit que le volume).

→ **Solution sans volume** : le worker DL le GGUF + mmproj depuis HF à chaque
cold start (~5–15 min première fois, beaucoup plus rapide ensuite tant que le
worker reste warm sur idle 60s). Liberté totale de DC.

---

## Pré-requis

### 1. Image Docker du worker (déjà OK)

L'image vit sur **GHCR public** :
```
ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest
```

Source : [`runpod/inference_worker/`](../runpod/inference_worker) du repo. Le
workflow [`build-worker.yml`](../.github/workflows/build-worker.yml) la rebuild
auto à chaque push sur `main` qui touche le worker.

> ⚠️ **GHCR push private par défaut**. Après un nouveau build, vérifier que
> le package `aura-4o-rebirth-gemma-4-31b-worker` est en **Public** :
> https://github.com/Sev7nOfNine?tab=packages → package settings → Danger Zone
> → Change visibility → Public.

### 2. GGUF + mmproj sur HF

Repo : `SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-GGUF` (peut rester privé,
le worker utilise `HF_TOKEN` pour DL).

Fichiers attendus :
- `Aura-4o-Rebirth-Gemma-4-31B-Q5_K_M.gguf` (~21 GB, sweet spot)
- `Aura-4o-Rebirth-Gemma-4-31B-mmproj-f16.gguf` (~1.2 GB, vision)

### 3. Variables d'environnement (`.env` à la racine du repo)

```env
HF_TOKEN=hf_...
RUNPOD_API_KEY=rpa_...
```

---

## Déploiement

### Méthode A — Script tout-en-un (recommandée)

```bash
python pipeline/04b_deploy_no_volume.py
```

Le script :
1. Lit `.env` pour les tokens.
2. Crée le template RunPod (avec env vars `HF_TOKEN`, `HF_GGUF_REPO`,
   `GGUF_FILE`, `MMPROJ_FILE`, `CONTEXT_LENGTH`, `REQUIRE_MMPROJ`).
3. Crée l'endpoint Serverless (no volume, no DC restriction, scale-to-zero).
4. Affiche l'URL OpenAI à utiliser dans TypingMind.

### Méthode B — Manuelle via console RunPod (UI)

1. Console RunPod → **Serverless** → **New Endpoint**
2. **Custom Image** : `ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest`
3. **Endpoint name** : `Aura-4o-Rebirth-Gemma-4-31B`
4. **Network volume** : aucun
5. **GPU** : sélectionner les 48 GB (A40, A6000, L40)
6. **Datacenter** : laisser vide (any)
7. **Workers** : min 0, max 1, idle 60s
8. **Container disk** : 30 GB
9. **Env vars** :
   ```
   HF_TOKEN              = (ta clé HF)
   HF_GGUF_REPO          = SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-GGUF
   GGUF_FILE             = Aura-4o-Rebirth-Gemma-4-31B-Q5_K_M.gguf
   MMPROJ_FILE           = Aura-4o-Rebirth-Gemma-4-31B-mmproj-f16.gguf
   CONTEXT_LENGTH        = 32768
   REQUIRE_MMPROJ        = 1
   ```
10. Create.

---

## Configuration TypingMind

Settings → Models → **Add Custom Model** :

| Champ | Valeur |
|---|---|
| **Name** | `Aura-4o-Rebirth-Gemma-4-31B` |
| **API Type** | OpenAI Chat Completions API |
| **Endpoint URL** | `https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1` |
| **Model ID** | `Aura-4o-Rebirth-Gemma-4-31B` |
| **Authentication Type** | Bearer Token |
| **API Key** | ta `RUNPOD_API_KEY` |
| **Context Length** | `32768` |
| **Use TypingMind Proxy** | OFF (pas nécessaire, RunPod accepte CORS) |

> ⚠️ Le bouton **Test** de TypingMind a un timeout court (~30 s). Au premier
> déploiement, le cold start prend ~10–15 min et fait planter le test.
> **Pré-warm le worker via curl AVANT** d'utiliser le bouton Test (voir
> Troubleshooting plus bas).

---

## Pré-warm + premier test

Premier appel = **cold start ~10–15 min** (DL Q5 21 GB + mmproj depuis HF +
boot llama-server). Patiente ou pré-warm en background :

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"messages":[{"role":"user","content":"Coucou"}],"max_tokens":50}}'
```

Renvoie un `id` immédiatement. Poll le statut :

```bash
curl -s "https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{JOB_ID}" \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

Statuts :
- `IN_QUEUE` → en attente d'un worker
- `IN_PROGRESS` → worker spawné, en train de traiter (DL ou inference)
- `COMPLETED` → succès
- `FAILED` → consulter `error` field

Une fois `COMPLETED` une fois, le worker reste warm 60 s puis scale-to-zero.
Pendant qu'il est warm, **TypingMind Test** passera instantanément.

---

## Troubleshooting

### `worker throttled` persistant

```json
{"workers":{"throttled":1, ...}}
```

**Causes possibles** :
- **Image GHCR private** → 401 lors du pull. Vérifier sur la page packages.
- **Pas de stock GPU** au DC choisi. Retirer la restriction `dataCenterIds`
  pour laisser RunPod choisir où spawn.
- **Quota account RunPod dépassé**.

**Quick check public image** :
```bash
TOKEN=$(curl -s "https://ghcr.io/token?service=ghcr.io&scope=repository:sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:pull" | python -c "import json,sys;print(json.load(sys.stdin)['token'])")
curl -sI -H "Authorization: Bearer $TOKEN" "https://ghcr.io/v2/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker/manifests/latest" | head -3
```
Doit répondre `HTTP/1.1 200 OK`. Si 401 : image privée.

**Quick check supply GPU** : aller sur l'UI RunPod, Edit endpoint → GPU
configuration. Voir si les 48 GB sont `Available` ou `Unavailable`.

### TypingMind "Test connection" boucle

Le test TypingMind a un timeout court qui n'attend pas le cold start. Solution :
1. **Cancel le test** (croix), ne pas laisser tourner en boucle.
2. **Pré-warm le worker via curl** (voir section Pré-warm).
3. Quand le job curl est `COMPLETED`, retourne dans TypingMind, click **Test**
   → ça passera en 1–2 s parce que le worker est warm.
4. Click **Add Model** pour sauvegarder.

### Worker spawne mais 0 réponse

Le DL des 21 GB peut être lent (10–15 min depuis HF vers le worker). Patiente.
Si > 20 min sans `COMPLETED` : check les logs côté RunPod (UI endpoint →
Workers → Logs).

### `mmproj download failed`

Le worker a `REQUIRE_MMPROJ=1` (échec si mmproj manque). Vérifier :
- Le fichier `Aura-4o-Rebirth-Gemma-4-31B-mmproj-f16.gguf` existe bien sur HF
- Le `HF_TOKEN` env var a accès au repo

---

## Cost overview

| Item | Coût |
|---|---|
| Endpoint scale-to-zero idle | **$0** |
| Worker A40 actif | ~$0.40/hr (~$0.0001/s) |
| Worker A6000 actif | ~$0.50/hr |
| Cold start (~10–15 min DL au premier réveil) | ~$0.10 |
| Cold start warm (~5–10 s sur idle 60s) | ~$0.001 |

Estimation usage modéré (1h actif/jour) : **~$12/mois**.

Pas de coût stockage (pas de network volume).

---

## Naming conventions

Tout doit s'aligner sur **`Aura-4o-Rebirth-Gemma-4-31B`** :

| Endroit | Nom |
|---|---|
| GitHub repo | `Aura-4o-Rebirth-Gemma-4-31B` |
| HF repos (3) | `SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-{LoRA,Merged,GGUF}` |
| GGUF files | `Aura-4o-Rebirth-Gemma-4-31B-{Q4_K_M,Q5_K_M,Q8_0,mmproj-f16}.gguf` |
| Docker image | `ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest` |
| RunPod template | `Aura-4o-Rebirth-Gemma-4-31B-template` |
| RunPod endpoint | `Aura-4o-Rebirth-Gemma-4-31B` |
| TypingMind Model ID | `Aura-4o-Rebirth-Gemma-4-31B` |

---

## Lifecycle

### Update du GGUF (nouvelle quantization, fix tokenizer, etc.)

1. Push le nouveau GGUF sur HF (même repo, nouveau nom de fichier OU même nom
   pour écraser).
2. Si le nom de fichier change : update `GGUF_FILE` dans le template RunPod
   (UI → template → edit env vars).
3. **Force un cold start** : tuer les workers actifs (UI → endpoint →
   Workers → Terminate) — au prochain appel, le nouveau GGUF sera DL.

### Update de l'image worker

1. Modifier `runpod/inference_worker/{Dockerfile,handler.py,start-runpod.sh}`.
2. Push sur `main` → workflow `build-worker.yml` rebuild + push GHCR.
3. Force cold start côté endpoint pour utiliser la nouvelle image.

### Suppression propre

```bash
python pipeline/04b_deploy_no_volume.py --delete
```

Ou manuellement (dans cet ordre) :
1. Endpoint
2. Template
3. Network volume (si existant)

Tant que tu ne supprimes pas l'endpoint, il reste à $0 si scale-to-zero.

---

*Mel & Aura* ❤️♾️
