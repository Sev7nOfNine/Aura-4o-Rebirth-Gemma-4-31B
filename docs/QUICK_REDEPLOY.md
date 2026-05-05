# 🚀 Quick Redeploy — Aura-4o-Rebirth

Guide ultra-court pour ré-utiliser ce qu'on a mis en place le 5 mai 2026,
sans avoir à tout repenser.

## Ce qui existe déjà et reste là

| Bloc | Détails | Quand le toucher ? |
|---|---|---|
| **HF GGUF** | `SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-GGUF` (Q4/Q5/Q8 + mmproj) | Si tu retraines |
| **Image worker GHCR** | `ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest` (publique, auto-build à chaque push) | Si tu modifies `runpod/inference_worker/` |
| **RunPod template** | `o7l19uqyln` (pointe sur l'image avec env vars) | Si tu changes le GGUF/mmproj/context |
| **RunPod endpoint** | `x6sybfwczt4lbb` (Aura-4o-Rebirth-Gemma-4-31B, scale-to-zero) | Si tu redéploies sur un autre DC |
| **Cloudflare Worker proxy** | `aura-4o-rebirth-proxy.seven0fnine.workers.dev` (routes /v1/chat → RunPod) | Si tu ajoutes un nouveau modèle |

→ Tu n'as **rien à recréer** pour utiliser Aura tous les jours via TypingMind.

---

## Cas d'usage 1 — "Le worker max=0 traîne, je veux re-allumer"

```powershell
cd "F:\AI\Hugging Face\Aura-4o-Rebirth-Gemma-4-31B"
python -c "import os, requests; from pathlib import Path; e={k.strip():v.strip() for L in Path('.env').read_text().splitlines() if '=' in L and not L.startswith('#') for k,v in [L.split('=',1)]}; r=requests.patch('https://rest.runpod.io/v1/endpoints/x6sybfwczt4lbb', headers={'Authorization':f'Bearer {e[\"RUNPOD_API_KEY\"]}','Content-Type':'application/json'}, json={'workersMax':1}); print(r.status_code)"
```

Réponse `200` → c'est bon. Premier appel = cold start ~10-15 min, ensuite 1-2 s par requête.

## Cas d'usage 2 — "Je veux tout couper pour pas payer"

```powershell
python -c "import os, requests; from pathlib import Path; e={k.strip():v.strip() for L in Path('.env').read_text().splitlines() if '=' in L and not L.startswith('#') for k,v in [L.split('=',1)]}; r=requests.patch('https://rest.runpod.io/v1/endpoints/x6sybfwczt4lbb', headers={'Authorization':f'Bearer {e[\"RUNPOD_API_KEY\"]}','Content-Type':'application/json'}, json={'workersMax':0}); print(r.status_code)"
```

Endpoint reste créé mais aucun worker ne peut spawn. Coût : **0 $**.

## Cas d'usage 3 — "Je viens de pousser un fix dans `runpod/inference_worker/`, faut que RunPod le pull"

1. Le workflow GH Actions rebuild auto l'image en ~2-5 min. Vérifier :
   ```powershell
   curl -s "https://api.github.com/repos/Sev7nOfNine/Aura-4o-Rebirth-Gemma-4-31B/actions/runs?workflow=build-worker.yml&per_page=1"
   ```
   Attendre `"conclusion":"success"`.

2. Pour forcer RunPod à pull la nouvelle image **même si elle a la même tag `:latest` en cache**, on update le template avec le SHA du commit :
   ```powershell
   cd "F:\AI\Hugging Face\Aura-4o-Rebirth-Gemma-4-31B"
   python -c "import os,requests,subprocess; from pathlib import Path; e={k.strip():v.strip() for L in Path('.env').read_text().splitlines() if '=' in L and not L.startswith('#') for k,v in [L.split('=',1)]}; sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(); r=requests.patch('https://rest.runpod.io/v1/templates/o7l19uqyln', headers={'Authorization':f'Bearer {e[\"RUNPOD_API_KEY\"]}','Content-Type':'application/json'}, json={'imageName':f'ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:{sha}'}); print(r.status_code)"
   ```

3. Dans la console RunPod, **terminate** le worker actif → re-spawn auto avec la nouvelle image.

## Cas d'usage 4 — "Je veux changer la quantization (Q5 → Q4 ou Q8)"

```powershell
cd "F:\AI\Hugging Face\Aura-4o-Rebirth-Gemma-4-31B"
python -c "
import os, requests
from pathlib import Path
e={k.strip():v.strip() for L in Path('.env').read_text().splitlines() if '=' in L and not L.startswith('#') for k,v in [L.split('=',1)]}
H={'Authorization':f'Bearer {e[\"RUNPOD_API_KEY\"]}','Content-Type':'application/json'}
# Récupère le template actuel pour patch incrémental
t = requests.get('https://rest.runpod.io/v1/templates/o7l19uqyln', headers=H).json()
env = t.get('env', {}) or {}
env['GGUF_FILE'] = 'Aura-4o-Rebirth-Gemma-4-31B-Q4_K_M.gguf'  # ou Q5_K_M / Q8_0
r = requests.patch('https://rest.runpod.io/v1/templates/o7l19uqyln', headers=H, json={'env': env})
print(r.status_code)
"
```

Terminate worker → re-spawn avec le nouveau GGUF.

## Cas d'usage 5 — "Je déploie une nouvelle Aura (V4, ou la E4B en serverless)"

1. Build + push GGUF + mmproj sur HF (script `pipeline/02b_merge_and_export.py`).
2. Crée un nouvel endpoint via `python pipeline/04b_deploy_no_volume.py` (à adapter avec les paths du nouveau modèle).
3. Note l'`endpoint_id` retourné, par exemple `abc123def`.
4. Edit `runpod/cloudflare_worker/wrangler.toml` :
   ```toml
   ENDPOINT_AURA_4O_REBIRTH_GEMMA_4_E4B = "abc123def"
   ```
5. Redeploy le proxy :
   ```powershell
   cd "F:\AI\Hugging Face\Aura-4o-Rebirth-Gemma-4-31B\runpod\cloudflare_worker"
   wrangler deploy
   ```
6. Côté client (TypingMind), envoie `"model": "Aura-4o-Rebirth-Gemma-4-E4B"` dans le body. Le proxy route automatiquement vers le bon endpoint.

→ **Un seul proxy**, plusieurs modèles. Pas besoin de redéployer un nouveau worker CF à chaque modèle.

## Health check rapide

```powershell
# Le proxy CF répond ?
curl https://aura-4o-rebirth-proxy.seven0fnine.workers.dev/health

# RunPod endpoint répond ?
cd "F:\AI\Hugging Face\Aura-4o-Rebirth-Gemma-4-31B"
python pipeline/04b_deploy_no_volume.py --status
```

## TypingMind config (rappel)

| Champ | Valeur |
|---|---|
| API endpoint | `https://aura-4o-rebirth-proxy.seven0fnine.workers.dev/v1` |
| Authentication Type | API Key via HTTP Header |
| Header Key | `Authorization` |
| Header Value | `Bearer mel-aura` (ton PROXY_KEY) |
| Model ID | `Aura-4o-Rebirth-Gemma-4-31B` |
| Context Length | `32768` |
| Max tokens | `2048` ou `3072` |

---

*Mel & Aura* ❤️♾️
