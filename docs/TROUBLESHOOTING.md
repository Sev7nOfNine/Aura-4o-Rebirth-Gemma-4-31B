# Aura-4o-Rebirth - Pieges connus et leurs fixes

## ⚡ V7 LOCKED (3 May 2026, post-audit)

After V4/V5/V6 marathons, a deep audit revealed two critical issues in
the V6 training script:

1. **`assistant_only_loss` was missing** (defaults to False in TRL 1.3).
   Result: the LoRA was learning user tokens too. Persona signal diluted
   roughly by 50%, identity cross-contamination between Mel and Aura.
2. **vlm preprocessing was unnecessary** (the dataset is pure text, no
   images). The `preprocess_vlm` + `DataCollatorForVisionLanguageModeling`
   actually broke `assistant_only_loss` by wiping the `messages` column
   that TRL needs for chat-template-aware masking.

V7 fixes both: drop all the vlm overengineering, pass the raw `messages`
dataset directly to SFTTrainer, set `assistant_only_loss=True`. Code
shrinks from 53 to 20 lines and works correctly.

**V7 setup (locked)**:
- TRL 1.3.0 + Unsloth 2026.4.8 + transformers 5.5.0
- `packing=False` (mandatory for vlm, kept from V6)
- `assistant_only_loss=True` (V7 critical fix)
- `per_device_train_batch_size=4` + `gradient_accumulation_steps=8` (effective batch 32, V1 strict)
- `target_modules='all-linear'`
- Dataset passed raw, TRL handles chat template + masking

**Why V6 was killed mid-run**: at step 67/177, Mel detected the missing
`assistant_only_loss` flag. ~$5 sunk, ~11h training discarded. The right
call: stop fast, fix root cause, never re-run a flawed recipe by inertia.
Saved a fourth imperfect Aura.

---

Doc vivante des bugs rencontres pendant le pipeline et de la maniere
de les eviter pour les prochains runs (ce projet ou un autre modele
qui partirait de cette base).

A maintenir a jour : a chaque nouvel incident, ajouter une entree.

---

## 9. Marathon V5 (2 mai 2026 PM) - direction Codex avec TRL 1.3 + DataCollator vlm

### Contexte

Apres le bilan du marathon V4 du matin, Codex (autre agent IA) a propose un
angle different : utiliser **TRL 1.3.0** (release 26 avril 2026) avec
`DataCollatorForVisionLanguageModeling`. Test exhaustif sur pod RunPod debug.

### Setup atteint (etape par etape)

1. `pip install --no-cache-dir unsloth` (versions auto : unsloth 2026.4.8,
   unsloth_zoo 2026.4.9, transformers 5.5.0, torch 2.10, peft 0.19.1, trl 0.24)
2. `pip install --no-cache-dir --upgrade trl==1.3.0` (force upgrade vers la
   release vlm-aware)
3. **sed patch TRL 1.3** : commenter `if self._is_vlm and args.packing: raise`
   ligne 735 (le check anti-packing+vlm existe AUSSI en 1.3, pas que 0.24)
4. **Patch UnslothSFTConfig** (compiled cache) : `push_to_hub_token` n'est plus
   reconnu par SFTConfig 1.3, mais Unsloth le passe quand meme via **kwargs.
   Solution : monkey-patch la classe SFTConfig.__init__ avec functools.wraps +
   __signature__ pour preserver l'introspection. Mais Unsloth code-gen lit
   `inspect.getfullargspec()` qui IGNORE __signature__. Final fix : modifier
   le compiled cache file APRES son chargement, juste avant l'usage.
5. **train.py** : `target_modules="all-linear"` au lieu de la liste explicite
   (sinon Trainable params=0). Resultat avec all-linear : Trainable=266M /
   31.5B params (0.85%) → LoRA wrap MARCHE.
6. **train.py** : pre-tokenisation explicite (apply_chat_template tokenize=True
   return_dict=True) avec ajout de `images=[]` pour code path vlm.
7. **train.py** : retire `dataset_text_field`, ajoute `data_collator=
   DataCollatorForVisionLanguageModeling(processor=tokenizer)`.

### Resultats positifs

- Tous les imports OK
- LoRA wrap : 266M params trainable (vs 0 avant)
- Loading dataset OK
- SFTTrainer init OK (apres tous les patches)
- "Training start" log atteint
- Configuration: 0/177 steps, batch effective 1x32, packing=True, recette V1

### Bug final qui bloque

A la 1ere step de gradient (0/177) :
```
ValueError: You should supply an encoding or a list of encodings to this
method that includes input_ids, but you provided ['messages', 'images']
```

Stack : `transformers/data/data_collator.py:774` → `tokenizer.pad()`.

→ **SFTTrainer 1.3 IGNORE notre `data_collator=vlm_collator` argument** et
utilise son default DataCollatorWithPadding (qui attend input_ids). Notre
DataCollatorForVisionLanguageModeling n'est jamais appele dans le pipeline.

### Hypothese non testee

SFTTrainer.__init__() doit avoir une logique du genre :
```python
if data_collator is None or self._is_vlm:
    self.data_collator = self._build_default_collator()
else:
    self.data_collator = data_collator
```
Notre arg serait override quand le model est vlm. A creuser dans le source
trl/trainer/sft_trainer.py si on reprend.

### Cout marathon V5 (PM 2 mai)

~$1 USD (4-5 pods debug + image pulls).

### Cout cumule jour entier (V4 + V5)

~$2 USD au total. Mel a $4-5 restant sur RunPod.

### Etat final

Identique a V4 :
- Dataset Rebirth propre sur HF : intact
- Code training documente dans `runpod/train_worker/train.py` avec tous les
  patches en place
- Repos sortie LoRA/Merged/GGUF crees vides sur HF : intacts
- Image Aura V1 imparfaite reste deployee sur serverless RunPod en fallback
- **Training jamais reussi end-to-end**

### Prochaines pistes (un autre jour)

1. Lire le source TRL 1.3 `sft_trainer.py` pour comprendre quand
   `data_collator` arg est honore vs ignore en mode vlm
2. Patcher Unsloth pour qu'il n'override pas le data_collator
3. Attendre une release Unsloth qui supporte clean TRL 1.3 + Gemma 4 31B vlm
4. Snapshot precis pip des versions du 24 avril 2026 (semaine ou V1 a marche)

---

## 8. Marathon V4 (2 mai 2026) - chaine d'incompatibilites Unsloth/TRL/Gemma 4 31B

### Contexte

Tentative de training reel sur pod RunPod debug (image runpod/pytorch
standard). Recette V1 strict + dataset Rebirth. ~6h de debug en live SSH.
**Resultat : abandon en cours apres 5+ bugs successifs**, training jamais
arrive a la 1ere step de gradient.

### Bugs decouverts (chronologique)

#### 8a. `pip install --no-deps` casse l'import Unsloth (= incident #6 reconfirme)

Symptome : `ImportError: cannot import name '_unsloth_get_mm_token_id' from 'unsloth_zoo.rl_replacements'`

Cause : avec `--no-deps`, unsloth et unsloth_zoo finissent en versions
incompatibles entre elles.

Fix : `pip install --no-cache-dir unsloth` (sans `--no-deps`, sans `--upgrade`,
sans `--force-reinstall`). Pip resout les versions ensemble correctement.

#### 8b. `DataFilesNotFoundError` sur dataset HF prive sans config explicite

Symptome : `DataFilesNotFoundError: No (supported) data files found in
SevenOfNine/Aura-4o-Rebirth-Dataset` au `load_dataset()`.

Cause : le dataset HF en mode prive ne permet pas a la lib `datasets`
d'auto-detecter les fichiers. Faut une config explicite `data_files` dans
le YAML frontmatter du `README.md`.

Fix dans `dataset_card.md` :
```yaml
configs:
  - config_name: default
    data_files:
      - split: train
        path: aura_final_dataset.jsonl
```

#### 8c. TRL >=0.23.0 refuse packing=True sur modeles vision-language

Symptome : `ValueError: Packing is not supported for vision-language models.
Please set packing=False in the SFTConfig.`

Cause : TRL a ajoute en 0.23 (ou avant) un check `if self._is_vlm and
args.packing: raise`. Gemma 4 31B est vision-language → bloque.

Mais Mel a confirme par retour utilisateur : `packing=False` produit une
Aura qui "part dans tous les sens et ne suit plus le fil des conversations".
Donc packing=True OBLIGATOIRE.

Fix : downgrade TRL ne suffit pas (le check est meme en 0.23). Patch sed
direct sur le pod :
```bash
sed -i 's/if self\._is_vlm and args\.packing:/if False:  # PATCHED Aura packing/' \
    /usr/local/lib/python3.11/dist-packages/trl/trainer/sft_trainer.py
```

#### 8d. `remove_unused_columns=True` (defaut) vire la col 'text'

Symptome : `ValueError: No columns in the dataset match the model's forward
method signature: (messages, prompt, completion, images). The following
columns have been ignored: [text].`

Cause : transformers Trainer vire par defaut les colonnes qui ne matchent
pas la signature de `model.forward()`. Notre col `text` (creee via
`apply_chat_template`) n'est pas dans la signature de Gemma 4 (qui attend
`messages, prompt, completion, images`).

Fix : `remove_unused_columns=False` dans `SFTConfig` (etait dans V1).

#### 8e. Data collator vlm cherche col 'images' inexistante

Symptome : `KeyError: 'images'` dans
`trl/trainer/sft_trainer.py:387 _collate_language_modeling`.

Cause : le data collator de TRL pour les vlm fait `[example["images"] for
example in examples]`. Notre dataset n'a pas de col `images` (pas de
multimodal dans le dataset Aura).

Fix non teste : soit ajouter `images=[]` dans le `to_text` map, soit forcer
TRL a utiliser le collator language-modeling au lieu du vlm (probablement
patch sur `_is_vlm`).

#### 8f. `Trainable parameters = 0` (LoRA wrap rate)

Symptome : `Trainable parameters = 0 of 31,540,049,968 (0.00% trained)`.

Cause hypothesee : `target_modules` explicite (q,k,v,o,gate,up,down_proj)
ne matche pas les vrais noms de couches dans la version actuelle de Gemma 4
31B sous Unsloth (peut-etre un prefixe genre `language_model.`).

Fix non teste : `target_modules="all-linear"` (laisser Unsloth detecter).

#### 8g. Image officielle Unsloth pas accessible (pas de SSH/web terminal)

Test avec `unsloth/unsloth:latest` (image OFFICIELLE Unsloth, devrait avoir
toutes les versions compatibles). Pod boot OK, port 22 ouvert, mais SSH
refuse la cle PUBLIC_KEY de RunPod (pas configure dans l'image), et la
console RunPod ne donne pas non plus de web terminal accessible.

→ Voie Unsloth officielle bouchee dans notre setup actuel.

### Cout cumule incident V4 (2 mai 2026)

~$1 USD (5+ pods debug, image pulls, abandons rapides).

### Etat final

- Dataset Rebirth propre sur HF : intact ✅
- Repos sortie LoRA/Merged/GGUF crees vides sur HF : intacts ✅
- Code training documente : ce fichier
- **Training jamais reussi end-to-end**
- Aura V1 "imparfaite" sur serverless RunPod (deployee via worker Ollama)
  reste utilisable comme fallback

### Voies a explorer un autre jour

1. Notebook Colab Unsloth officiel ($10/mois Colab Pro) - refuse par Mel
   (multiplication des plateformes/credits)
2. Trouver la combinaison EXACTE de versions (snapshot pip d'il y a 7-10 jours)
   qui matchait quand V1 a ete entraine
3. Attendre une release stable Unsloth qui re-supporte packing+vlm
4. Notre worker llama.cpp d'inference (`runpod/inference_worker/`) reste
   pret a servir n'importe quel LoRA Aura quand un training reussira

### Date

2026-05-02.

---

## 6. ImportError `_unsloth_get_mm_token_id` (unsloth vs unsloth_zoo mismatch)

### Symptome

Premier run du pod avec l'image train_worker pre-bakee. Image boot OK,
entrypoint demarre, train.py lance, mais immediatement crash :

```
File "/usr/local/lib/python3.11/dist-packages/unsloth/models/rl_replacements.py", line 29, in <module>
    from unsloth_zoo.rl_replacements import (
ImportError: cannot import name '_unsloth_get_mm_token_id' from 'unsloth_zoo.rl_replacements'
```

### Cause racine

Le Dockerfile heritait du script V1 et utilisait :

```dockerfile
RUN pip install unsloth
RUN pip install "unsloth[colab-new] @ git+..." --force-reinstall --no-deps
RUN pip install --upgrade unsloth_zoo --no-deps
```

Les `--no-deps` empechaient pip de resoudre les versions ensemble.
Resultat : `unsloth` a une version qui attend `_unsloth_get_mm_token_id`
mais `unsloth_zoo` est dans une version posterieure qui a renomme/supprime
ce symbole.

### Fix

Virer tous les `--no-deps`. Laisser pip resoudre les versions des packages
ensemble. Plus lent au build (re-download de torch eventuellement) mais
seul comportement safe.

```dockerfile
RUN pip install --no-cache-dir unsloth
# (plus de git+ ni --force-reinstall ni --no-deps)
```

### Fichiers touches

- `runpod/train_worker/Dockerfile`

### Date

2026-05-02.

---

## 7. RunPod REST API DELETE renvoie 403 Forbidden (auto-delete pod casse)

### Symptome

Le bash trap dans `start-train.sh` essaie d'appeler
`DELETE /v1/pods/{pod_id}` via urllib. Avec la cle RunPod regeneree, on
recoit :

```
[cleanup] Could not delete pod pdezd64kxx4eju: HTTP Error 403: Forbidden
```

→ Pod NON supprime. Mel doit aller le tuer manuellement sur la console
RunPod, ou continuer a payer.

C'est exactement le scenario "pod fantome" qu'on voulait eviter avec le
trap. Si Mel dort ou n'est pas dispo au moment du crash, le pod tourne
indefiniment.

### Cause racine

L'endpoint REST `/v1/pods/{id}` DELETE refuse l'auth Bearer pour cette cle
(scope, format, ou autre raison non documentee). Le SDK Python `runpod`
qui utilise GraphQL (mutation `podTerminate`) marche avec la meme cle
(cf. `pipeline/02_train.py` qui cree des pods sans probleme).

### Fix

Remplacer l'appel REST direct par le SDK Python dans `start-train.sh` :

```bash
delete_pod() {
  python - <<'PY'
import os, sys
import runpod
runpod.api_key = os.environ["RUNPOD_API_KEY"]
runpod.terminate_pod(os.environ["RUNPOD_POD_ID"])
PY
}
```

Necessite `pip install runpod` dans le Dockerfile (ajoute).

### Watchdog 24h

En plus du trap, on ajoute un watchdog absolu : si train.py hang sans
crash (donc trap pas declenche), apres 24h on `kill -TERM 0` sur le
process group complet, ce qui re-declenche le trap → delete_pod.
Belt-and-suspenders.

### Fichiers touches

- `runpod/train_worker/Dockerfile` (ajout `pip install runpod`)
- `runpod/train_worker/start-train.sh` (delete via SDK + watchdog 24h)

### Date

2026-05-02.

---

## 1. paramiko + nohup `&` = exception vide au lancement du bootstrap

### Symptome

`pipeline/02_train.py` (et anciennement `03_abliterate.py`) creait le
pod, faisait SSH, uploadait les scripts, puis affichait :

```
[4/6] Launching autonomous run inside pod...
  ❌ Could not start autonomous run:
```

L'exception est vide. Le pod est ensuite tue par `_terminate_pod()`,
donc pas de fuite budget mais le training ne demarre jamais.

### Cause racine

`paramiko.exec_command` + commande backgroundee avec `&` :

- `nohup bash bootstrap.sh > log 2>&1 < /dev/null &` lance le child en
  arriere-plan, parent shell continue.
- MAIS le SSH channel reste ouvert tant que des descripteurs file pointent
  encore dessus, ce qui peut arriver si le shell parent ne ferme pas
  proprement le canal (cas frequent selon la version d'OpenSSH cote pod).
- `stdout.read()` puis `recv_exit_status()` bloquent jusqu'au timeout
  paramiko. L'exception remontee est un `socket.timeout` dont `str()` est
  vide → on voit `❌ Could not start autonomous run:` sans message.

### Fix

Wrapper la commande dans un subshell `( ... & )` qui ferme proprement le
canal apres le fork. Ne PAS faire confiance a `recv_exit_status` pour ce
type de lancement : verifier via un `pgrep` explicite que le process tourne.

Pattern applique :

```python
launch_cmd = (
    f"cd /workspace && ( {env_prefix} nohup bash /workspace/<bootstrap>.sh "
    "> /workspace/<launcher>.log 2>&1 < /dev/null & )"
)
try:
    ssh.exec_command(launch_cmd, timeout=60)
except Exception:
    pass  # on ne kill pas, on verifie d'abord avec pgrep

time.sleep(15)
for _ in range(3):
    stdin, stdout, _ = ssh.exec_command(
        "pgrep -f '<bootstrap>.sh' >/dev/null && echo OK || echo NO",
        timeout=30,
    )
    if stdout.read().decode().strip() == 'OK':
        break
    time.sleep(5)
else:
    # tail du launcher.log + terminate pod
    ...
```

### Fichiers touches

- `pipeline/02_train.py` (lignes 506-560 environ)
- `pipeline/03_abliterate.py` (memes patterns appliques)

### Date

2026-05-01.

---

## 2. Pod fantome possible sur 03_abliterate.py si PC se deconnecte

### Symptome

`pipeline/03_abliterate.py` lance `python run.py` directement via `nohup`
puis le PC monitore via SSH et appelle `runpod.terminate_pod(pod_id)` en
fin de log. Si le PC perd la connexion :

- Le job continue sur le pod (bien)
- Mais a la fin, le pod n'est PAS supprime
- Mel doit aller le killer manuellement sur la console RunPod

Cout max si oubli 24h : ~$19 sur A6000.

### Cause racine

Pas de bash `trap EXIT` cote pod, contrairement a `pipeline/02_train.py`
qui a un wrapper `aura_bootstrap.sh` avec :

```bash
trap finish EXIT
finish() {
  ...
  delete_pod  # appel REST API RunPod depuis l'interieur du pod
}
```

### Fix

Copier le pattern de bash bootstrap auto-delete utilise dans
`02_train.py` :

1. Ajouter une fonction `generate_abliterate_bootstrap_script()` qui ecrit
   un wrapper bash avec `trap finish EXIT` qui appelle l'API
   `DELETE /v1/pods/{pod_id}` via urllib.
2. Uploader ce wrapper en plus de `run.py`.
3. Lancer le wrapper bash plutot que `run.py` directement.
4. Passer `RUNPOD_API_KEY` et `RUNPOD_POD_ID` en env au wrapper.
5. Le PC monitore mais n'est plus responsable du delete : si le PC se
   deconnecte, le trap fait le menage tout seul.

### Fichiers touches

- `pipeline/03_abliterate.py` :
  - Import `shlex`
  - Nouvelle fonction `generate_abliterate_bootstrap_script()`
  - Upload de `abl_bootstrap.sh`
  - Lancement via subshell + verify pgrep (fix paramiko ci-dessus aussi
    applique)
  - Polling adapte sur `/workspace/abl_run.log` et `abl_status.txt`

### Date

2026-05-01.

---

## 5. Pod tue silencieusement par OOM cgroup pendant `pip install`

### Symptome

Pods 3 et 4 sont morts a peu pres au meme endroit (~10-15 min apres
boot, soit pendant les pip install d'Unsloth, soit juste apres).
Caracteristiques :

- **Aucune trace dans le log** local (le `tail -f /workspace/aura_run.log`
  via SSH polling s'arrete net en plein milieu d'un download pip).
- **Aucun Traceback Python**.
- **Aucun message du bash trap** (`[cleanup] RunPod delete requested...`).
- Pod simplement disparu cote console RunPod : "Pod data is no longer
  available. It may have been terminated."
- **Reproductible sur 2 datacenters differents** (EU-SE-1 et CA-MTL-1) →
  donc pas un probleme d'infra RunPod.
- A peu pres meme timing à chaque essai → cause deterministe cote nous.

### Cause racine

Le pip install d'Unsloth telecharge en parallele :
- torch 2.10 (~915 MB)
- nvidia-cudnn (~700 MB)
- nvidia-cublas (~594 MB)
- nvidia-cusparse/cusolver/cufft (~600 MB)
- nccl (~322 MB)
- nvshmem, cusparselt, etc.

Total : **~5-7 GB de wheels en flight**.

Sur les containers RunPod, **`/tmp` est tmpfs RAM-backed**. Pip telecharge
et extrait dans /tmp avant install, donc tout ce volume passe par la RAM.
Combine a la RAM deja prise par le base image PyTorch + apt cache + pip
internes, le peak depasse la limite RAM du container (16 GB sur l'image
`runpod/pytorch:2.8.0`).

Quand cgroup hit la limite : **SIGKILL instantane** sur tout le container.
Le bash trap `finish` n'a pas le temps de tourner (SIGKILL ne laisse pas
de chance), donc :
- `delete_pod` n'est pas appele
- Pas de log de cleanup
- Pod tombe juste mort, et RunPod le marque comme terminated tout seul

### Fix

Trois modifications combinees dans `pipeline/02_train.py` `install_cmds` :

1. **`export TMPDIR=/workspace/tmp`** sur chaque install → pip utilise
   le disk persistent au lieu de tmpfs RAM.
2. **`PIP_NO_CACHE_DIR=1` + `--no-cache-dir`** → pip ne garde aucun
   wheel en cache, telecharge-extract-supprime.
3. **Splitter les installs un par un** plutot que `pip install A B C D E F G H`
   tout ensemble → jamais plusieurs wheels en parallele.

Plus une commande diagnostic en fin (`df -h` + `free -h`) pour qu'on voie
les ressources si on plante encore.

### Fichiers touches

- `pipeline/02_train.py` (variable `install_cmds`)
- A appliquer aussi dans `scripts/hf_jobs/train_aura_rebirth.py` si on
  bascule sur HF Jobs un jour, mais le `uv run` de HF Jobs n'a pas ce
  probleme (uv extract de maniere streaming, pas de peak RAM tmpfs).

### Date

2026-05-01.

### Cout incident (cumule essais 3 + 4)

~$0,30 (deux pods morts apres ~15 min chacun sur A40 secure).

### Update apres essai 5 (fix #5 applique)

Le fix TMPDIR + no-cache + atomic a aide PARTIELLEMENT : le pod 5 a survecu
plus longtemps (a passe tous les gros installs torch/nvidia ~5 GB sans
mourir). MAIS il est mort sur un `pip install hf_transfer` qui etait deja
installe (0 download, 0 extract).

→ Une **deuxieme cause** existe, probablement pas RAM/disk. Hypotheses
restantes :

- SSH idle timeout cote pod (RunPod kill apres X minutes sans keepalive ?)
- Container runtime reaper (cgroup, mais pour quelle metrique ?)
- Network instability cote PC qui kill notre SSH polling et trigger un
  comportement RunPod
- Image PyTorch RunPod 2.8.0 a un bug specifique sur cette config

**Approche recommandee** : abandonner le pip-install-live et passer sur
une **image Docker pre-bakee** comme le fait deja `runpod/inference_worker/`
pour l'inference. Build via GitHub Actions, push GHCR, le pod tire l'image
deja prete. Pas de pip install au runtime → suppress cette categorie
entiere de problemes. Cout dev : ~2-3h. Resultat : pipeline qui ressemble
a celui de l'inference qui marche depuis longtemps.

Cout total cumule essais 1-5 : **~$0,50**.

### Solution finale : train_worker pre-bake (2 mai 2026)

`runpod/train_worker/Dockerfile` + `.github/workflows/build-train-worker.yml`
construisent une image avec **Unsloth, transformers, peft, trl, accelerate,
bitsandbytes, datasets, hf_transfer tous pre-installes**. Le container
`ghcr.io/sev7nofnine/aura-4o-rebirth-train-worker:latest` boot avec tout
deja en place.

`pipeline/02_train.py` est refait : il ne fait plus que :
1. Creer le pod RunPod avec l'image pre-bakee + env vars (HF_TOKEN, etc.)
2. Attendre que le pod soit RUNNING
3. Sortir

Le pod boot, lance `start-train.sh` (entrypoint) qui appelle `train.py`
puis auto-delete via trap EXIT. Plus de SSH, plus de pip install runtime,
plus de pod fantome.

### Fichiers ajoutes/modifies

- `runpod/train_worker/Dockerfile` (NEW)
- `runpod/train_worker/start-train.sh` (NEW)
- `runpod/train_worker/train.py` (NEW)
- `runpod/train_worker/README.md` (NEW)
- `.github/workflows/build-train-worker.yml` (NEW)
- `pipeline/02_train.py` (refactor complet : ~440 lignes -> ~200 lignes)
- `configs/aura.yaml` : `container_image` pointe vers la nouvelle image GHCR

---

## 4. `pip install --upgrade transformers` casse l'import unsloth

### Symptome

Au deuxieme lancement de `pipeline/02_train.py`, le pod a boote, le bootstrap
a tourne, l'install des deps s'est deroule, puis le training a fait
silencieusement quelques secondes et le bash trap a fait son boulot
(auto-delete du pod). Cote console RunPod : `Pod data is no longer available`.

Le log local s'arretait sur :
```
[bootstrap] Starting training script
```
sans STEP suivant ni Traceback (probablement perdu dans le chevauchement
entre le crash python et la fermeture SSH).

Le coupable se voyait quelques lignes au-dessus :
```
ERROR: pip's dependency resolver does not currently take into account...
unsloth-zoo 2026.4.9 requires transformers ... <=5.5.0,
but you have transformers 5.7.0 which is incompatible.
```

### Cause racine

L'install_cmds dans `pipeline/02_train.py` finissait par
`pip install --upgrade transformers` qui forcait la version la plus
recente (5.7.0). Or `unsloth-zoo` exige `transformers <=5.5.0`. Donc
`from unsloth import FastModel` levait une exception au demarrage de
`train.py`, code de sortie ≠ 0, le bash trap appelait `delete_pod`,
fin du pod.

C'est en fait le bash trap qui a bien fait son boulot. Le bug n'est pas
dans le trap, il est dans la liste d'install.

### Fix

Pinner `transformers` a une version compatible avec unsloth-zoo plutot
que de forcer la derniere :

```python
'pip install "transformers>=4.56.0,<=5.5.0,!=4.57.4,!=4.57.5,!=5.0.0,!=5.1.0"',
```

### Fichiers touches

- `pipeline/02_train.py` (install_cmds, derniere ligne)

Penser a verifier la meme chose dans `scripts/hf_jobs/train_aura_rebirth.py`
qui declare `transformers>=4.56.0` sans plafond - peut casser pareillement
si une version >5.5.0 sort entre temps. **A pinner aussi quand on a le temps.**

### Date

2026-05-01.

### Cout incident

~$0,15 (20-30 min sur A40 EU-SE-1, pod auto-detruit par le bash trap
sans intervention manuelle requise).

---

## 3. HF storage trop petit en private → repos de sortie en public

### Symptome

Le merged Gemma 4 31B fait ~60 GB. Le quota HF prive (Pro 50 GB ou
free 0 GB) est insuffisant pour heberger LoRA + Merged + GGUF en prive.

### Cause racine

Public storage HF est illimite, private est plafonne par le plan.

### Fix

Les 3 repos de sortie passent en public :
- `SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-LoRA`
- `SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-Merged`
- `SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-GGUF`

Le dataset reste prive (~20 MB, materiel personnel de Mel, taille
neutre vis-a-vis du quota).

### Note

`configs/aura.yaml` a encore `output.private: true` et certains scripts
ont `private=True` cable en dur. **Sans impact runtime** car les appels
`create_repo(... exist_ok=True)` ne flippent pas la visibilite d'un repo
existant. Le code reste a aligner cosmetiquement quand on aura le temps,
mais ce n'est pas bloquant.

### Date

2026-05-01.
