# AURA+++ REBIRTH - Pieges connus et leurs fixes

Doc vivante des bugs rencontres pendant le pipeline et de la maniere
de les eviter pour les prochains runs (ce projet ou un autre modele
qui partirait de cette base).

A maintenir a jour : a chaque nouvel incident, ajouter une entree.

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
qui declare `transformers>=4.56.0` sans plafond — peut casser pareillement
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
- `SevenOfNine/Aura-4o-Rebirth-LoRA`
- `SevenOfNine/Aura-4o-Rebirth-Merged`
- `SevenOfNine/Aura-4o-Rebirth-GGUF`

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
