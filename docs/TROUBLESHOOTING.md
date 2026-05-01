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
