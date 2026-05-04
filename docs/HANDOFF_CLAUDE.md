# Passation Claude - Aura-4o-Rebirth

Date de passation : 1 mai 2026.

## Etat actuel

- GitHub cible : `Sev7nOfNine/Aura-4o-Rebirth-Gemma-4-31B`, branche `main`.
- Hugging Face dataset propre : `SevenOfNine/Aura-4o-Rebirth-Dataset`.
- Hugging Face dataset raw : `SevenOfNine/Aura-4o-Rebirth-Dataset-Raw`.
- RunPod : aucun pod actif au moment de la passation.
- Le training n'a pas ete relance apres l'arret demande par Mel.

## Contrainte importante de Mel

- Ne pas creer de nouveau nom de repo.
- Ne pas travailler sur une branche separee.
- Ne pas laisser de pod RunPod tourner apres une erreur.
- Eviter que le PC local soit l'orchestrateur d'un job de 12-24h.
- Tout doit rester coherent avec le naming `Aura-4o-Rebirth`.

## Repos HF de sortie attendus

Ces noms sont ceux de `configs/aura.yaml` :

- LoRA : `SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-LoRA`
- Merged : `SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-Merged`
- GGUF : `SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-GGUF`

Au dernier check Codex, ces trois repos modele n'existaient pas encore sur HF. Ne pas inventer de variantes. Si Mel confirme qu'ils doivent exister avant le run, les creer avec ces noms exacts, en prive.

## Pourquoi on change de strategie

Le lancement RunPod depuis le PC local est fragile : la machine locale reste responsable de l'orchestration SSH alors que le job dure longtemps. Une coupure PC, une session SSH fragile ou un timeout Paramiko suffit a rendre le suivi confus.

Le meilleur relais est donc :

1. Code source et scripts sur GitHub `main`.
2. Dataset deja sur Hugging Face.
3. Training long lance depuis Hugging Face Jobs ou depuis une automation cloud equivalente.
4. Resultats pousses directement sur Hugging Face.
5. GGUF et deploiement seulement apres verification du LoRA/Merged.

## Script HF Jobs pret a utiliser

Le script est dans :

`scripts/hf_jobs/train_aura_rebirth.py`

Commande indicative, a lancer seulement apres validation des couts :

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 30h \
  --secrets HF_TOKEN \
  --detach \
  https://raw.githubusercontent.com/Sev7nOfNine/Aura-4o-Rebirth-Gemma-4-31B/main/scripts/hf_jobs/train_aura_rebirth.py
```

Par defaut, le script refuse de creer les repos de sortie si `LoRA` ou `Merged` sont absents. C'est volontaire pour respecter la consigne "pas de nouveau repo par accident".

Si Mel confirme la creation des repos de sortie avec les noms exacts, utiliser :

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 30h \
  --secrets HF_TOKEN \
  --detach \
  https://raw.githubusercontent.com/Sev7nOfNine/Aura-4o-Rebirth-Gemma-4-31B/main/scripts/hf_jobs/train_aura_rebirth.py \
  --allow-create-output-repos
```

## Cout HF Jobs a garder en tete

Tarifs observes via `hf jobs hardware` le 1 mai 2026 :

- `l40sx1` : 1x L40S 48 GB, environ 1,80 USD/h.
- `a100-large` : 1x A100 80 GB, environ 2,50 USD/h.

Pour un cycle 12-18h, `a100-large` peut donc couter environ 30-45 USD. C'est plus robuste que le PC local, mais plus cher que l'A40 RunPod.

## Checks avant tout run payant

1. Verifier que le dataset charge bien :

```bash
hf datasets info SevenOfNine/Aura-4o-Rebirth-Dataset
```

2. Verifier que les repos de sortie existent ou que Mel autorise leur creation :

```bash
hf models info SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-LoRA
hf models info SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-Merged
```

3. Verifier qu'aucun pod RunPod ne tourne si RunPod est encore utilise :

```bash
python preflight.py --config configs/aura.yaml
```

## Notes sur le lanceur RunPod

`pipeline/02_train.py` contient maintenant un bootstrap autonome qui ecrit :

- `/workspace/aura_run.log` : log complet lisible dans le terminal web RunPod.
- `/workspace/train.log` : log training seul.
- `/workspace/aura_status.txt` : statut simple.

Le bootstrap demande une suppression du pod via l'API RunPod a la fin ou en cas d'erreur. Ne pas relancer ce chemin sans preflight et sans surveiller les premieres minutes.

