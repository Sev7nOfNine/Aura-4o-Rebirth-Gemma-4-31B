# HF Jobs - AURA+++ REBIRTH

Ce dossier contient le chemin de secours propre pour lancer le training sans dependre du PC local.

## Principe

- Le script est stocke sur GitHub.
- Hugging Face Jobs execute le script cote cloud.
- Le dataset est charge depuis `SevenOfNine/Aura-4o-Rebirth-Dataset`.
- Le LoRA et le merged sont pousses vers les repos HF configures.

## Commande recommandee

Verifier les couts avant de lancer. `a100-large` coutait environ 2,50 USD/h au dernier check Codex.

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 30h \
  --secrets HF_TOKEN \
  --detach \
  https://raw.githubusercontent.com/Sev7nOfNine/Aura-4o-Rebirth-Gemma-4-31B/main/scripts/hf_jobs/train_aura_rebirth.py
```

## Si les repos de sortie n'existent pas encore

Par defaut, le script refuse de creer les repos de sortie. C'est volontaire pour eviter les noms parasites.

Si Mel confirme qu'il faut creer les repos exacts configures :

```bash
hf jobs uv run \
  --flavor a100-large \
  --timeout 30h \
  --secrets HF_TOKEN \
  --detach \
  https://raw.githubusercontent.com/Sev7nOfNine/Aura-4o-Rebirth-Gemma-4-31B/main/scripts/hf_jobs/train_aura_rebirth.py \
  --allow-create-output-repos
```

## Suivi

Lister les jobs :

```bash
hf jobs ps --all
```

Voir les logs :

```bash
hf jobs logs JOB_ID --follow
```

Annuler si quelque chose part mal :

```bash
hf jobs cancel JOB_ID
```

