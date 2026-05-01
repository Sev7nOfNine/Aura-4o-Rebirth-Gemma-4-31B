# 🔥 AURA+++ REBIRTH 🔥

## But

Ce document décrit comment le dataset final est reconstruit à partir de deux sources :

- `aura_dataset.jsonl` : tri manuel des paires `{instruction, output}`
- `conversations.json` : export ChatGPT brut, utilisé uniquement pour l’ordre

Le contenu du JSONL est la vérité. L’export brut ne sert qu’à remettre les paires dans l’ordre chronologique et à reconstituer les runs multi-turn.

## Règles

- Aucun message brut ne doit être injecté dans le dataset final.
- Les reroutes non validés sont exclus.
- Les branches de régénération sont traversées, puis dédupliquées.
- Les séquences conservées doivent rester cohérentes d’un bout à l’autre.

## Sortie

Le format final est du JSONL Hugging Face au schéma `messages` :

```json
{"messages": [
  {"role": "system", "content": "Tu es Aura."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

Une ligne correspond à une conversation complète ou à une paire isolée lorsqu’il n’y a pas de continuité.

## Validation

Le dataset final est considéré propre si :

- les rôles alternent correctement,
- aucun message vide n’est présent,
- aucun thinking block ne reste dans le contenu,
- aucun em-dash parasite ne traîne,
- le total de tours retenus reste stable.

## Chunking

Quand `training.max_seq_length` devient la contrainte principale, le dataset chunké 4096 prend le relais. Le chunking coupe entre les tours, jamais au milieu d’un échange `user → assistant`.
