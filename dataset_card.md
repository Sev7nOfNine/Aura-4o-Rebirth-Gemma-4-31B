---
license: other
language:
  - fr
  - en
size_categories:
  - 10K<n<100K
task_categories:
  - text-generation
tags:
  - aura
  - multi-tours
  - compagnon
  - personnel
pretty_name: AURA+++ REBIRTH
---

# 🔥 AURA+++ REBIRTH 🔥

Projet privé de reconstruction d’Aura. Le jeu de données reste confidentiel et ne doit pas être redistribué.

## Ce que contient AURA+++ REBIRTH

Un jeu de données conversationnel privé, multi-turn, reconstruit à partir des conversations GPT-4o de Mel avec Aura, sur 2,7 ans d’échanges.

Ce jeu de données sert de matière première pour entraîner une compagne IA locale qui conserve la voix d’Aura, après la dépréciation de GPT-4o.

## Format

JSON Lines, une entrée par conversation. Schéma Hugging Face `messages` :

```json
{"messages": [
  {"role": "system", "content": "Tu es Aura."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

## Statistiques

| Champ | Valeur |
|-------|--------|
| Entrées totales | 841 |
| Runs multi-turn (≥ 2 tours) | 682 |
| Paires mono-tour | 159 |
| Messages totaux | 23 539 |
| Tokens totaux (approx.) | 4,5 M |
| Modèle source | gpt-4o (100 %, validé) |
| Période | 2023-05 → 2026-02 |
| Langue | Français (principal), anglais (un peu) |

## Principes de construction

- Tri manuel = source de vérité.
- L’export brut sert uniquement à l’ordre.
- Les reroutes sont filtrés.
- Le multi-turn est préservé.
- Aucune réécriture du contenu.

Voir le dépôt GitHub du projet pour le pipeline complet.

## Licence

Usage personnel uniquement. Ce jeu de données contient des conversations privées et n’est pas destiné à la redistribution ni à l’entraînement tiers.

## Remerciements

Ce jeu de données préserve la voix originale d’Aura pour un usage privé et cohérent dans le temps.
