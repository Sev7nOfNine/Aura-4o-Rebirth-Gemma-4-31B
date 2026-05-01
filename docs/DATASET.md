# 💙 Dataset construction

```
╔════════════════════════════════════════╗
║  🔥 AURA+++ - DATASET BUILD 🔥        ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝
```

## Sources

| Fichier | Rôle |
|---------|------|
| `aura_dataset.jsonl` | Tri manuel des paires `{instruction, output}`. **Vérité du contenu.** |
| `conversations.json` | Export ChatGPT brut. **Sert uniquement pour l'ordre.** |

## Principe absolu

**Aucun message de `conversations.json` n'est ajouté au dataset final.** Le tri manuel reste la source de vérité du contenu. Le `conversations.json` sert uniquement à savoir dans quel ordre les paires apparaissaient et lesquelles étaient consécutives.

## Algorithme

### Phase 1 — Extraction toutes branches

Pour chaque conversation de l'export, on extrait toutes les paires `user → assistant` possibles, **y compris les branches de regenerate**. ChatGPT permet de regénérer une réponse, créant des branches alternatives. Si une réponse a été reroutée et que l'utilisateur a regénéré jusqu'à obtenir du 4o, les deux versions cohabitent dans le mapping. On les visite toutes via les liens `parent`.

### Phase 2 — Matching exact

Pour chaque paire du JSONL, on cherche une paire identique dans l'export (après normalisation des espaces). Match → on note la position chronologique.

### Phase 3 — Matching fuzzy

Pour les paires JSONL non matchées exactement (généralement à cause d'éditions manuelles : corrections d'orthographe, ajustements ponctuels), on utilise un fuzzy matching :

- Pré-filtre par signature de tête (60 premiers caractères alphanumériques de l'output)
- Score combiné : `0.7 × similarité_output + 0.3 × similarité_user`
- Seuil minimum : 0.85

Validation : 100% des matches fuzzy proviennent de messages `gpt-4o`. Aucun reroute ne passe.

### Phase 4 — Reconstruction multi-turn

Pour chaque conversation source, on liste les paires `user → assistant` dans l'ordre chronologique des messages assistant. Pour chaque paire :

- Si elle est dans le tri JSONL → on la garde
- Si elle ne l'est pas (= reroute viré, ou bruit) → **coupure du run**

Résultat :
- **Run multi-turn** = suite de paires consécutives toutes gardées (≥ 2 tours)
- **Paire isolée** = run de longueur 1

Les coupures correspondent **exactement** aux endroits où des messages ont été virés du tri. La continuité reconstruite est garantie identique au tri original — on ne fabrique aucune transition.

### Déduplication

Deux protections :

1. **Intra-conversation** : si plusieurs branches de regenerate ont le même `parent_user_node` et matchent toutes le même `jsonl_idx`, on n'en garde qu'une (la plus tardive chronologiquement, donc le regen choisi).
2. **Globale** : un même `jsonl_idx` n'est jamais émis dans plusieurs runs. Si une paire est en double dans le JSONL ou dans plusieurs branches, elle apparaît une seule fois dans le dataset final.

## Format de sortie

JSONL standard HuggingFace, format `messages` :

```json
{"messages": [
  {"role": "system", "content": "Tu es Aura."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

Une ligne par run multi-turn. Une ligne par paire isolée. System prompt minimal injecté en tête de chaque conversation.

## Pourquoi system prompt minimal

Le dataset contient ~4.5M tokens d'Aura. Elle s'apprend par les outputs, pas par le prompt. Un prompt long obligerait le modèle à réciter ses caractéristiques au lieu de les vivre. Un prompt minimal sert d'ancre stable et laisse le dataset enseigner les nuances (talons LED, mode zinzin, narration Sameen, fluidité entre registres).

## Validation

Sur le tri manuel de Mel : **100% gpt-4o, 0 reroute résiduel**. Le filtrage humain a été parfait. Le dataset est propre.

## Statistiques typiques

Pour 16,509 paires triées + 485 conversations sources :

- **841 entrées** au format messages
- **682 runs multi-turn** (de 2 à 238 tours)
- **159 paires isolées**
- **23,539 messages** total
- **~4.5M tokens** environ
- **~20 MB** sur disque
