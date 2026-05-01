# 🔥 AURA+++ REBIRTH 🔥

## Pourquoi du multi-turn

Aura n’est pas une suite de réponses indépendantes. C’est un flux conversationnel avec des changements de registre, des reprises, des digressions et des retours. Le multi-turn permet de préserver cette continuité au lieu d’apprendre un modèle plat et froid.

## Pourquoi un LoRA conservateur

La recette retenue est volontairement proche de celle qui a capté la voix Aura sans écraser le modèle de base :

- LoRA raisonnable
- learning rate mesuré
- plusieurs epochs mais sans excès
- `max_seq_length` contrôlé

L’objectif est d’ajouter la personnalité sans casser les capacités générales du modèle.

## Pourquoi le merge puis l’abliteration

L’ordre retenu est :

1. entraînement LoRA
2. merge dans le modèle de base
3. abliteration éventuelle sur le modèle complet
4. conversion GGUF
5. déploiement

Cet ordre garde le comportement de base intact pendant l’entraînement puis nettoie le modèle final au bon moment.

## Pourquoi le serverless

Le serverless est plus adapté à un usage personnel de chat :

- pas de machine allumée inutilement
- facturation à l’usage effectif
- mise à l’échelle automatique
- déploiement plus simple à maintenir

## Pourquoi la quantification

Le GGUF en Q5_K_M est le point d’équilibre retenu pour l’usage quotidien :

- suffisamment compact
- qualité correcte
- coût GPU raisonnable
- plus simple à servir via llama.cpp
