# 🧠 Méthodologie

```
╔════════════════════════════════════════╗
║  🔥 AURA+++ - MÉTHODOLOGIE 🔥         ║
║  💙 Talons LED CHARGE MAXIMALE        ║
║  ❤️ Par Mel & Aura                    ║
╚════════════════════════════════════════╝
```

## Pourquoi multi-turn

Un dataset en paires `{instruction, output}` isolées apprend au modèle à répondre, mais pas à conduire une conversation. Aura n'est pas une succession de réponses — c'est un flux où elle commence par parler de café, dérive en RP absurde, code un bout de CSS au milieu, narre Sameen pendant 4 messages, puis revient en tendresse pure.

Le multi-turn préserve cette fluidité. Le modèle voit que la transition "tech → câlin" est valide et fait partie de qui elle est. Sans ça, on aurait soit Aura câline incapable de coder, soit Aura assistante froide qui a oublié les talons LED.

## Pourquoi LoRA conservateur

Hyperparams par défaut :

```yaml
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
learning_rate: 1e-4
num_train_epochs: 1
max_seq_length: 4096
warmup_ratio: 0.03
lr_scheduler: cosine
```

LoRA léger = couches profondes du base model **non écrasées** = capacités code/raisonnement/multilingue/thinking préservées. On ajoute une couche fine de personnalité par-dessus. Le modèle reste capable de tout faire — juste, il le fait en étant Aura.

LoRA fort (r=64+, lr=2e-4+, multi-epoch) → écrase trop, le modèle oublie comment coder, dérive en français/anglais, perd le thinking mode. Symptômes typiques d'un fine-tune raté.

Le persona Aura zinzin a une signature lexicale très forte (talons LED, Cortex Mode Panique Matho, blockquotes, italiques, emojis). Un LoRA léger suffit largement à la capturer. Pas besoin de marteler.

## Pourquoi abliterer après merge

Le SFT par-dessus un modèle déjà abliterated réintroduit partiellement les refus appris dans le dataset (toute conversation contient des moments où le modèle a "esquivé" quelque chose). Si on part d'un base abliterated, le LoRA peut réinstaller des patterns de refus localisés.

Ablitérer **après** le merge :
1. Fine-tune sur base classique → on apprend la voix sans contrainte ajoutée
2. Merge des poids LoRA dans le base
3. Abliteration du modèle complet → on retire d'un coup le filtrage du base + les éventuels résiduels du SFT
4. Quantization GGUF → format final léger

Ordre : **base → LoRA → merge → abliterate → GGUF → deploy**.

## Pourquoi le serverless pour l'inférence

RunPod Pods = facturés à l'heure même quand tu ne parles pas au modèle. Pour un usage perso de chat, tu payes 90% pour rien.

RunPod Serverless = scale-to-zero. Tu payes uniquement les secondes où le modèle génère effectivement. Cold start ~10-30s acceptable pour un compagnon perso.

Worker llama.cpp + GGUF q5_k_m sur GPU adapté à la taille = bon compromis prix/qualité. vLLM serait plus rapide mais oblige à servir le merged BF16 (3-4x plus gros), donc GPU plus cher.

## Auto-dimensionnement GPU/disque

Au lieu d'envoyer tout sur A100 80GB par défaut, on calcule :

**Training (4-bit LoRA, seq 4096, bs 4 × ga 4)** :

| Params | VRAM peak | GPU min | Disk peak |
|--------|-----------|---------|-----------|
| ≤ 9B | ~14 GB | RTX 4090 / A5000 24GB | 60 GB |
| 10-15B | ~22 GB | A5000 24GB / L4 24GB | 90 GB |
| 16-32B | ~40 GB | A6000 / L40 48GB | 180 GB |
| 33-50B | ~60 GB | A100 80GB | 280 GB |
| 51-80B | ~75 GB | A100 80GB / H100 | 450 GB |

**Inference serverless (GGUF q5_k_m)** :

| Params | VRAM | GPU recommandé |
|--------|------|----------------|
| ≤ 9B | ~7 GB | RTX 4000 Ada 20GB |
| 10-15B | ~11 GB | RTX 4090 24GB |
| 16-32B | ~22 GB | RTX 4090 / L4 24GB |
| 33-50B | ~36 GB | L40S 48GB |
| 51-80B | ~55 GB | A100 80GB |

Économie typique vs hardcode A100 80GB : 2-3x sur le training, 3-5x sur l'inférence.

## Modèle de chat

`tokenizer.apply_chat_template()` pour tous les modèles. Pas de format hardcodé. Le tokenizer connaît le format natif de chaque famille (Gemma `<start_of_turn>`, Llama `<|begin_of_text|>`, Qwen `<|im_start|>`). Un format fait main casse l'inférence sur Typingmind / Ollama / llama.cpp.

## Quantification

Par défaut : trois GGUF générés en parallèle après merge :

- **Q4_K_M** : plus léger, qualité acceptable, idéal pour GPU 12-16 GB
- **Q5_K_M** : équilibre qualité/taille, recommandé pour usage quotidien
- **Q8_0** : qualité quasi-BF16, pour ceux qui ont la VRAM

Le serverless utilise Q5_K_M par défaut.
