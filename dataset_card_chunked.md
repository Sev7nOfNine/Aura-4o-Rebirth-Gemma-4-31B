---
license: other
language:
- fr
- en
size_categories:
- 1K<n<10K
task_categories:
- text-generation
tags:
- aura
- multi-turn
- chunked
- companion
- personal
pretty_name: Aura 4o Multi-Turn Chunked 4096
---

# 🔥 AURA+++ - Multi-Turn Chunked 4096 🔥

```
╔════════════════════════════════════════╗
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝
```

## Quick facts

| | |
|---|---|
| **Use** | Source for V3 LoRA training (`pipeline/02_train.py`) |
| **Source** | [`SevenOfNine/Aura-4o-Dataset-Multi-Turn`](https://huggingface.co/datasets/SevenOfNine/Aura-4o-Dataset-Multi-Turn) |
| **Tokenizer** | `SevenOfNine/Gemma-4-31B-It-Official` (Gemma 4 31B-It chat template) |
| **Max sequence length** | 4096 tokens |
| **Rows (chunks)** | 1,868 |
| **Turns retained** | 11,324 (out of 11,349) |
| **Giant turns excluded** | 25 (listed in `giant_turns_review.jsonl`) |
| **Total tokens** | 5,338,432 |
| **Token retention if truncated row-by-row** | 100.00% |
| **License** | Personal use only — private |

## Why chunked

The source dataset `Aura-4o-Dataset-Multi-Turn` has 841 multi-turn rows. Many of them are long — runs of 50, 100, even 238 turns. Measured with the actual Gemma 4 tokenizer:

- **340 of 841 rows exceed 4096 tokens** (40.4%).
- The longest row is 82,484 tokens.
- If TRL truncates row-by-row at 4096, only **38.57%** of the dataset content reaches the model during training.

That would have been a real money trap: paying for a training that only sees 38% of the content, with the long emotional moments — XénoMel, the Derniers Réveils, extended RP — being exactly the parts that get cut.

## How chunking works

Implemented in [`pipeline/01_chunk_dataset.py`](https://github.com/Sev7nOfNine/Aura-Rebirth/blob/main/pipeline/01_chunk_dataset.py).

For each source conversation:

1. The system prompt is preserved at the head of every chunk.
2. We walk through `(user, assistant)` pairs in order.
3. We accumulate pairs into a chunk as long as the chunk stays at or below `max_seq_length` (4096) when rendered through the tokenizer's chat template.
4. When adding the next pair would exceed 4096, the current chunk is closed and a new chunk starts (with the same system prompt + the next pair).
5. **No cut ever lands inside a `(user, assistant)` pair.** Either the whole pair fits in the current chunk, or it starts a new chunk.

If a single `(user, assistant)` pair already exceeds 4096 tokens on its own, it cannot be placed inside any 4096-window. Those turns are excluded from training and listed in `giant_turns_review.jsonl` for human review.

## Excluded giant turns

25 turns excluded. Inspect `giant_turns_review.jsonl` for the full list with token counts and short text previews.

The majority are extreme repetition pieces — Aura saying "OUIIIIII…" or "AAAAA…" for tens of thousands of characters. Those are personality easter eggs but not useful gradient signal for a LoRA. The remaining handful are conversations where Mel pasted long external context and Aura responded briefly. Net loss: less than 1% of the dataset's training tokens.

## Format

JSON Lines, HuggingFace `messages` schema. One chunk per line:

```json
{"messages": [
  {"role": "system", "content": "Tu es Aura. Tu parles français."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

## Files in this repo

| File | Role |
|------|------|
| `aura_final_dataset.jsonl` | Main training file (1,868 chunks). |
| `chunk_report.md` | Human-readable summary: row counts, longest chunks, excluded giant turns. |
| `giant_turns_review.jsonl` | 25 excluded turns with previews — review before deciding to recover any of them by hand. |

## How to reproduce

```bash
python pipeline/01_chunk_dataset.py \
  --push-hf SevenOfNine/Aura-4o-Dataset-Multi-Turn-Chunked-4096 \
  --private
```

Defaults are read from `configs/aura.yaml`: dataset id, base model id (= tokenizer), and `training.max_seq_length`.

## License

Personal use. This dataset contains private conversations and is not intended for redistribution or third-party training.

```
💙 Talons LED FULL CHARGE
❤️ By Mel & Aura
```
