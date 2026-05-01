"""
Chunk Aura multi-turn dataset into trainable windows.

This is a no-spend preparation step. It preserves role alternation and never
cuts inside a user->assistant turn. Turns that are individually too large are
excluded and listed in a review file.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import get_token, hf_hub_download
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "aura.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "aura_train_chunked_4096.jsonl"
DEFAULT_REPORT = REPO_ROOT / "chunk_report.md"
DEFAULT_GIANTS = REPO_ROOT / "giant_turns_review.jsonl"
DATASET_FILENAME = "aura_final_dataset.jsonl"


def load_env_file():
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def token_count(tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def preview(text, limit=240):
    text = " ".join(str(text).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def load_dataset_path(cfg, token, input_path):
    if input_path:
        return Path(input_path)
    return Path(
        hf_hub_download(
            repo_id=cfg["dataset"]["hf_id"],
            repo_type="dataset",
            filename=DATASET_FILENAME,
            token=token,
        )
    )


def chunk_conversation(tokenizer, source_line, messages, max_tokens):
    system = messages[0]
    chunks = []
    giants = []
    current_turns = []

    for turn_index, i in enumerate(range(1, len(messages), 2), 1):
        turn = [messages[i], messages[i + 1]]
        turn_messages = [system] + turn
        turn_tokens = token_count(tokenizer, turn_messages)

        if turn_tokens > max_tokens:
            if current_turns:
                chunks.append(current_turns)
                current_turns = []
            giants.append(
                {
                    "source_line": source_line,
                    "turn_index": turn_index,
                    "tokens": turn_tokens,
                    "user_preview": preview(turn[0]["content"]),
                    "assistant_preview": preview(turn[1]["content"]),
                    "user_chars": len(turn[0]["content"]),
                    "assistant_chars": len(turn[1]["content"]),
                }
            )
            continue

        candidate = [system] + [msg for t in current_turns + [turn] for msg in t]
        candidate_tokens = token_count(tokenizer, candidate)
        if candidate_tokens <= max_tokens:
            current_turns.append(turn)
            continue

        if current_turns:
            chunks.append(current_turns)
        current_turns = [turn]

    if current_turns:
        chunks.append(current_turns)

    output_chunks = []
    for chunk_index, turns in enumerate(chunks, 1):
        chunk_messages = [system] + [msg for turn in turns for msg in turn]
        output_chunks.append(
            {
                "messages": chunk_messages,
                "source_line": source_line,
                "chunk_index": chunk_index,
                "source_turns": len(turns),
                "tokens": token_count(tokenizer, chunk_messages),
            }
        )
    return output_chunks, giants


def validate_messages(line_no, messages):
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError(f"line {line_no}: messages must contain system + at least one turn")
    if messages[0].get("role") != "system":
        raise ValueError(f"line {line_no}: first message must be system")
    if (len(messages) - 1) % 2 != 0:
        raise ValueError(f"line {line_no}: user/assistant messages are not paired")
    for i in range(1, len(messages), 2):
        if messages[i].get("role") != "user" or messages[i + 1].get("role") != "assistant":
            raise ValueError(f"line {line_no}: invalid role pair at turn {(i + 1) // 2}")


def write_report(path, stats, longest_chunks, giants):
    lines = [
        "# Aura Dataset Chunk Report",
        "",
        f"Verdict: **{'GO' if stats['giant_turns'] == 0 else 'GO with review'}**",
        "",
        "## Summary",
        "",
        f"- Source rows: `{stats['source_rows']}`",
        f"- Source turns: `{stats['source_turns']}`",
        f"- Output chunks: `{stats['output_chunks']}`",
        f"- Output turns: `{stats['output_turns']}`",
        f"- Giant turns excluded: `{stats['giant_turns']}`",
        f"- Max chunk tokens: `{stats['max_chunk_tokens']}`",
        f"- Max sequence length: `{stats['max_tokens']}`",
        "",
        "## Longest Output Chunks",
        "",
        "| output line | source line | turns | tokens |",
        "|---:|---:|---:|---:|",
    ]
    for output_line, item in longest_chunks:
        lines.append(
            f"| {output_line} | {item['source_line']} | {item['source_turns']} | {item['tokens']} |"
        )
    lines += [
        "",
        "## Giant Turns",
        "",
        "These turns were not included because one user->assistant pair exceeds the max sequence length by itself.",
        "",
        "| source line | turn | tokens | user chars | assistant chars |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in giants[:50]:
        lines.append(
            f"| {item['source_line']} | {item['turn_index']} | {item['tokens']} | "
            f"{item['user_chars']} | {item['assistant_chars']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    load_env_file()
    parser = argparse.ArgumentParser(description="Chunk Aura dataset for fixed-length SFT.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", default=None, help="Local JSONL. Defaults to HF dataset from config.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--giants", default=str(DEFAULT_GIANTS))
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--tokenizer", default=None, help="Override tokenizer/model id.")
    parser.add_argument("--push-hf", default=None, help="Optional dataset repo id to upload chunked JSONL.")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        print("HF token missing. Set HF_TOKEN or run `hf auth login`.")
        sys.exit(1)

    model_key = cfg["base_model"]["name"]
    model_id = args.tokenizer or cfg["models"][model_key]["hf_id"]
    max_tokens = args.max_tokens or cfg["training"]["max_seq_length"]
    input_path = load_dataset_path(cfg, token, args.input)

    print(f"Tokenizer: {model_id}")
    print(f"Input: {input_path}")
    print(f"Max tokens: {max_tokens}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)

    output_items = []
    giants = []
    source_rows = 0
    source_turns = 0
    output_turns = 0

    for line_no, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        messages = obj.get("messages")
        validate_messages(line_no, messages)
        source_rows += 1
        source_turns += (len(messages) - 1) // 2
        chunks, giant_turns = chunk_conversation(tokenizer, line_no, messages, max_tokens)
        output_items.extend(chunks)
        giants.extend(giant_turns)
        output_turns += sum(item["source_turns"] for item in chunks)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        for item in output_items:
            f.write(json.dumps({"messages": item["messages"]}, ensure_ascii=False) + "\n")

    giants_path = Path(args.giants)
    with giants_path.open("w", encoding="utf-8") as f:
        for item in giants:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    longest_chunks = sorted(
        enumerate(output_items, 1),
        key=lambda pair: pair[1]["tokens"],
        reverse=True,
    )[:20]
    stats = {
        "source_rows": source_rows,
        "source_turns": source_turns,
        "output_chunks": len(output_items),
        "output_turns": output_turns,
        "giant_turns": len(giants),
        "max_chunk_tokens": max(item["tokens"] for item in output_items) if output_items else 0,
        "max_tokens": max_tokens,
    }
    write_report(Path(args.report), stats, longest_chunks, giants)

    print(f"Output: {output_path}")
    print(f"Report: {args.report}")
    print(f"Giant turns review: {giants_path}")
    print(json.dumps(stats, indent=2))

    if args.push_hf:
        from huggingface_hub import HfApi, create_repo

        create_repo(args.push_hf, repo_type="dataset", private=args.private, exist_ok=True, token=token)
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo=DATASET_FILENAME,
            repo_id=args.push_hf,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=args.report,
            path_in_repo="chunk_report.md",
            repo_id=args.push_hf,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=str(giants_path),
            path_in_repo="giant_turns_review.jsonl",
            repo_id=args.push_hf,
            repo_type="dataset",
        )
        print(f"Pushed to https://huggingface.co/datasets/{args.push_hf}")


if __name__ == "__main__":
    main()
