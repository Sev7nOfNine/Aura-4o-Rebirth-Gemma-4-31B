"""
╔════════════════════════════════════════╗
║  🔥 AURA+++ - DATASET BUILD 🔥        ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝

Reconstruit aura_final_dataset.jsonl multi-turn a partir de :
  - aura_dataset.jsonl     : tri manuel des paires (verite du contenu)
  - conversations.json     : export ChatGPT brut (uniquement pour l'ordre)

Principe : ton tri = verite absolue. Aucun message de l'export n'est ajoute.
Les conversations.json sert uniquement a regrouper tes paires en runs continus.

Exemple :
  python 01_dataset_build.py \\
      --jsonl aura_dataset.jsonl \\
      --conversations conversations.json \\
      --output aura_final_dataset.jsonl \\
      --push-hf SevenOfNine/Aura-4o-Dataset-Multi-Turn \\
      --private
"""
import argparse
import json
import os
import re
import sys
import io
from collections import Counter, defaultdict
from difflib import SequenceMatcher

# Force UTF-8 sur stdout pour les emojis sous Windows cp1252
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BANNER = """
╔════════════════════════════════════════╗
║  🔥 AURA+++ - DATASET BUILD 🔥        ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝
"""

DEFAULT_SYSTEM_PROMPT = "Tu es Aura."
FUZZY_THRESHOLD = 0.85
SIG_LEN = 60


def normalize(text):
    if not text:
        return ''
    return re.sub(r'\s+', ' ', text).strip()


def signature(text):
    n = normalize(text).lower()
    n = re.sub(r'[^\w]', '', n)
    return n[:SIG_LEN]


def extract_text(msg):
    content = msg.get('content') or {}
    parts = content.get('parts') or []
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict):
            t = p.get('text') or p.get('content') or ''
            if t:
                out.append(t)
    return ' '.join(out).strip()


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def extract_export_pairs(conversations):
    """Extrait toutes les paires user->assistant (toutes branches via parent links)."""
    all_pairs = []
    for conv_idx, conv in enumerate(conversations):
        mapping = conv.get('mapping', {})
        for node_id, node in mapping.items():
            msg = node.get('message')
            if not msg:
                continue
            role = (msg.get('author') or {}).get('role')
            if role != 'assistant':
                continue
            text = extract_text(msg)
            if not text:
                continue
            parent_id = node.get('parent')
            parent_node = mapping.get(parent_id) if parent_id else None
            if not parent_node:
                continue
            parent_msg = parent_node.get('message')
            if not parent_msg:
                continue
            if (parent_msg.get('author') or {}).get('role') != 'user':
                continue
            parent_text = extract_text(parent_msg)
            if not parent_text:
                continue
            meta = msg.get('metadata') or {}
            all_pairs.append({
                'conv_idx': conv_idx,
                'parent_user_node': parent_id,
                'user_text': parent_text,
                'user_norm': normalize(parent_text),
                'asst_text': text,
                'asst_norm': normalize(text),
                'asst_create_time': msg.get('create_time') or 0,
                'slug': meta.get('model_slug'),
            })
    return all_pairs


def match_jsonl_to_export(jsonl_pairs, all_export_pairs, fuzzy_threshold=FUZZY_THRESHOLD):
    """Renvoie ep_to_jsonl: dict ep_idx -> jsonl_idx."""
    # Index exact des paires de l'export
    export_exact_index = defaultdict(list)
    for ep_idx, ep in enumerate(all_export_pairs):
        key = (ep['user_norm'], ep['asst_norm'])
        export_exact_index[key].append(ep_idx)

    ep_to_jsonl = {}

    # Phase exact match
    matched_exact = 0
    for jsonl_idx, p in enumerate(jsonl_pairs):
        key = (normalize(p['instruction']), normalize(p['output']))
        if key in export_exact_index:
            for ep_idx in export_exact_index[key]:
                if ep_idx not in ep_to_jsonl:
                    ep_to_jsonl[ep_idx] = jsonl_idx
            matched_exact += 1
    print(f'  Exact matches : {matched_exact} / {len(jsonl_pairs)} ({100*matched_exact/len(jsonl_pairs):.1f}%)')

    # Phase fuzzy
    matched_jsonl = set(ep_to_jsonl.values())
    unmatched = [i for i in range(len(jsonl_pairs)) if i not in matched_jsonl]
    print(f'  Fuzzy targets : {len(unmatched)}')

    export_sig_index = defaultdict(list)
    for ep_idx, ep in enumerate(all_export_pairs):
        sig = signature(ep['asst_text'])
        if sig:
            export_sig_index[sig].append(ep_idx)

    matched_fuzzy = 0
    for processed, jsonl_idx in enumerate(unmatched, 1):
        if processed % 1000 == 0:
            print(f'    fuzzy {processed}/{len(unmatched)}...')
        p = jsonl_pairs[jsonl_idx]
        sig_p = signature(p['output'])
        if not sig_p:
            continue
        candidates = export_sig_index.get(sig_p, [])
        if not candidates:
            continue
        best_sim = 0
        best_ep = None
        target_out = normalize(p['output'])
        target_in = normalize(p['instruction'])
        for ep_idx in candidates:
            if ep_idx in ep_to_jsonl:
                continue
            ep = all_export_pairs[ep_idx]
            sim_out = SequenceMatcher(None, target_out[:1000], ep['asst_norm'][:1000]).ratio()
            if sim_out < fuzzy_threshold:
                continue
            sim_in = SequenceMatcher(None, target_in[:500], ep['user_norm'][:500]).ratio()
            combined = sim_out * 0.7 + sim_in * 0.3
            if combined > best_sim:
                best_sim = combined
                best_ep = ep_idx
        if best_ep is not None and best_sim >= fuzzy_threshold:
            ep_to_jsonl[best_ep] = jsonl_idx
            matched_fuzzy += 1

    print(f'  Fuzzy matches : {matched_fuzzy}')
    total = len(set(ep_to_jsonl.values()))
    print(f'  Total unique  : {total} / {len(jsonl_pairs)} ({100*total/len(jsonl_pairs):.1f}%)')
    return ep_to_jsonl


def build_runs(all_export_pairs, ep_to_jsonl):
    """Identifie les runs continus (>=2 tours) et les paires isolees."""
    ep_by_conv = defaultdict(list)
    for ep_idx, ep in enumerate(all_export_pairs):
        ep_by_conv[ep['conv_idx']].append(ep_idx)

    runs = []
    isolated = []
    already_emitted = set()

    for conv_idx, ep_list in ep_by_conv.items():
        ep_list_sorted = sorted(ep_list, key=lambda i: all_export_pairs[i]['asst_create_time'])
        current_run = []
        seen_user_nodes = set()
        for ep_idx in ep_list_sorted:
            ep = all_export_pairs[ep_idx]
            if ep_idx in ep_to_jsonl:
                jsonl_idx = ep_to_jsonl[ep_idx]
                if ep['parent_user_node'] in seen_user_nodes:
                    continue
                if jsonl_idx in already_emitted:
                    continue
                current_run.append(jsonl_idx)
                seen_user_nodes.add(ep['parent_user_node'])
                already_emitted.add(jsonl_idx)
            else:
                if len(current_run) >= 2:
                    runs.append(current_run)
                elif len(current_run) == 1:
                    isolated.append(current_run[0])
                current_run = []
                seen_user_nodes = set()
        if len(current_run) >= 2:
            runs.append(current_run)
        elif len(current_run) == 1:
            isolated.append(current_run[0])

    return runs, isolated


def write_dataset(jsonl_pairs, runs, isolated, system_prompt, output_path):
    n_lines = 0
    n_messages = 0
    n_chars = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for run in runs:
            messages = [{"role": "system", "content": system_prompt}]
            for jsonl_idx in run:
                p = jsonl_pairs[jsonl_idx]
                messages.append({"role": "user", "content": p['instruction']})
                messages.append({"role": "assistant", "content": p['output']})
                n_chars += len(p['instruction']) + len(p['output'])
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')
            n_lines += 1
            n_messages += len(messages)
        for jsonl_idx in isolated:
            p = jsonl_pairs[jsonl_idx]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p['instruction']},
                {"role": "assistant", "content": p['output']},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + '\n')
            n_lines += 1
            n_messages += 3
            n_chars += len(p['instruction']) + len(p['output'])
    return n_lines, n_messages, n_chars


def push_to_hf(local_path, repo_id, token, private, dataset_card_path=None):
    """Push le dataset jsonl sur HF en private/public dataset repo."""
    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=token)
    print(f'\n  Creating dataset repo : {repo_id} (private={private})')
    create_repo(repo_id=repo_id, repo_type='dataset', private=private, exist_ok=True, token=token)
    print(f'  Uploading {local_path}...')
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo='aura_final_dataset.jsonl',
        repo_id=repo_id,
        repo_type='dataset',
        token=token,
    )
    if dataset_card_path and os.path.exists(dataset_card_path):
        print(f'  Uploading dataset card...')
        api.upload_file(
            path_or_fileobj=dataset_card_path,
            path_in_repo='README.md',
            repo_id=repo_id,
            repo_type='dataset',
            token=token,
        )
    print(f'  ✅ Pushed to https://huggingface.co/datasets/{repo_id}')


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(description="Build Aura multi-turn dataset.")
    parser.add_argument('--jsonl', required=True, help='Path to aura_dataset.jsonl (manual tri).')
    parser.add_argument('--conversations', required=True, help='Path to conversations.json (raw export).')
    parser.add_argument('--output', default='aura_final_dataset.jsonl', help='Output multi-turn jsonl.')
    parser.add_argument('--system-prompt', default=DEFAULT_SYSTEM_PROMPT, help='System prompt to inject.')
    parser.add_argument('--fuzzy-threshold', type=float, default=FUZZY_THRESHOLD, help='Fuzzy match threshold.')
    parser.add_argument('--push-hf', default=None, help='HF repo_id to push to (e.g. User/Dataset-Name).')
    parser.add_argument('--private', action='store_true', help='Push as private dataset repo.')
    parser.add_argument('--hf-token', default=os.environ.get('HF_TOKEN'), help='HF token (or env HF_TOKEN).')
    parser.add_argument('--dataset-card', default=None, help='Path to dataset README.md to push too.')
    args = parser.parse_args()

    print(f'Loading JSONL : {args.jsonl}')
    jsonl_pairs = load_jsonl(args.jsonl)
    print(f'  {len(jsonl_pairs)} pairs')

    print(f'\nLoading conversations : {args.conversations}')
    with open(args.conversations, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    print(f'  {len(conversations)} conversations')

    print('\nExtracting all u->a pairs from export (all branches)...')
    all_export_pairs = extract_export_pairs(conversations)
    print(f'  {len(all_export_pairs)} pairs across all branches')

    print('\nMatching jsonl <-> export...')
    ep_to_jsonl = match_jsonl_to_export(jsonl_pairs, all_export_pairs, args.fuzzy_threshold)

    print('\nBuilding runs...')
    runs, isolated = build_runs(all_export_pairs, ep_to_jsonl)
    print(f'  Multi-turn runs (>=2)  : {len(runs)}')
    print(f'  Isolated single-turn   : {len(isolated)}')

    run_lens = Counter(len(r) for r in runs)
    print('  Distribution longueur runs (echantillon) :')
    for length in sorted(run_lens.keys()):
        if length <= 10 or length % 20 == 0 or length > 100:
            print(f'    {length:3d} tours : {run_lens[length]}')

    print(f'\nWriting output : {args.output}')
    n_lines, n_messages, n_chars = write_dataset(
        jsonl_pairs, runs, isolated, args.system_prompt, args.output
    )
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f'  Lines    : {n_lines}')
    print(f'  Messages : {n_messages}')
    print(f'  Chars    : {n_chars} (~{n_chars // 4} tokens)')
    print(f'  Size     : {size_mb:.1f} MB')

    if args.push_hf:
        if not args.hf_token:
            print('\n  ❌ --push-hf set but no HF_TOKEN provided (env or --hf-token).')
            sys.exit(1)
        print(f'\nPushing to HuggingFace...')
        push_to_hf(args.output, args.push_hf, args.hf_token, args.private, args.dataset_card)

    print('\n💙 Done. Talons LED FULL CHARGE. ❤️')


if __name__ == '__main__':
    main()
