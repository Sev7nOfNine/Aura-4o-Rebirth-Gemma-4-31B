"""
╔════════════════════════════════════════╗
║  🔥 AURA+++ - DEPLOY 🔥               ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝

Cree un endpoint RunPod Serverless pour servir le GGUF Aura.

PROTECTIONS :
  - Lit la liste 'protect_existing_endpoints' du config (ex: 01p64ykg6u3p0i)
  - REFUSE de modifier ou supprimer un endpoint protege
  - Cree un nouveau network volume + nouveau template + nouvel endpoint
  - L'ancien endpoint reste 100% intact

Pre-requis :
  - L'image docker du worker doit etre buildee et pushee.
    Voir runpod/inference_worker/README.md
  - Le GGUF + mmproj doivent etre sur HF (output de 03_abliterate.py)

Exemple :
  python 04_deploy.py \\
      --worker-image sevenofnine/aura-4o-rebirth-gemma-4-31b-worker:latest

  python 04_deploy.py --worker-image ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest --dry-run
"""
import argparse
import json
import os
import sys
import io
import time

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BANNER = """
╔════════════════════════════════════════╗
║  🔥 AURA+++ - DEPLOY 🔥               ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝
"""


def load_yaml(path):
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(description="AURA+++ deploy serverless endpoint.")
    parser.add_argument('--config', default='configs/aura.yaml')
    parser.add_argument('--hf-token', default=os.environ.get('HF_TOKEN'))
    parser.add_argument('--runpod-key', default=os.environ.get('RUNPOD_API_KEY'))
    parser.add_argument('--worker-image', required=True, help='Docker image of the worker (must be pushed and accessible).')
    parser.add_argument('--gguf-file', default='model-q5_k_m.gguf')
    parser.add_argument('--mmproj-file', default='mmproj-f16.gguf')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-confirm', action='store_true')
    args = parser.parse_args()

    if not args.hf_token:
        print('❌ HF_TOKEN required.')
        sys.exit(1)
    if not args.runpod_key:
        print('❌ RUNPOD_API_KEY required.')
        sys.exit(1)

    cfg = load_yaml(args.config)
    sl = cfg['runpod_serverless']
    o = cfg['output']

    # === Protect existing endpoints (read-only check) ===
    import requests
    H = {'Authorization': f'Bearer {args.runpod_key}', 'Content-Type': 'application/json'}
    protected = set(sl.get('protect_existing_endpoints', []))

    print(f'📋 Plan')
    print(f'   GGUF source       : {o["gguf_repo"]}/{args.gguf_file}')
    print(f'   mmproj source     : {o["gguf_repo"]}/{args.mmproj_file}')
    print(f'   Worker image      : {args.worker_image}')
    print(f'   Endpoint name     : {sl["endpoint_name"]}')
    print(f'   Datacenter        : {sl["network_volume"]["datacenter"]}')
    print(f'   GPU preference    : {", ".join(sl["gpu_preference"])}')
    print(f'   New volume size   : {sl["network_volume"]["size_gb"]} GB')
    print(f'   Scale             : min={sl["scale"]["workers_min"]} max={sl["scale"]["workers_max"]} idle={sl["scale"]["idle_timeout_sec"]}s')
    print(f'   Reasoning default : OFF (capability present, not forced)')
    print()
    if protected:
        print(f'🛡️  Protected endpoints (NEVER touched) : {", ".join(protected)}')
        print()

    # Confirm protected endpoints exist and we won't touch them
    r = requests.get('https://rest.runpod.io/v1/endpoints', headers=H)
    r.raise_for_status()
    existing = r.json()
    for ep in existing:
        if ep.get('id') in protected:
            print(f'   ✅ {ep["id"]} ({ep.get("name")}) confirmed protected')
    print()

    if args.dry_run:
        print('--dry-run, exiting.')
        return

    if not args.skip_confirm:
        ans = input('Proceed and create new endpoint? [y/N] ').strip().lower()
        if ans != 'y':
            print('Aborted.')
            return

    # === Step 1 : Create new network volume ===
    print('[1/4] Creating new network volume...')
    nv_payload = {
        'dataCenterId': sl['network_volume']['datacenter'],
        'name': sl['network_volume']['name'],
        'size': sl['network_volume']['size_gb'],
    }
    r = requests.post('https://rest.runpod.io/v1/networkvolumes', headers=H, json=nv_payload)
    if not r.ok:
        print(f'  ❌ Volume create failed: {r.status_code} {r.text}')
        sys.exit(1)
    volume = r.json()
    volume_id = volume['id']
    print(f'  ✅ Volume created: {volume_id} ({volume.get("size")} GB, {volume.get("dataCenterId")})')

    # === Step 2 : Create template ===
    print()
    print('[2/4] Creating template...')
    # Env vars passed to the worker container
    env_vars = {
        'HF_TOKEN': args.hf_token,
        'HF_GGUF_REPO': o['gguf_repo'],
        'GGUF_FILE': args.gguf_file,
        'MMPROJ_FILE': args.mmproj_file,
        'CHAT_TEMPLATE_URL': sl['worker']['chat_template_url'],
        'CONTEXT_LENGTH': str(sl['inference']['context_length']),
        'N_GPU_LAYERS': str(sl['worker']['n_gpu_layers']),
        'PARALLEL': str(sl['worker']['parallel']),
        'THREADS': str(sl['worker']['threads']),
        'REASONING_FORMAT': sl['worker']['reasoning_format'],
        'REQUIRE_MMPROJ': '1' if sl['worker']['require_mmproj'] else '0',
        'DEFAULT_TEMPERATURE': str(sl['inference']['temperature']),
        'DEFAULT_TOP_P': str(sl['inference']['top_p']),
        'DEFAULT_TOP_K': str(sl['inference']['top_k']),
        'DEFAULT_MIN_P': str(sl['inference']['min_p']),
        'DEFAULT_REPETITION_PENALTY': str(sl['inference']['repetition_penalty']),
        'DEFAULT_MAX_TOKENS': str(sl['inference']['max_tokens']),
    }
    # Per RunPod OpenAPI spec :
    #   env is an OBJECT (dict key->value), not an array of {key, value}
    #   dockerStartCmd is an ARRAY (or omitted if entrypoint suffices)
    template_payload = {
        'name': f'{sl["endpoint_name"]}-template',
        'imageName': args.worker_image,
        'isServerless': True,
        'containerDiskInGb': 5,  # small, big stuff goes on network volume
        'volumeMountPath': '/runpod-volume',
        'env': env_vars,  # object/dict, NOT [{key, value}]
        # dockerStartCmd omis : ENTRYPOINT du Dockerfile prend le relai
    }
    r = requests.post('https://rest.runpod.io/v1/templates', headers=H, json=template_payload)
    if not r.ok:
        print(f'  ❌ Template create failed: {r.status_code} {r.text}')
        sys.exit(1)
    template = r.json()
    template_id = template['id']
    print(f'  ✅ Template created: {template_id}')

    # === Step 3 : Create endpoint ===
    print()
    print('[3/4] Creating serverless endpoint...')
    # Per RunPod OpenAPI spec :
    #   dataCenterIds (array), pas 'locations'
    #   networkVolumeId (string singulier)
    #   workersStandby N'EXISTE PAS dans la spec endpoint creation
    #   scalerValue est en SECONDES si scalerType=QUEUE_DELAY
    endpoint_payload = {
        'name': sl['endpoint_name'],
        'templateId': template_id,
        'networkVolumeId': volume_id,
        'dataCenterIds': [sl['network_volume']['datacenter']],
        'gpuTypeIds': sl['gpu_preference'],
        'gpuCount': 1,
        'computeType': 'GPU',
        'workersMin': sl['scale']['workers_min'],
        'workersMax': sl['scale']['workers_max'],
        'idleTimeout': sl['scale']['idle_timeout_sec'],
        'flashboot': True,
        'scalerType': 'QUEUE_DELAY',
        'scalerValue': 4,
        'executionTimeoutMs': 600000,
    }
    r = requests.post('https://rest.runpod.io/v1/endpoints', headers=H, json=endpoint_payload)
    if not r.ok:
        print(f'  ❌ Endpoint create failed: {r.status_code} {r.text}')
        sys.exit(1)
    ep = r.json()
    ep_id = ep['id']
    print(f'  ✅ Endpoint created: {ep_id}')

    # === Step 4 : Final summary ===
    print()
    print('[4/4] Done.')
    print()
    print('=' * 60)
    print(' 💙 AURA+++ Endpoint deployed')
    print('=' * 60)
    print(f' Endpoint ID  : {ep_id}')
    print(f' Name         : {sl["endpoint_name"]}')
    print(f' URL (sync)   : https://api.runpod.ai/v2/{ep_id}/runsync')
    print(f' URL (openai) : https://api.runpod.ai/v2/{ep_id}/openai/v1/chat/completions')
    print(f' Volume       : {volume_id} ({sl["network_volume"]["size_gb"]} GB, {sl["network_volume"]["datacenter"]})')
    print(f' Template     : {template_id}')
    print(f' GPU pref     : {", ".join(sl["gpu_preference"])}')
    print(f' Scale        : 0 → {sl["scale"]["workers_max"]} workers, idle {sl["scale"]["idle_timeout_sec"]}s')
    print()
    print(' Use in TypingMind :')
    print(f'   API endpoint : https://api.runpod.ai/v2/{ep_id}/openai/v1')
    print(f'   API key      : your RunPod key (or aiKey from endpoint)')
    print(f'   Model        : aura  (or anything, llama-server ignores it)')
    print()
    print(' First request will trigger cold-start :')
    print('   - Worker pulls GGUF from HF (~21 GB)  : ~5-15 min')
    print('   - Then llama-server boots             : ~30s')
    print('   Subsequent requests : ~2-5s cold-start (volume cached)')
    print()
    print(' 💙 Talons LED FULL CHARGE.')
    print(' ❤️  By Mel & Aura.')


if __name__ == '__main__':
    main()
