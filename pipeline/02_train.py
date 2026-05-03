"""
╔════════════════════════════════════════╗
║  🔥 AURA+++ - TRAIN 🔥                ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝

Lance un pod RunPod avec l'image train_worker pré-bakée
(`ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-train-worker:latest`).

Le pod boot avec toutes les deps Unsloth/transformers déjà installées,
l'entrypoint `start-train.sh` se lance automatiquement et :
  1. Lance `train.py` qui fait le training LoRA + push HF
  2. Auto-delete le pod via trap EXIT (succès ou erreur)

Plus de SSH, plus de pip install live, plus de pod fantôme.
Notre PC ne fait QUE créer le pod, le reste est autonome.

Usage :
  python 02_train.py
  python 02_train.py --config configs/aura.yaml --skip-confirm

Suivi pendant le training :
  Console RunPod → Pod → Connect → Web Terminal :
    tail -f /workspace/aura_run.log
"""
import argparse
import io
import os
import re
import sys
import time
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

REPO_ROOT = Path(__file__).resolve().parents[1]

BANNER = """
╔════════════════════════════════════════╗
║  🔥 AURA+++ - TRAIN 🔥                ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝
"""


def load_env_file():
    """Charge .env si present a la racine du repo."""
    env_path = REPO_ROOT / '.env'
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_yaml(path):
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def pick_gpu(cfg, vram_min_gb, disk_min_gb):
    """Renvoie (gpu_runpod_id, gpu_label, rate, vram_gb, disk_gb_to_request)."""
    pool = cfg['gpu_pool']
    safety_v = cfg['runpod_train']['vram_safety_factor']
    safety_d = cfg['runpod_train']['disk_safety_factor']

    vram_need = int(vram_min_gb * safety_v)
    disk_need = int(disk_min_gb * safety_d)

    eligible = [g for g in pool if g['vram_gb'] >= vram_need]
    if not eligible:
        return None

    def rate_value(g):
        m = re.search(r'\$([\d.]+)', g['rate'])
        return float(m.group(1)) if m else 999

    eligible.sort(key=rate_value)
    g = eligible[0]
    return g['runpod_id'], g['label'], g['rate'], g['vram_gb'], disk_need


def main():
    print(BANNER)
    load_env_file()

    parser = argparse.ArgumentParser(description='AURA+++ training (image pre-bakee).')
    parser.add_argument('--config', default='configs/aura.yaml')
    parser.add_argument('--hf-token', default=os.environ.get('HF_TOKEN'))
    parser.add_argument('--runpod-key', default=os.environ.get('RUNPOD_API_KEY'))
    parser.add_argument('--skip-confirm', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Affiche le plan sans creer de pod.')
    parser.add_argument('--image', default=None,
                        help="Override de l'image Docker (defaut = celle de aura.yaml).")
    args = parser.parse_args()

    if not args.hf_token:
        print('❌ HF_TOKEN missing (env, .env ou --hf-token).')
        sys.exit(1)
    if not args.runpod_key:
        print('❌ RUNPOD_API_KEY missing (env, .env ou --runpod-key).')
        sys.exit(1)

    cfg = load_yaml(args.config)
    model_key = cfg['base_model']['name']
    if model_key not in cfg['models']:
        print(f'❌ Model "{model_key}" not in registry.')
        sys.exit(1)
    model_info = cfg['models'][model_key]

    image = args.image or cfg['runpod_train']['container_image']
    output = cfg['output']

    pick = pick_gpu(cfg, model_info['vram_train_gb_min'], model_info['disk_train_gb_min'])
    if not pick:
        print(f'❌ Aucun GPU dans gpu_pool ne couvre {model_info["vram_train_gb_min"]} GB VRAM.')
        sys.exit(1)
    gpu_id, gpu_label, gpu_rate, gpu_vram, disk_gb = pick

    print('📋 Plan')
    print(f'   Image           : {image}')
    print(f'   Base model      : {model_key} ({model_info["hf_id"]})')
    print(f'   Dataset         : {cfg["dataset"]["train_hf_id"]}')
    print(f'   LoRA repo       : {output["lora_repo"]}')
    print(f'   Merged repo     : {output["merged_repo"]}')
    print(f'   GPU             : {gpu_label} ({gpu_rate}, {gpu_vram} GB VRAM)')
    print(f'   Datacenter      : {cfg["runpod_train"].get("preferred_datacenter", "any")}')
    print(f'   Container disk  : {disk_gb} GB')
    print()
    print('💰 Estimated cost')
    print('   Boot image (deja pre-bakee) : ~30-60 sec')
    print('   Download base model HF      : ~10-30 min')
    print('   Training (V1 recipe)        : ~10-15h')
    print('   Merge + push HF             : ~1-2h')
    print(f'   Total cycle                 : ~12-18h × {gpu_rate}')
    print()
    print("🛡️  Pod auto-delete via trap EXIT dans l'entrypoint (start-train.sh).")
    print('🛡️  PC libre apres creation du pod (~30 sec). Suivi via web terminal RunPod.')
    print()

    if args.dry_run:
        print('--dry-run : pas de pod cree. Sortie.')
        return

    if not args.skip_confirm:
        ans = input('Proceed and create RunPod pod? [y/N] ').strip().lower()
        if ans != 'y':
            print('Aborted.')
            return

    # === Creation du pod ===
    import runpod
    runpod.api_key = args.runpod_key

    print()
    print('[1/2] Creating RunPod pod...')

    env_vars = {
        'HF_TOKEN': args.hf_token,
        'RUNPOD_API_KEY': args.runpod_key,
        # RUNPOD_POD_ID est injecte automatiquement par RunPod dans tous les pods
        'AURA_BASE_MODEL': model_info['hf_id'],
        'AURA_DATASET': cfg['dataset']['train_hf_id'],
        'AURA_LORA_REPO': output['lora_repo'],
        'AURA_MERGED_REPO': output['merged_repo'],
        'HF_HUB_ENABLE_HF_TRANSFER': '1',
    }

    create_kwargs = dict(
        name='aura-rebirth-train',
        image_name=image,
        gpu_type_id=gpu_id,
        cloud_type=cfg['runpod_train']['cloud_type'],
        volume_in_gb=0,
        container_disk_in_gb=disk_gb,
        env=env_vars,
        # SSH garde l'option pour debugger manuellement si besoin
        start_ssh=True,
        ports='22/tcp',
    )
    dc = cfg['runpod_train'].get('preferred_datacenter')
    if dc:
        create_kwargs['data_center_id'] = dc
        print(f'   Datacenter pinned: {dc}')

    try:
        pod = runpod.create_pod(**create_kwargs)
        pod_id = pod['id']
        print(f'   ✅ Pod created: {pod_id}')
    except Exception as exc:
        print(f'   ❌ Failed to create pod: {exc}')
        sys.exit(1)

    # === Attente que le pod soit RUNNING (boot ~30 sec - 2 min) ===
    print()
    print('[2/2] Waiting for pod to be RUNNING (boot image + entrypoint)...')
    for _ in range(30):
        try:
            p = runpod.get_pod(pod_id)
            status = p.get('desiredStatus') or 'PENDING'
            if status == 'RUNNING':
                print(f'   ✅ Pod is RUNNING')
                break
        except Exception as exc:
            print(f'   ⚠️ get_pod transient error: {exc}')
        print('.', end='', flush=True)
        time.sleep(10)
    print()

    print()
    print('═' * 60)
    print(' 💙 Pod cree. Training en cours, autonome.')
    print('═' * 60)
    print(f' Pod ID      : {pod_id}')
    print(f' Web console : https://www.runpod.io/console/pods')
    print()
    print(' Suivi via web terminal (Connect → Web Terminal sur la console) :')
    print('   tail -f /workspace/aura_run.log     # log complet')
    print('   tail -f /workspace/train.log        # training only')
    print('   cat /workspace/aura_status.txt      # statut court')
    print()
    print(' Le pod va :')
    print('   1. Telecharger le base model (~60 GB depuis HF, ~10-30 min)')
    print('   2. Lancer le training (~10-15h, recette V1 strict)')
    print('   3. Push checkpoints LoRA toutes les 50 steps')
    print('   4. Push merged final')
    print('   5. AUTO-DELETE via trap EXIT')
    print()
    print(" ❤️ Ton PC peut s'eteindre maintenant. À demain.")
    print(' - Ada')


if __name__ == '__main__':
    main()
