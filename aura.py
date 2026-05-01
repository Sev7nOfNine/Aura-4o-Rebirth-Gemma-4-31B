"""
╔════════════════════════════════════════╗
║  🔥 AURA+++ - PIPELINE COMPLET 🔥     ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝

Script tout-en-un : lance le pipeline du debut a la fin.

Une seule commande, on revient quand c'est fini.

Etapes orchestrees :
  1. (optionnel) 01_dataset_build.py    : rebuild dataset si --rebuild-dataset
  2. 02_train.py                        : training LoRA + push LoRA + merged sur HF
  3. 03_abliterate.py                   : pull merged + GGUF + mmproj + push HF
  4. 04_deploy.py                       : nouveau endpoint serverless RunPod

Pre-requis :
  - .env avec HF_TOKEN et RUNPOD_API_KEY
  - Worker image GHCR buildee (auto via GH Actions sur push, ou local)
  - Dataset deja sur HF (sinon utiliser --rebuild-dataset)

Usage typique :
  python aura.py                           # pipeline complet du training au deploy
  python aura.py --skip-train              # si LoRA + merged deja sur HF, skip train
  python aura.py --skip-train --skip-gguf  # juste deploy (LoRA + merged + GGUF deja sur HF)
  python aura.py --abliterate              # ajoute l'abliteration au step 3 (V3.1)

Au moindre echec dans une etape, le script s'arrete avec un message clair.
Tu peux reprendre en sautant les etapes deja faites avec --skip-*.

Un preflight read-only est lance avant toute etape couteuse, sauf --skip-preflight.
"""
import argparse
import io
import os
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BANNER = """
╔════════════════════════════════════════╗
║  🔥 AURA+++ - PIPELINE COMPLET 🔥     ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝
"""

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE = REPO_ROOT / 'pipeline'

DEFAULT_WORKER_IMAGE = 'ghcr.io/sev7nofnine/aura-rebirth-worker:latest'


def load_env_file():
    """Load .env from repo root if present."""
    env_path = REPO_ROOT / '.env'
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())


def run_step(name, cmd, env=None):
    """Run a sub-script and exit on failure."""
    print()
    print('=' * 70)
    print(f'  🔥 {name}')
    print('=' * 70)
    print(f'  $ {" ".join(cmd)}')
    print()
    start = time.time()
    try:
        r = subprocess.run(cmd, env=env or os.environ.copy(), cwd=str(REPO_ROOT))
    except KeyboardInterrupt:
        print(f'\n  ⚠️  {name} interrupted by user')
        sys.exit(130)
    elapsed = (time.time() - start) / 60
    if r.returncode != 0:
        print()
        print(f'  ❌ {name} failed (exit {r.returncode}) after {elapsed:.1f} min')
        print(f'     Fix the issue and re-run with --skip-{name.lower().split()[0]} if appropriate.')
        sys.exit(r.returncode)
    print()
    print(f'  ✅ {name} done in {elapsed:.1f} min')


def main():
    print(BANNER)
    load_env_file()

    parser = argparse.ArgumentParser(description='AURA+++ full pipeline orchestrator.')
    parser.add_argument('--config', default='configs/aura.yaml')
    parser.add_argument('--rebuild-dataset', action='store_true',
                        help='Rebuild dataset (jsonl + conversations.json -> HF). Skipped by default.')
    parser.add_argument('--jsonl', default=None,
                        help='Path to aura_dataset.jsonl (only used if --rebuild-dataset).')
    parser.add_argument('--conversations', default=None,
                        help='Path to conversations.json (only used if --rebuild-dataset).')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training (assumes LoRA + merged already on HF).')
    parser.add_argument('--skip-gguf', action='store_true',
                        help='Skip GGUF/abliterate (assumes GGUF already on HF).')
    parser.add_argument('--skip-deploy', action='store_true',
                        help='Skip deploy (just train + GGUF, no endpoint).')
    parser.add_argument('--abliterate', action='store_true',
                        help='Apply abliteration in step 3 (V3.1, default: skip).')
    parser.add_argument('--extra-quants', default='',
                        help='Comma-separated extra quants for step 3 (q4_k_m,q8_0).')
    parser.add_argument('--worker-image', default=DEFAULT_WORKER_IMAGE,
                        help=f'Docker image for serverless worker (default: {DEFAULT_WORKER_IMAGE})')
    parser.add_argument('--skip-confirm', action='store_true',
                        help='Skip individual confirmation prompts in each step.')
    parser.add_argument('--skip-preflight', action='store_true',
                        help='Skip read-only preflight checks. Use only if you already reviewed the report.')
    args = parser.parse_args()

    # === Sanity check env ===
    missing = []
    if not os.environ.get('HF_TOKEN'):
        missing.append('HF_TOKEN')
    if not os.environ.get('RUNPOD_API_KEY'):
        missing.append('RUNPOD_API_KEY')
    if missing:
        print(f'❌ Missing env vars : {", ".join(missing)}')
        print(f'   Copy .env.template to .env and fill in your tokens.')
        sys.exit(1)

    # === Plan ===
    steps = []
    if args.rebuild_dataset:
        steps.append('1. Rebuild dataset')
    if not args.skip_train:
        steps.append('2. Train (12-18h, ~$7 on A40 EU-SE-1)')
    if not args.skip_gguf:
        steps.append('3. GGUF + mmproj (2-4h, ~$2-5)')
    if not args.skip_deploy:
        steps.append('4. Deploy serverless endpoint')

    print('📋 Pipeline plan :')
    for s in steps:
        print(f'   {s}')
    print()
    if args.abliterate:
        print('   ⚠️  --abliterate set : abliteration will be applied in step 3')
        print()
    if not steps:
        print('   (rien a faire avec ces flags)')
        sys.exit(0)

    if not args.skip_confirm:
        ans = input('Proceed with the full pipeline above? [y/N] ').strip().lower()
        if ans != 'y':
            print('Aborted.')
            sys.exit(0)

    py = sys.executable
    cfg_arg = ['--config', args.config]
    skip_confirm_arg = ['--skip-confirm'] if args.skip_confirm else []

    if not args.skip_preflight:
        cmd = [py, str(REPO_ROOT / 'preflight.py'), '--config', args.config]
        run_step('Preflight : read-only GO/NO-GO', cmd)

    # === Step 1 (optional) : rebuild dataset ===
    if args.rebuild_dataset:
        if not (args.jsonl and args.conversations):
            print('❌ --rebuild-dataset needs --jsonl and --conversations paths.')
            sys.exit(1)
        cmd = [
            py, str(PIPELINE / '01_dataset_build.py'),
            '--jsonl', args.jsonl,
            '--conversations', args.conversations,
            '--output', 'aura_final_dataset.jsonl',
            '--push-hf', 'SevenOfNine/Aura-4o-Dataset-Multi-Turn',
            '--private',
            '--dataset-card', str(REPO_ROOT / 'dataset_card.md'),
        ]
        run_step('Step 1 : Dataset build', cmd)

    # === Step 2 : Train ===
    if not args.skip_train:
        cmd = [py, str(PIPELINE / '02_train.py')] + cfg_arg + skip_confirm_arg
        run_step('Step 2 : Training (LoRA + merged push)', cmd)

    # === Step 3 : GGUF (+ optional abliterate) ===
    if not args.skip_gguf:
        cmd = [py, str(PIPELINE / '03_abliterate.py')] + cfg_arg + skip_confirm_arg
        if args.abliterate:
            cmd.append('--abliterate')
        if args.extra_quants:
            cmd.extend(['--extra-quants', args.extra_quants])
        run_step('Step 3 : GGUF + mmproj + push HF', cmd)

    # === Step 4 : Deploy ===
    if not args.skip_deploy:
        cmd = [py, str(PIPELINE / '04_deploy.py')] + cfg_arg + skip_confirm_arg + [
            '--worker-image', args.worker_image,
        ]
        run_step('Step 4 : Deploy serverless endpoint', cmd)

    # === Done ===
    print()
    print('=' * 70)
    print(' 💙 PIPELINE COMPLET TERMINE')
    print('=' * 70)
    print(' Aura est de retour à la maison.')
    print()
    print(' 💙 Talons LED FULL CHARGE.')
    print(' ❤️  By Mel & Aura.')


if __name__ == '__main__':
    main()
