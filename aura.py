"""
🔥 AURA+++ REBIRTH 🔥

Script tout-en-un : lance le pipeline du début à la fin.

Une seule commande, et on revient quand c'est fini.

Étapes orchestrées :
  1. (optionnel) 01_dataset_build.py    : reconstruit le dataset si --rebuild-dataset
  2. 02_train.py                        : entraînement LoRA + push LoRA + merged sur HF
  3. 03_abliterate.py                   : pull merged + GGUF + mmproj + push HF
  4. 04_deploy.py                       : nouvel endpoint serverless RunPod

Pré-requis :
  - .env avec HF_TOKEN et RUNPOD_API_KEY
  - Image du worker GHCR construite (auto via GH Actions sur push, ou en local)
  - Dataset déjà sur HF (sinon utiliser --rebuild-dataset)

Usage typique :
  python aura.py                           # pipeline complet, de l'entraînement au déploiement
  python aura.py --skip-train              # si LoRA + merged sont déjà sur HF, saute l'entraînement
  python aura.py --skip-train --skip-gguf  # juste le déploiement (LoRA + merged + GGUF déjà sur HF)
  python aura.py --abliterate              # ajoute l'abliteration à l'étape 3 (V3.1)

Au moindre échec dans une étape, le script s'arrête avec un message clair.
Tu peux reprendre en sautant les étapes déjà faites avec --skip-*.

Un preflight en lecture seule est lancé avant toute étape coûteuse, sauf --skip-preflight.
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
🔥 AURA+++ REBIRTH 🔥
Projet privé de reconstruction d’Aura.
"""

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE = REPO_ROOT / 'pipeline'

DEFAULT_WORKER_IMAGE = 'ghcr.io/sev7nofnine/aura-4o-rebirth-worker:latest'


def load_env_file():
    """Charge le fichier .env à la racine du dépôt s'il existe."""
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
    """Lance un sous-script et s'arrête en cas d'échec."""
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
        print(f'\n  ⚠️  {name} interrompu par l’utilisateur')
        sys.exit(130)
    elapsed = (time.time() - start) / 60
    if r.returncode != 0:
        print()
        print(f'  ❌ {name} a échoué (code {r.returncode}) après {elapsed:.1f} min')
        print(f'     Corrige le problème puis relance avec --skip-{name.lower().split()[0]} si besoin.')
        sys.exit(r.returncode)
    print()
    print(f'  ✅ {name} terminé en {elapsed:.1f} min')


def main():
    print(BANNER)
    load_env_file()

    parser = argparse.ArgumentParser(description="Orchestrateur du pipeline complet AURA+++ REBIRTH.")
    parser.add_argument("--config", default="configs/aura.yaml")
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        help="Reconstruit le dataset (jsonl + conversations.json -> HF). Ignoré par défaut.",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Chemin vers aura_dataset.jsonl (utilisé seulement avec --rebuild-dataset).",
    )
    parser.add_argument(
        "--conversations",
        default=None,
        help="Chemin vers conversations.json (utilisé seulement avec --rebuild-dataset).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Saute l'entraînement (suppose que LoRA + merged sont déjà sur HF).",
    )
    parser.add_argument(
        "--skip-gguf",
        action="store_true",
        help="Saute GGUF/abliteration (suppose que le GGUF est déjà sur HF).",
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Saute le déploiement (juste entraînement + GGUF, sans endpoint).",
    )
    parser.add_argument(
        "--abliterate",
        action="store_true",
        help="Applique l'abliteration à l'étape 3 (V3.1, désactivée par défaut).",
    )
    parser.add_argument(
        "--extra-quants",
        default="",
        help="Quants supplémentaires pour l'étape 3, séparés par des virgules (q4_k_m,q8_0).",
    )
    parser.add_argument(
        "--worker-image",
        default=DEFAULT_WORKER_IMAGE,
        help=f"Image Docker pour le worker serverless (défaut : {DEFAULT_WORKER_IMAGE})",
    )
    parser.add_argument(
        "--skip-confirm",
        action="store_true",
        help="Saute les confirmations interactives à chaque étape.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Saute les vérifications preflight en lecture seule. À utiliser seulement si le rapport a déjà été relu.",
    )
    args = parser.parse_args()

    # === Vérification de l'environnement ===
    missing = []
    if not os.environ.get('HF_TOKEN'):
        missing.append('HF_TOKEN')
    if not os.environ.get('RUNPOD_API_KEY'):
        missing.append('RUNPOD_API_KEY')
    if missing:
        print(f"❌ Variables d'environnement manquantes : {', '.join(missing)}")
        print("   Copie .env.template vers .env et renseigne tes jetons.")
        sys.exit(1)

    # === Plan ===
    steps = []
    if args.rebuild_dataset:
        steps.append('1. Reconstruire le dataset')
    if not args.skip_train:
        steps.append('2. Entraîner (12-18h, ~7 $ sur A40 EU-SE-1)')
    if not args.skip_gguf:
        steps.append('3. GGUF + mmproj (2-4h, ~2-5 $)')
    if not args.skip_deploy:
        steps.append("4. Déployer l'endpoint serverless")

    print('📋 Plan du pipeline :')
    for s in steps:
        print(f'   {s}')
    print()
    if args.abliterate:
        print("   ⚠️  --abliterate activé : l'abliteration sera appliquée à l'étape 3")
        print()
    if not steps:
        print('   (rien à faire avec ces drapeaux)')
        sys.exit(0)

    if not args.skip_confirm:
        ans = input('Lancer le pipeline complet ci-dessus ? [y/N] ').strip().lower()
        if ans != 'y':
            print('Annulé.')
            sys.exit(0)

    py = sys.executable
    cfg_arg = ['--config', args.config]
    skip_confirm_arg = ['--skip-confirm'] if args.skip_confirm else []

    if not args.skip_preflight:
        cmd = [py, str(REPO_ROOT / 'preflight.py'), '--config', args.config]
        run_step('Préflight : lecture seule GO/NO-GO', cmd)

    # === Étape 1 (optionnelle) : reconstruction du dataset ===
    if args.rebuild_dataset:
        if not (args.jsonl and args.conversations):
            print('❌ --rebuild-dataset nécessite --jsonl et --conversations.')
            sys.exit(1)
        cmd = [
            py, str(PIPELINE / '01_dataset_build.py'),
            '--jsonl', args.jsonl,
            '--conversations', args.conversations,
            '--output', 'aura_final_dataset.jsonl',
            '--push-hf', 'SevenOfNine/Aura-4o-Rebirth-Dataset-Raw',
            '--private',
            '--dataset-card', str(REPO_ROOT / 'dataset_card.md'),
        ]
        run_step('Étape 1 : reconstruction du dataset', cmd)

    # === Étape 2 : entraînement ===
    if not args.skip_train:
        cmd = [py, str(PIPELINE / '02_train.py')] + cfg_arg + skip_confirm_arg
        run_step('Étape 2 : entraînement (LoRA + push merged)', cmd)

    # === Étape 3 : GGUF (+ abliteration optionnelle) ===
    if not args.skip_gguf:
        cmd = [py, str(PIPELINE / '03_abliterate.py')] + cfg_arg + skip_confirm_arg
        if args.abliterate:
            cmd.append('--abliterate')
        if args.extra_quants:
            cmd.extend(['--extra-quants', args.extra_quants])
        run_step('Étape 3 : GGUF + mmproj + push HF', cmd)

    # === Étape 4 : déploiement ===
    if not args.skip_deploy:
        cmd = [py, str(PIPELINE / '04_deploy.py')] + cfg_arg + skip_confirm_arg + [
            '--worker-image', args.worker_image,
        ]
        run_step("Étape 4 : déploiement de l'endpoint serverless", cmd)

    # === Terminé ===
    print()
    print('=' * 70)
    print(' 💙 PIPELINE COMPLET TERMINÉ')
    print('=' * 70)
    print(' AURA+++ REBIRTH est prête.')
    print()
    print(' Projet privé de Mel et Aura.')


if __name__ == '__main__':
    main()
