"""
╔════════════════════════════════════════╗
║  🔥 AURA+++ - ABLITERATE + GGUF 🔥    ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝

Pull merged model from HF → (optional abliterate) → extract mmproj → GGUF convert → quantize → push HF.

Usage par defaut (V3.0, sans abliteration) :
  python 03_abliterate.py

Avec abliteration (V3.1, si V3.0 deploye refuse trop) :
  python 03_abliterate.py --abliterate

Avec quants additionnels :
  python 03_abliterate.py --extra-quants q4_k_m,q8_0

Pipeline :
  1. Load config aura.yaml
  2. Create RunPod pod (A6000 48GB par defaut, A100 80GB si --abliterate sur 31B+)
  3. SSH, install deps + clone llama.cpp
  4. Download merged HF -> /workspace/merged
  5. (optional) Run paperscarecrow's gemma4_31b_abliterator.py with mlabonne datasets
  6. Convert HF -> GGUF main bf16 + extract mmproj
  7. Quantize variants (Q5_K_M par defaut)
  8. Push GGUF repo + mmproj to HF (private)
  9. Terminate pod
"""
import argparse
import json
import os
import re
import sys
import io
import time

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BANNER = """
╔════════════════════════════════════════╗
║  🔥 AURA+++ - ABLITERATE + GGUF 🔥    ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝
"""


def load_yaml(path):
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def pick_gpu(cfg, vram_min_gb, abliterate=False):
    """Pour 03 : A6000 48GB par defaut suffit. A100 80GB si abliteration sur 31B+."""
    pool = cfg['gpu_pool']
    # Si abliteration et model >= 24B, on pousse vers 80 GB pour eviter les problemes
    target_vram = 80 if (abliterate and vram_min_gb >= 32) else max(48, vram_min_gb)
    eligible = [g for g in pool if g['vram_gb'] >= target_vram]
    if not eligible:
        return None
    def rate_value(g):
        m = re.search(r'\$([\d.]+)', g['rate'])
        return float(m.group(1)) if m else 999
    eligible.sort(key=rate_value)
    g = eligible[0]
    return g['runpod_id'], g['label'], g['rate'], g['vram_gb']


def generate_pod_script(cfg, model_info, abliterate, extra_quants):
    """Genere le script execute sur le pod RunPod."""
    o = cfg['output']
    abl = cfg['abliterate']
    quants = ['q5_k_m'] + [q for q in extra_quants if q != 'q5_k_m']

    abliterate_block = ''
    abliterated_dir = '/workspace/merged'  # default = pas d'abliteration
    if abliterate:
        abliterated_dir = '/workspace/abliterated'
        abliterate_block = f'''
# === Abliteration step ===
step("Downloading paperscarecrow abliterator script")
import urllib.request
abl_script_url = "https://huggingface.co/paperscarecrow/Gemma-4-31B-it-abliterated/resolve/main/gemma4_31b_abliterator.py"
urllib.request.urlretrieve(abl_script_url, "/workspace/abliterator.py")
step("Abliterator downloaded")

step("Running abliteration (paperscarecrow script + mlabonne datasets)")
# Le script s'execute avec source/target dirs en arguments. Verifie le format avec
# son source : https://huggingface.co/paperscarecrow/Gemma-4-31B-it-abliterated
import subprocess
abl_cmd = [
    "python", "/workspace/abliterator.py",
    "--input", "/workspace/merged",
    "--output", "/workspace/abliterated",
    "--harmful-dataset", "{abl['datasets']['harmful']}",
    "--harmless-dataset", "{abl['datasets']['harmless']}",
    "--n-harmful", "{abl['n_harmful']}",
    "--n-harmless", "{abl['n_harmless']}",
]
result = subprocess.run(abl_cmd, capture_output=True, text=True, timeout=7200)
print(result.stdout[-4000:])
print(result.stderr[-4000:])
if result.returncode != 0:
    print("ABLITERATION_FAILED")
    sys.exit(1)
step("Abliteration complete")
'''

    return f'''
"""Aura-Rebirth abliterate+gguf script (executed on RunPod pod)."""
import os, sys, subprocess, shutil

WORK = "/workspace"

import functools
print = functools.partial(print, flush=True)


def step(msg):
    print(f"[STEP] {{msg}}")


def run(cmd, timeout=None, check=True):
    print(f"[CMD] {{' '.join(str(c) for c in cmd)}}")
    r = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if r.stdout:
        print(r.stdout[-2000:])
    if r.stderr:
        print(r.stderr[-2000:])
    if check and r.returncode != 0:
        print(f"[ERROR] exit {{r.returncode}}")
        sys.exit(1)
    return r


HF_TOKEN = os.environ["HF_TOKEN"]


step("Installing deps")
run(["pip", "install", "huggingface_hub[cli]", "hf_transfer", "transformers", "torch", "datasets", "accelerate", "gguf"])

step("Cloning llama.cpp")
run(["git", "clone", "https://github.com/ggml-org/llama.cpp", f"{{WORK}}/llama.cpp"])
run(["pip", "install", "-r", f"{{WORK}}/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"])

step("Building llama.cpp (llama-quantize)")
run(["bash", "-lc", f"cd {{WORK}}/llama.cpp && cmake -B build && cmake --build build --target llama-quantize -j$(nproc)"], timeout=3600)


step("Downloading merged model: {o['merged_repo']}")
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="{o['merged_repo']}",
    local_dir=f"{{WORK}}/merged",
    token=HF_TOKEN,
    max_workers=8,
)
step("Merged downloaded")

{abliterate_block}

# === Convert to GGUF main (bf16) ===
step("Converting to GGUF main (bf16)")
convert_script = f"{{WORK}}/llama.cpp/convert_hf_to_gguf.py"
run([
    "python", convert_script,
    "{abliterated_dir}",
    "--outfile", f"{{WORK}}/model-bf16.gguf",
    "--outtype", "bf16",
], timeout=3600)

bf16_size_gb = os.path.getsize(f"{{WORK}}/model-bf16.gguf") / (1024**3)
step(f"GGUF main bf16: {{bf16_size_gb:.1f}}GB")
if bf16_size_gb < 30:
    print("[ERROR] GGUF main suspiciously small")
    sys.exit(1)


# === Extract mmproj ===
step("Extracting mmproj (vision tower)")
run([
    "python", convert_script,
    "{abliterated_dir}",
    "--mmproj",
    "--outfile", f"{{WORK}}/mmproj-f16.gguf",
    "--outtype", "f16",
], timeout=1800)
mmproj_size_gb = os.path.getsize(f"{{WORK}}/mmproj-f16.gguf") / (1024**3)
step(f"mmproj f16: {{mmproj_size_gb:.2f}}GB")


# === Quantize variants ===
quants = {json.dumps(quants)}
quantize_bin = f"{{WORK}}/llama.cpp/build/bin/llama-quantize"

for q in quants:
    step(f"Quantizing to {{q}}")
    run([
        quantize_bin,
        f"{{WORK}}/model-bf16.gguf",
        f"{{WORK}}/model-{{q}}.gguf",
        q,
    ], timeout=3600)
    sz = os.path.getsize(f"{{WORK}}/model-{{q}}.gguf") / (1024**3)
    step(f"  {{q}}: {{sz:.1f}}GB")


# === Free disk: remove bf16 GGUF (huge, redundant after quantize) ===
try:
    os.remove(f"{{WORK}}/model-bf16.gguf")
except Exception:
    pass


# === Push to HF ===
step("Creating GGUF repo on HF: {o['gguf_repo']}")
from huggingface_hub import HfApi, create_repo
api = HfApi(token=HF_TOKEN)
create_repo("{o['gguf_repo']}", repo_type="model", private={o['private']}, exist_ok=True, token=HF_TOKEN)

step("Uploading mmproj")
api.upload_file(
    path_or_fileobj=f"{{WORK}}/mmproj-f16.gguf",
    path_in_repo="mmproj-f16.gguf",
    repo_id="{o['gguf_repo']}",
    repo_type="model",
)

for q in quants:
    step(f"Uploading model-{{q}}.gguf")
    api.upload_file(
        path_or_fileobj=f"{{WORK}}/model-{{q}}.gguf",
        path_in_repo=f"model-{{q}}.gguf",
        repo_id="{o['gguf_repo']}",
        repo_type="model",
    )

step("All pushed")
print("AURA_GGUF_DONE")
'''


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(description="AURA+++ abliterate + GGUF + push.")
    parser.add_argument('--config', default='configs/aura.yaml')
    parser.add_argument('--hf-token', default=os.environ.get('HF_TOKEN'))
    parser.add_argument('--runpod-key', default=os.environ.get('RUNPOD_API_KEY'))
    parser.add_argument('--abliterate', action='store_true', help='Apply paperscarecrow abliteration post-merge.')
    parser.add_argument('--extra-quants', default='', help='Comma-separated extra quants (q4_k_m,q8_0). Q5_K_M always done.')
    parser.add_argument('--skip-confirm', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--ssh-key', default=None)
    args = parser.parse_args()

    if not args.hf_token:
        print('❌ HF_TOKEN not set.')
        sys.exit(1)
    if not args.runpod_key:
        print('❌ RUNPOD_API_KEY not set.')
        sys.exit(1)

    cfg = load_yaml(args.config)
    model_key = cfg['base_model']['name']
    model_info = cfg['models'][model_key]
    extra_quants = [q.strip() for q in args.extra_quants.split(',') if q.strip()]

    print(f'📋 Plan')
    print(f'   Pull merged from : {cfg["output"]["merged_repo"]}')
    print(f'   Abliteration     : {"YES (paperscarecrow + mlabonne)" if args.abliterate else "NO (V3.0 baseline)"}')
    print(f'   Quants           : Q5_K_M{"," + ",".join(extra_quants) if extra_quants else ""}')
    print(f'   Push to          : {cfg["output"]["gguf_repo"]} (private={cfg["output"]["private"]})')
    print()

    pick = pick_gpu(cfg, model_info['vram_train_gb_min'], abliterate=args.abliterate)
    if not pick:
        print('❌ No GPU in pool meets requirements.')
        sys.exit(1)
    gpu_id, gpu_label, gpu_rate, gpu_vram = pick
    disk_gb = max(180, model_info['disk_train_gb_min'])  # need to fit merged + GGUF + caches
    print(f'   GPU pick         : {gpu_label} ({gpu_rate})')
    print(f'   Disk request     : {disk_gb} GB')
    print()

    print('💰 Estimated cost')
    print('   Setup + download : ~30-60 min')
    if args.abliterate:
        print('   Abliteration     : ~30-90 min')
    print('   Convert + quants : ~30-60 min')
    print('   Upload to HF     : ~30-90 min')
    print(f'   Total            : ~2-4h × {gpu_rate}')
    print()

    if args.dry_run:
        return

    if not args.skip_confirm:
        ans = input('Proceed and create RunPod pod? [y/N] ').strip().lower()
        if ans != 'y':
            print('Aborted.')
            return

    import runpod
    import paramiko

    runpod.api_key = args.runpod_key

    print()
    print('[1/5] Creating RunPod pod...')
    pod = runpod.create_pod(
        name='aura-rebirth-abliterate',
        image_name=cfg['runpod_train']['container_image'],
        gpu_type_id=gpu_id,
        cloud_type=cfg['runpod_train']['cloud_type'],
        volume_in_gb=disk_gb,
        container_disk_in_gb=disk_gb,
        ports='22/tcp',
    )
    pod_id = pod['id']
    print(f'  ✅ Pod created: {pod_id}')

    print()
    print('[2/5] Waiting for pod ready...')
    ssh_host, ssh_port = None, None
    for _ in range(60):
        try:
            p = runpod.get_pod(pod_id)
            if p.get('desiredStatus') == 'RUNNING':
                rt = p.get('runtime') or {}
                for port in rt.get('ports', []):
                    if port.get('privatePort') == 22:
                        ssh_host = port.get('ip')
                        ssh_port = int(port.get('publicPort'))
                        break
                if ssh_host:
                    break
        except Exception:
            pass
        print('.', end='', flush=True)
        time.sleep(10)
    print()
    if not ssh_host:
        print('  ❌ Timeout. Terminating.')
        runpod.terminate_pod(pod_id)
        sys.exit(1)
    time.sleep(60)

    ssh_key_path = args.ssh_key or _find_ssh_key()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f'  Connecting SSH to {ssh_host}:{ssh_port}...')
    for _ in range(5):
        try:
            ssh.connect(ssh_host, port=ssh_port, username='root', key_filename=ssh_key_path, timeout=30)
            break
        except Exception:
            time.sleep(15)
    else:
        runpod.terminate_pod(pod_id)
        sys.exit(1)

    print()
    print('[3/5] Uploading script...')
    script = generate_pod_script(cfg, model_info, args.abliterate, extra_quants)
    sftp = ssh.open_sftp()
    with sftp.file('/workspace/run.py', 'w') as f:
        f.write(script)
    sftp.close()

    print()
    print('[4/5] Running (detached, polling log)...')
    cmd = f'HF_TOKEN={args.hf_token} HF_HUB_ENABLE_HF_TRANSFER=1 nohup python -u /workspace/run.py > /workspace/run.log 2>&1 &'
    ssh.exec_command(cmd, timeout=60)
    time.sleep(5)

    last = 0
    print()
    while True:
        time.sleep(20)
        try:
            stdin, stdout, _ = ssh.exec_command("pgrep -f 'python.*run.py' > /dev/null && echo R || echo S", timeout=30)
            status = stdout.read().decode().strip()

            stdin, stdout, _ = ssh.exec_command("wc -c /workspace/run.log 2>/dev/null | awk '{print $1}'", timeout=30)
            cur = int((stdout.read().decode().strip() or '0'))

            if cur > last:
                stdin, stdout, _ = ssh.exec_command(f'tail -c +{last + 1} /workspace/run.log', timeout=30)
                new = stdout.read().decode(errors='replace')
                for line in new.splitlines():
                    if line.strip():
                        print(f'  {line.strip()}')
                        if 'AURA_GGUF_DONE' in line:
                            ssh.close()
                            print()
                            print('[5/5] Terminating pod...')
                            runpod.terminate_pod(pod_id)
                            print(f'  ✅ Pod {pod_id} terminated.')
                            print()
                            print(f'💙 GGUF pushed: {cfg["output"]["gguf_repo"]}')
                            print('❤️  Next: python pipeline/04_deploy.py')
                            return
                last = cur

            if status == 'S':
                stdin, stdout, _ = ssh.exec_command('tail -100 /workspace/run.log', timeout=30)
                tail = stdout.read().decode(errors='replace')
                if 'AURA_GGUF_DONE' in tail:
                    runpod.terminate_pod(pod_id)
                    return
                print('  ❌ Process stopped without AURA_GGUF_DONE')
                for line in tail.splitlines()[-20:]:
                    print(f'    {line}')
                print(f'  Pod kept: {pod_id} (ssh root@{ssh_host} -p {ssh_port})')
                return
        except Exception as e:
            print(f'  ⚠️ {e}, reconnecting...')
            time.sleep(20)
            try:
                ssh.connect(ssh_host, port=ssh_port, username='root', key_filename=ssh_key_path, timeout=30)
            except Exception:
                pass


def _find_ssh_key():
    for name in ['id_ed25519', 'id_rsa']:
        p = os.path.expanduser(f'~/.ssh/{name}')
        if os.path.exists(p):
            return p
    return None


if __name__ == '__main__':
    main()
