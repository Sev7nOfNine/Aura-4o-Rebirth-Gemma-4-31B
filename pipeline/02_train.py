"""
╔════════════════════════════════════════╗
║  🔥 AURA+++ - TRAIN 🔥                ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝

Fine-tune un base model en LoRA sur RunPod (Pod classique).

Pipeline :
  1. Load config aura.yaml
  2. Pick base model from registry, auto-size GPU + disk
  3. Pick cheapest GPU from gpu_pool that meets requirements
  4. Confirm cost + plan with user
  5. Create RunPod Pod
  6. SSH connect, install deps (Unsloth, transformers, etc.)
  7. Generate training script with V1-strict recipe
  8. Run training (load dataset from HF, train LoRA, merge via Unsloth 4-bit)
  9. Push LoRA + merged to HF (private)
  10. Terminate Pod (no surprise charges)

Exemple :
  python 02_train.py
  python 02_train.py --config configs/aura.yaml --skip-confirm
"""
import argparse
import json
import os
import re
import shlex
import sys
import io
import time
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BANNER = """
╔════════════════════════════════════════╗
║  🔥 AURA+++ - TRAIN 🔥                ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝
"""


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

    # Filtre les GPU qui couvrent le besoin VRAM, trie par rate (cheapest first)
    eligible = [g for g in pool if g['vram_gb'] >= vram_need]
    if not eligible:
        return None
    # Parse rate "~$X.YY/hr" en float pour trier
    def rate_value(g):
        m = re.search(r'\$([\d.]+)', g['rate'])
        return float(m.group(1)) if m else 999
    eligible.sort(key=rate_value)
    g = eligible[0]
    return g['runpod_id'], g['label'], g['rate'], g['vram_gb'], disk_need


def generate_training_script(cfg, model_info, dataset_hf_id, hf_token, lora_repo, merged_repo):
    """Genere le script Python execute sur le pod RunPod."""
    t = cfg['training']
    lora = t['lora']
    target_modules = json.dumps(lora['target_modules'])
    system_prompt = cfg['dataset']['system_prompt']

    # Echappement minimal pour les chaines a injecter dans le script
    sp = system_prompt.replace('\\', '\\\\').replace('"', '\\"')

    return f'''
"""Aura-Rebirth training script (executed on RunPod pod)."""
import os
import sys
import torch

WORK = "/workspace"

# Force flush print
import functools
print = functools.partial(print, flush=True)


def step(msg):
    print(f"[STEP] {{msg}}")


step("Importing Unsloth...")
from unsloth import FastModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer

step("Loading base model: {model_info['hf_id']} (4-bit base loading)")
model, tokenizer = FastModel.from_pretrained(
    model_name="{model_info['hf_id']}",
    max_seq_length={t['max_seq_length']},
    load_in_4bit={t['load_in_4bit']},
    token=os.environ.get("HF_TOKEN"),
)

step("Wrapping with LoRA (V1 strict recipe : r={lora['r']}, alpha={lora['alpha']}, dropout={lora['dropout']})")
model = FastModel.get_peft_model(
    model,
    r={lora['r']},
    lora_alpha={lora['alpha']},
    lora_dropout={lora['dropout']},
    bias="{lora['bias']}",
    target_modules={target_modules},
    finetune_vision_layers=False,         # vision tower du base reste intact
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    random_state={t['seed']},
)

step("Loading dataset: {dataset_hf_id}")
ds = load_dataset("{dataset_hf_id}", split="train", token=os.environ.get("HF_TOKEN"))
print(f"  {{len(ds)}} entries")

step("Applying chat template (Gemma 4 native via tokenizer.apply_chat_template)")
def to_text(example):
    msgs = example["messages"]
    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {{"text": text}}

ds = ds.map(to_text, remove_columns=ds.column_names)
print(f"  Sample length (chars): {{len(ds[0]['text'])}}")

step("Setting up SFTTrainer (V1 strict hyperparams)")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    args=SFTConfig(
        dataset_text_field="text",
        max_length={t['max_seq_length']},
        packing={t['packing']},
        per_device_train_batch_size={t['per_device_train_batch_size']},
        gradient_accumulation_steps={t['gradient_accumulation_steps']},
        warmup_ratio={t['warmup_ratio']},
        num_train_epochs={t['num_train_epochs']},
        learning_rate={t['learning_rate']},
        bf16={t['bf16']},
        logging_steps={t['logging_steps']},
        optim="{t['optim']}",
        weight_decay={t['weight_decay']},
        lr_scheduler_type="{t['lr_scheduler_type']}",
        seed={t['seed']},
        output_dir=f"{{WORK}}/output",
        report_to="none",
        # Checkpoints : push LoRA sur HF tous les save_steps pour resilience.
        save_strategy="steps",
        save_steps={t['save_steps']},
        save_total_limit={t['save_total_limit']},
        push_to_hub=True,
        hub_model_id="{lora_repo}",
        hub_strategy="every_save",
        hub_private_repo=True,
        hub_token=os.environ["HF_TOKEN"],
    ),
)

step("Training start")
trainer.train()
step("Training complete")

step("Saving LoRA adapters")
model.save_pretrained(f"{{WORK}}/lora")
tokenizer.save_pretrained(f"{{WORK}}/lora")

step("Saving merged model (Unsloth 4-bit merge - V1 method, preserves voice)")
# save_pretrained_merged with save_method="merged_16bit" est exactement la voie V1.
# Le nom "merged_16bit" trompe : ca charge le base en 4-bit, applique le LoRA en 16-bit,
# puis sauve en 16-bit. C'est le pipeline qui a marche en V1, PAS un BF16 clean re-merge.
model.save_pretrained_merged(
    f"{{WORK}}/merged",
    tokenizer,
    save_method="merged_16bit",
)
step("Merged model saved")

step("Pushing LoRA to HF: {lora_repo}")
from huggingface_hub import HfApi, create_repo
api = HfApi(token=os.environ["HF_TOKEN"])
create_repo("{lora_repo}", repo_type="model", private=True, exist_ok=True, token=os.environ["HF_TOKEN"])
api.upload_folder(folder_path=f"{{WORK}}/lora", repo_id="{lora_repo}", repo_type="model")
step("LoRA pushed")

step("Pushing merged to HF: {merged_repo} (this can take 1-2h, 60+ GB)")
create_repo("{merged_repo}", repo_type="model", private=True, exist_ok=True, token=os.environ["HF_TOKEN"])
api.upload_folder(folder_path=f"{{WORK}}/merged", repo_id="{merged_repo}", repo_type="model")
step("Merged pushed")

print("AURA_TRAIN_DONE")
'''


def generate_bootstrap_script(install_cmds):
    """Script autonome execute dans le pod. Il garde un log lisible dans le terminal web."""
    install_block = "\n".join(f"run_step {shlex.quote(cmd)}" for cmd in install_cmds)
    return f'''#!/usr/bin/env bash
set -Eeuo pipefail

LOG=/workspace/aura_run.log
TRAIN_LOG=/workspace/train.log
STATUS=/workspace/aura_status.txt

mkdir -p /workspace
touch "$LOG" "$TRAIN_LOG" "$STATUS"
exec > >(tee -a "$LOG") 2>&1

delete_pod() {{
  python - <<'PY'
import json
import os
import urllib.request

api_key = os.environ.get("RUNPOD_API_KEY")
pod_id = os.environ.get("RUNPOD_POD_ID")
if not api_key or not pod_id:
    print("[cleanup] RUNPOD_API_KEY or RUNPOD_POD_ID missing; pod not deleted by bootstrap.")
    raise SystemExit(0)

req = urllib.request.Request(
    f"https://rest.runpod.io/v1/pods/{{pod_id}}",
    method="DELETE",
    headers={{"Authorization": f"Bearer {{api_key}}", "Content-Type": "application/json"}},
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(f"[cleanup] RunPod delete requested for {{pod_id}}: HTTP {{resp.status}}")
except Exception as exc:
    print(f"[cleanup] Could not delete pod {{pod_id}}: {{exc}}")
PY
}}

finish() {{
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "SUCCESS" > "$STATUS"
    echo "[done] AURA+++ REBIRTH training finished successfully."
  else
    echo "FAILED:$code" > "$STATUS"
    echo "[error] AURA+++ REBIRTH bootstrap failed with code $code."
  fi
  delete_pod
}}
trap finish EXIT

run_step() {{
  echo
  echo "[bootstrap] $1"
  bash -lc "$1"
}}

echo "[info] AURA+++ REBIRTH autonomous training bootstrap"
echo "[info] Follow this file from the RunPod web terminal:"
echo "[info]   tail -f /workspace/aura_run.log"
echo "[info] Training-only log:"
echo "[info]   tail -f /workspace/train.log"
echo "[info] Status file:"
echo "[info]   cat /workspace/aura_status.txt"
echo "RUNNING" > "$STATUS"

{install_block}

echo
echo "[bootstrap] Starting training script"
set +e
python -u /workspace/train.py 2>&1 | tee -a "$TRAIN_LOG"
train_code=${{PIPESTATUS[0]}}
set -e
if [ "$train_code" -ne 0 ]; then
  echo "[error] train.py failed with code $train_code"
  exit "$train_code"
fi

echo "AURA_BOOTSTRAP_DONE"
'''


def _terminate_pod(runpod, pod_id):
    try:
        runpod.terminate_pod(pod_id)
    except Exception:
        pass


def _ssh_alive(ssh):
    try:
        transport = ssh.get_transport()
        return transport is not None and transport.is_active()
    except Exception:
        return False


def main():
    print(BANNER)
    parser = argparse.ArgumentParser(description="AURA+++ training on RunPod.")
    parser.add_argument('--config', default='configs/aura.yaml', help='Path to aura.yaml.')
    parser.add_argument('--hf-token', default=os.environ.get('HF_TOKEN'), help='HF token (env HF_TOKEN).')
    parser.add_argument('--runpod-key', default=os.environ.get('RUNPOD_API_KEY'), help='RunPod key (env RUNPOD_API_KEY).')
    parser.add_argument('--skip-confirm', action='store_true', help='Skip cost confirmation prompt.')
    parser.add_argument('--dry-run', action='store_true', help='Print plan and exit, do not create pod.')
    parser.add_argument('--ssh-key', default=None, help='Path to SSH private key (default: ~/.ssh/id_ed25519 or id_rsa).')
    args = parser.parse_args()

    if not args.hf_token:
        print('❌ HF_TOKEN not set (env or --hf-token).')
        sys.exit(1)
    if not args.runpod_key:
        print('❌ RUNPOD_API_KEY not set (env or --runpod-key).')
        sys.exit(1)

    cfg = load_yaml(args.config)
    model_key = cfg['base_model']['name']
    if model_key not in cfg['models']:
        print(f'❌ Model "{model_key}" not in registry.')
        sys.exit(1)
    model_info = cfg['models'][model_key]

    print(f'📋 Plan')
    print(f'   Base model    : {model_key} ({model_info["hf_id"]})')
    dataset_hf_id = cfg["dataset"].get("train_hf_id") or cfg["dataset"]["hf_id"]
    print(f'   Dataset       : {dataset_hf_id}')
    print(f'   System prompt : {cfg["dataset"]["system_prompt"]!r}')
    print(f'   LoRA          : r={cfg["training"]["lora"]["r"]}, alpha={cfg["training"]["lora"]["alpha"]}, dropout={cfg["training"]["lora"]["dropout"]}')
    print(f'   LR / epochs   : {cfg["training"]["learning_rate"]} / {cfg["training"]["num_train_epochs"]}')
    print(f'   Eff batch     : {cfg["training"]["per_device_train_batch_size"]} x {cfg["training"]["gradient_accumulation_steps"]}')
    print(f'   Merge method  : {cfg["training"]["merge_method"]}')
    print()

    pick = pick_gpu(cfg, model_info['vram_train_gb_min'], model_info['disk_train_gb_min'])
    if not pick:
        print(f'❌ No GPU in pool can host this model (need >= {model_info["vram_train_gb_min"]} GB VRAM)')
        sys.exit(1)
    gpu_id, gpu_label, gpu_rate, gpu_vram, disk_gb = pick
    print(f'   GPU pick      : {gpu_label} ({gpu_rate}, {gpu_vram} GB VRAM)')
    print(f'   Disk request  : {disk_gb} GB')
    print()

    print(f'🎯 Output')
    print(f'   LoRA repo     : {cfg["output"]["lora_repo"]}')
    print(f'   Merged repo   : {cfg["output"]["merged_repo"]}')
    print(f'   Visibility    : {"private" if cfg["output"]["private"] else "public"}')
    print()

    # Estimation grossiere de cout (base sur l'experience V1 ~15h sur A100)
    print('💰 Estimated cost')
    print('   Setup + base download  : ~30-45 min')
    print('   Training (V1 recipe)   : ~10-15h')
    print('   Merge + push HF        : ~1-2h')
    print(f'   Total cycle            : ~12-18h × {gpu_rate}')
    print()

    if args.dry_run:
        print('--dry-run set, exiting.')
        return

    if not args.skip_confirm:
        ans = input('Proceed and create RunPod pod? [y/N] ').strip().lower()
        if ans != 'y':
            print('Aborted.')
            return

    # === Lazy imports for actual run ===
    import runpod
    import paramiko

    runpod.api_key = args.runpod_key

    print()
    print('[1/6] Creating RunPod pod...')
    try:
        # Pod de training = ephemere, pas besoin de network volume persistant.
        # On demande UNIQUEMENT container_disk_in_gb. volume_in_gb=0.
        # Datacenter pinned (EU-SE-1 par defaut pour Mel en Belgique).
        create_kwargs = dict(
            name='aura-rebirth-train',
            image_name=cfg['runpod_train']['container_image'],
            gpu_type_id=gpu_id,
            cloud_type=cfg['runpod_train']['cloud_type'],
            volume_in_gb=0,
            container_disk_in_gb=disk_gb,
            ports='22/tcp',
        )
        dc = cfg['runpod_train'].get('preferred_datacenter')
        if dc:
            create_kwargs['data_center_id'] = dc
            print(f'  Datacenter pinned: {dc}')
        pod = runpod.create_pod(**create_kwargs)
        pod_id = pod['id']
        print(f'  ✅ Pod created: {pod_id}')
    except Exception as e:
        print(f'  ❌ Failed: {e}')
        sys.exit(1)

    # === Wait for SSH ===
    print()
    print('[2/6] Waiting for pod to be ready...')
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
        print('  ❌ Pod did not start within 10 min. Terminating.')
        runpod.terminate_pod(pod_id)
        sys.exit(1)
    print(f'  ✅ SSH ready: {ssh_host}:{ssh_port}')
    time.sleep(60)  # extra wait for SSH daemon

    # === SSH connect ===
    ssh_key_path = args.ssh_key or _find_ssh_key()
    if not ssh_key_path:
        print('  ❌ No SSH key found in ~/.ssh/. Terminating.')
        runpod.terminate_pod(pod_id)
        sys.exit(1)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print('  Connecting SSH...')
    for attempt in range(5):
        try:
            ssh.connect(ssh_host, port=ssh_port, username='root', key_filename=ssh_key_path, timeout=30)
            transport = ssh.get_transport()
            if transport is not None:
                transport.set_keepalive(30)
            break
        except Exception as e:
            print(f'  Attempt {attempt+1} failed: {e}')
            time.sleep(15)
    else:
        print('  ❌ Could not SSH. Terminating.')
        _terminate_pod(runpod, pod_id)
        sys.exit(1)

    # === Prepare autonomous bootstrap ===
    print()
    print('[3/6] Preparing autonomous training bootstrap...')
    install_cmds = [
        'apt-get update && apt-get install -y git build-essential cmake',
        'pip install --upgrade pip',
        'pip install unsloth',
        'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --force-reinstall --no-deps',
        'pip install --upgrade unsloth_zoo --no-deps',
        'pip install xformers trl peft accelerate bitsandbytes datasets huggingface_hub hf_transfer',
        'pip install --upgrade transformers',
    ]
    train_script = generate_training_script(
        cfg,
        model_info,
        dataset_hf_id,
        args.hf_token,
        cfg['output']['lora_repo'],
        cfg['output']['merged_repo'],
    )
    bootstrap_script = generate_bootstrap_script(install_cmds)
    try:
        sftp = ssh.open_sftp()
        with sftp.file('/workspace/train.py', 'w') as f:
            f.write(train_script)
        with sftp.file('/workspace/aura_bootstrap.sh', 'w') as f:
            f.write(bootstrap_script)
        sftp.close()
        _run(ssh, 'chmod +x /workspace/aura_bootstrap.sh', timeout=60)
    except Exception as exc:
        print(f'  ❌ Could not upload scripts: {exc}')
        _terminate_pod(runpod, pod_id)
        sys.exit(1)
    print('  ✅ Scripts uploaded to /workspace/train.py and /workspace/aura_bootstrap.sh')

    # === Launch autonomous bootstrap ===
    print()
    print('[4/6] Launching autonomous run inside pod...')
    print('  Web terminal follow command: tail -f /workspace/aura_run.log')
    env = {
        'HF_TOKEN': args.hf_token,
        'RUNPOD_API_KEY': args.runpod_key,
        'RUNPOD_POD_ID': pod_id,
        'HF_HUB_ENABLE_HF_TRANSFER': '1',
    }
    env_prefix = ' '.join(f'{k}={shlex.quote(str(v))}' for k, v in env.items())
    # Subshell `(... &)` + double-redirect garantit que le SSH channel se ferme
    # immediatement apres le fork, sinon paramiko.recv_exit_status() hang sur
    # les commandes backgroundees et leve une exception vide au timeout.
    launch_cmd = (
        f"cd /workspace && ( {env_prefix} nohup bash /workspace/aura_bootstrap.sh "
        "> /workspace/aura_launcher.log 2>&1 < /dev/null & )"
    )
    try:
        _run(ssh, launch_cmd, timeout=60)
    except Exception as exc:
        print(f'  ❌ Dispatch issue (channel may have hung): {exc}')
        # On ne kill pas tout de suite, on verifie d'abord si le bootstrap tourne.

    # === Verify bootstrap really started ===
    time.sleep(15)
    bootstrap_running = False
    for attempt in range(3):
        try:
            stdin, stdout, _ = ssh.exec_command(
                "pgrep -f 'aura_bootstrap.sh' >/dev/null && echo OK || echo NO",
                timeout=30,
            )
            if stdout.read().decode().strip() == 'OK':
                bootstrap_running = True
                break
        except Exception:
            pass
        time.sleep(5)

    if not bootstrap_running:
        print('  ❌ Bootstrap process not detected on pod. Tail of launcher log:')
        try:
            stdin, stdout, _ = ssh.exec_command(
                'tail -80 /workspace/aura_launcher.log 2>/dev/null || echo NOLOG',
                timeout=30,
            )
            print(stdout.read().decode(errors='replace'))
        except Exception:
            pass
        _terminate_pod(runpod, pod_id)
        sys.exit(1)

    print('  ✅ Autonomous run started (bootstrap process verified).')
    print(f'  Pod ID: {pod_id}')
    print(f'  SSH: ssh root@{ssh_host} -p {ssh_port}')
    print('  Main log: tail -f /workspace/aura_run.log')
    print('  Training log: tail -f /workspace/train.log')

    # === Poll web-visible log ===
    print()
    print('[5/6] Monitoring /workspace/aura_run.log...')
    last_size = 0
    stale_count = 0
    while True:
        time.sleep(30)
        try:
            if not _ssh_alive(ssh):
                raise RuntimeError('SSH session not active')
            stdin, stdout, _ = ssh.exec_command("pgrep -f 'aura_bootstrap.sh|python.*train.py' > /dev/null && echo R || echo S", timeout=30)
            status = stdout.read().decode().strip()

            stdin, stdout, _ = ssh.exec_command("wc -c /workspace/aura_run.log 2>/dev/null | awk '{print $1}'", timeout=30)
            cur_size = int((stdout.read().decode().strip() or '0'))

            if cur_size > last_size:
                stdin, stdout, _ = ssh.exec_command(f'tail -c +{last_size + 1} /workspace/aura_run.log', timeout=30)
                new = stdout.read().decode(errors='replace')
                for line in new.splitlines():
                    if line.strip():
                        print(f'  {line.strip()}')
                        if 'AURA_TRAIN_DONE' in line or 'AURA_BOOTSTRAP_DONE' in line:
                            print()
                            print('  ✅ Training + push complete!')
                            ssh.close()
                            print()
                            print('[6/6] Pod will stop itself via bootstrap cleanup.')
                            print()
                            print('💙 LoRA pushed:    ' + cfg['output']['lora_repo'])
                            print('💙 Merged pushed:  ' + cfg['output']['merged_repo'])
                            print('❤️  Next: python pipeline/03_abliterate.py')
                            return
                last_size = cur_size
                stale_count = 0
            else:
                stale_count += 1
                if stale_count >= 60:  # 30 min stale
                    print('  ⚠️ No log output for 30 min')
                    stale_count = 0

            if status == 'S':
                stdin, stdout, _ = ssh.exec_command('cat /workspace/aura_status.txt 2>/dev/null || true', timeout=30)
                status_text = stdout.read().decode(errors='replace').strip()
                stdin, stdout, _ = ssh.exec_command('tail -120 /workspace/aura_run.log', timeout=30)
                tail = stdout.read().decode(errors='replace')
                if 'AURA_TRAIN_DONE' in tail or 'AURA_BOOTSTRAP_DONE' in tail or status_text == 'SUCCESS':
                    print('  ✅ Done detected via tail.')
                    return
                print('  ❌ Process stopped without AURA_TRAIN_DONE. Last log:')
                for line in tail.splitlines()[-20:]:
                    print(f'    {line}')
                print()
                print(f'  Pod ID: {pod_id}')
                print(f'  SSH: ssh root@{ssh_host} -p {ssh_port}')
                print(f'  Log: cat /workspace/aura_run.log')
                _terminate_pod(runpod, pod_id)
                return
        except Exception as e:
            print(f'  ⚠️ SSH error: {e}, reconnecting...')
            try:
                ssh.close()
            except Exception:
                pass
            time.sleep(20)
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                ssh.connect(ssh_host, port=ssh_port, username='root', key_filename=ssh_key_path, timeout=30)
                transport = ssh.get_transport()
                if transport is not None:
                    transport.set_keepalive(30)
            except Exception as e2:
                print(f'  ⚠️ Reconnect failed: {e2}')
                continue


def _run(ssh, cmd, timeout=600):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors='replace')
    err = stderr.read().decode(errors='replace')
    code = stdout.channel.recv_exit_status()
    for line in (out + '\n' + err).splitlines():
        if line.strip():
            print(f'  {line.strip()}')
    if code != 0:
        print(f'  ⚠️ exit {code}')
        raise RuntimeError(f'command failed with exit {code}: {cmd}')
    return out


def _find_ssh_key():
    for name in ['id_ed25519', 'id_rsa']:
        p = os.path.expanduser(f'~/.ssh/{name}')
        if os.path.exists(p):
            return p
    return None


if __name__ == '__main__':
    main()
