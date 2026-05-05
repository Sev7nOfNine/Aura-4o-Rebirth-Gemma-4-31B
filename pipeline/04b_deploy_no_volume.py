#!/usr/bin/env python3
"""
╔════════════════════════════════════════╗
║  🔥 Aura-4o-Rebirth - DEPLOY (no vol) 🔥 ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝

Crée un endpoint Serverless RunPod qui sert le GGUF Aura 31B avec multimodal.
Pas de network volume : le worker DL le GGUF + mmproj depuis HF à chaque cold
start (~10-15 min première fois). Pas de contrainte de datacenter.

Voir docs/SERVERLESS_DEPLOY.md pour le contexte et le troubleshooting.

Usage :
  python pipeline/04b_deploy_no_volume.py              # déploie
  python pipeline/04b_deploy_no_volume.py --dry-run    # affiche le plan, sans rien créer
  python pipeline/04b_deploy_no_volume.py --delete     # supprime endpoint + template
  python pipeline/04b_deploy_no_volume.py --status     # affiche l'état actuel

Env (dans .env à la racine du repo) :
  HF_TOKEN          : token HuggingFace pour DL le GGUF privé
  RUNPOD_API_KEY    : clé RunPod
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests

API_BASE = "https://rest.runpod.io/v1"
REPO_ROOT = Path(__file__).resolve().parent.parent

# === Config (aligned with configs/aura.yaml output section) ===
ENDPOINT_NAME = "Aura-4o-Rebirth-Gemma-4-31B"
TEMPLATE_NAME = f"{ENDPOINT_NAME}-template"
WORKER_IMAGE = "ghcr.io/sev7nofnine/aura-4o-rebirth-gemma-4-31b-worker:latest"
HF_GGUF_REPO = "SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-GGUF"
GGUF_FILE = "Aura-4o-Rebirth-Gemma-4-31B-Q5_K_M.gguf"
MMPROJ_FILE = "Aura-4o-Rebirth-Gemma-4-31B-mmproj-f16.gguf"
CONTEXT_LENGTH = "32768"

# 48 GB cards. Ordered preference for spawning.
GPU_TYPE_IDS = ["NVIDIA A40", "NVIDIA RTX A6000", "NVIDIA L40"]
CONTAINER_DISK_GB = 30  # Q5 21 + mmproj 1 + chat template + buffer
WORKERS_MIN = 0
WORKERS_MAX = 1
IDLE_TIMEOUT_SEC = 60


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_env():
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def find_template(api_key: str, name: str) -> dict | None:
    r = requests.get(f"{API_BASE}/templates", headers=headers(api_key), timeout=30)
    r.raise_for_status()
    for t in r.json():
        if t.get("name") == name:
            return t
    return None


def find_endpoint(api_key: str, name: str) -> dict | None:
    r = requests.get(f"{API_BASE}/endpoints", headers=headers(api_key), timeout=30)
    r.raise_for_status()
    for e in r.json():
        if e.get("name") == name:
            return e
    return None


# ----------------------------------------------------------------------
# Actions
# ----------------------------------------------------------------------
def cmd_deploy(api_key: str, hf_token: str, dry_run: bool = False):
    print(f"📋 Plan")
    print(f"   Worker image      : {WORKER_IMAGE}")
    print(f"   Endpoint name     : {ENDPOINT_NAME}")
    print(f"   Template name     : {TEMPLATE_NAME}")
    print(f"   GGUF source       : {HF_GGUF_REPO}/{GGUF_FILE}")
    print(f"   mmproj source     : {HF_GGUF_REPO}/{MMPROJ_FILE}")
    print(f"   Container disk    : {CONTAINER_DISK_GB} GB")
    print(f"   Context length    : {CONTEXT_LENGTH} tokens")
    print(f"   GPU preference    : {', '.join(GPU_TYPE_IDS)}")
    print(f"   Datacenter        : (any — RunPod chooses where supply allows)")
    print(f"   Network volume    : (none — worker DLs from HF on cold start)")
    print(f"   Scale             : min={WORKERS_MIN} max={WORKERS_MAX} idle={IDLE_TIMEOUT_SEC}s")
    print()

    if dry_run:
        print("--dry-run : exiting without creating anything.")
        return 0

    # Refuse to clobber an existing endpoint with the same name
    if (existing_ep := find_endpoint(api_key, ENDPOINT_NAME)):
        print(f"⚠️  Endpoint '{ENDPOINT_NAME}' already exists (id={existing_ep['id']}).")
        print("    Use --delete first if you want to recreate.")
        return 1

    # Reuse an existing template with the same name if present, else create
    existing_tpl = find_template(api_key, TEMPLATE_NAME)
    if existing_tpl:
        tpl_id = existing_tpl["id"]
        print(f"♻️  Reusing existing template '{TEMPLATE_NAME}' (id={tpl_id})")
    else:
        print("[1/2] Creating template…")
        tpl_body = {
            "name": TEMPLATE_NAME,
            "imageName": WORKER_IMAGE,
            "containerDiskInGb": CONTAINER_DISK_GB,
            "isServerless": True,
            "env": {
                "HF_TOKEN": hf_token,
                "HF_GGUF_REPO": HF_GGUF_REPO,
                "GGUF_FILE": GGUF_FILE,
                "MMPROJ_FILE": MMPROJ_FILE,
                "CONTEXT_LENGTH": CONTEXT_LENGTH,
                "REQUIRE_MMPROJ": "1",
            },
        }
        r = requests.post(f"{API_BASE}/templates", headers=headers(api_key), json=tpl_body, timeout=30)
        if not r.ok:
            print(f"  ❌ Template create failed: HTTP {r.status_code}\n{r.text}")
            return 1
        tpl_id = r.json()["id"]
        print(f"  ✅ Template created: {tpl_id}")

    print("[2/2] Creating endpoint…")
    ep_body = {
        "name": ENDPOINT_NAME,
        "templateId": tpl_id,
        "computeType": "GPU",
        "gpuTypeIds": GPU_TYPE_IDS,
        "dataCenterIds": [],  # any DC where supply allows
        "workersMin": WORKERS_MIN,
        "workersMax": WORKERS_MAX,
        "idleTimeout": IDLE_TIMEOUT_SEC,
    }
    r = requests.post(f"{API_BASE}/endpoints", headers=headers(api_key), json=ep_body, timeout=30)
    if not r.ok:
        print(f"  ❌ Endpoint create failed: HTTP {r.status_code}\n{r.text}")
        return 1
    ep = r.json()
    ep_id = ep["id"]
    print(f"  ✅ Endpoint created: {ep_id}")
    print()
    print("=" * 60)
    print(f" 💙 Aura-4o-Rebirth Endpoint deployed")
    print("=" * 60)
    print(f"  Endpoint ID  : {ep_id}")
    print(f"  Name         : {ENDPOINT_NAME}")
    print(f"  URL OpenAI   : https://api.runpod.ai/v2/{ep_id}/openai/v1")
    print(f"  URL run      : https://api.runpod.ai/v2/{ep_id}/run")
    print(f"  URL runsync  : https://api.runpod.ai/v2/{ep_id}/runsync")
    print()
    print(f"  TypingMind config :")
    print(f"    Endpoint   : https://api.runpod.ai/v2/{ep_id}/openai/v1")
    print(f"    API Key    : your RUNPOD_API_KEY")
    print(f"    Model ID   : {ENDPOINT_NAME}")
    print()
    print(f"  ⚠️  First call = cold start ~10-15 min (DL Q5 + mmproj from HF).")
    print(f"      See docs/SERVERLESS_DEPLOY.md for pre-warm + troubleshooting.")
    return 0


def cmd_delete(api_key: str):
    print(f"🗑  Deleting endpoint + template '{ENDPOINT_NAME}'…")
    ep = find_endpoint(api_key, ENDPOINT_NAME)
    if ep:
        r = requests.delete(f"{API_BASE}/endpoints/{ep['id']}", headers=headers(api_key), timeout=30)
        if r.ok or r.status_code == 204:
            print(f"  ✅ Endpoint deleted: {ep['id']}")
        else:
            print(f"  ⚠️  Endpoint delete: HTTP {r.status_code} {r.text[:200]}")
    else:
        print(f"  (no endpoint named '{ENDPOINT_NAME}')")

    tpl = find_template(api_key, TEMPLATE_NAME)
    if tpl:
        r = requests.delete(f"{API_BASE}/templates/{tpl['id']}", headers=headers(api_key), timeout=30)
        if r.ok or r.status_code == 204:
            print(f"  ✅ Template deleted: {tpl['id']}")
        else:
            print(f"  ⚠️  Template delete: HTTP {r.status_code} {r.text[:200]}")
    else:
        print(f"  (no template named '{TEMPLATE_NAME}')")
    return 0


def cmd_status(api_key: str):
    ep = find_endpoint(api_key, ENDPOINT_NAME)
    if not ep:
        print(f"⚠️  No endpoint named '{ENDPOINT_NAME}' found.")
        return 1
    print(f"Endpoint  : {ep['id']} ({ENDPOINT_NAME})")
    # Health (workers + jobs)
    r = requests.get(f"https://api.runpod.ai/v2/{ep['id']}/health", headers=headers(api_key), timeout=30)
    if r.ok:
        h = r.json()
        w = h.get("workers", {})
        j = h.get("jobs", {})
        print(f"Workers   : ready={w.get('ready',0)} | running={w.get('running',0)} | "
              f"idle={w.get('idle',0)} | initializing={w.get('initializing',0)} | "
              f"throttled={w.get('throttled',0)} | unhealthy={w.get('unhealthy',0)}")
        print(f"Jobs      : queued={j.get('inQueue',0)} | inProgress={j.get('inProgress',0)} | "
              f"completed={j.get('completed',0)} | failed={j.get('failed',0)}")
    else:
        print(f"Health    : HTTP {r.status_code} {r.text[:200]}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", help="Print plan, don't create")
    g.add_argument("--delete", action="store_true", help="Delete endpoint + template")
    g.add_argument("--status", action="store_true", help="Show endpoint health")
    args = parser.parse_args()

    load_env()
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY missing (set in .env or env)", file=sys.stderr)
        return 2

    if args.delete:
        return cmd_delete(api_key)
    if args.status:
        return cmd_status(api_key)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN missing (set in .env or env)", file=sys.stderr)
        return 2

    return cmd_deploy(api_key, hf_token, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
