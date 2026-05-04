#!/usr/bin/env python3
"""
╔════════════════════════════════════════╗
║  🔥 Aura-4o-Rebirth - VOLUME POLLER 🔥   ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝

Poll RunPod for network volume creation in a given datacenter, retry until
creation succeeds, then stop.

Why this exists :
  RunPod currently blocks volume creation in EU-SE-1 (data center "not found
  or does not support network volumes"). The V1 endpoint had its volume created
  back when it was supported. We hope RunPod re-enables creation eventually.
  Until then, this script polls in the background and grabs a volume the
  instant the gate opens (or whenever supply allows).

Usage :
  python scripts/poll_create_volume.py
  python scripts/poll_create_volume.py --interval 600 --datacenter EU-SE-1
  python scripts/poll_create_volume.py --size 30 --name MyVolume

Env : RUNPOD_API_KEY (loaded from .env at repo root, or pass --runpod-key)

Stop : Ctrl+C, or it stops itself on success.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

API_BASE = "https://rest.runpod.io/v1"
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env():
    """Load .env file from the repo root if present."""
    env_file = REPO_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def attempt_create(api_key: str, name: str, size: int, datacenter: str):
    """Single creation attempt. Returns (success: bool, payload_or_error)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {"name": name, "size": size, "dataCenterId": datacenter}
    try:
        r = requests.post(f"{API_BASE}/networkvolumes", headers=headers, json=body, timeout=30)
    except requests.RequestException as e:
        return False, f"network error: {e}"
    if r.ok:
        try:
            return True, r.json()
        except json.JSONDecodeError:
            return True, {"raw": r.text}
    snippet = r.text[:200].replace("\n", " ")
    return False, f"HTTP {r.status_code}: {snippet}"


def fmt_duration(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h: return f"{h}h{m:02d}m{s:02d}s"
    if m: return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    load_env()

    parser = argparse.ArgumentParser(
        description="Poll RunPod to create a network volume when the gate opens.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between attempts")
    parser.add_argument("--size", type=int, default=25,
                        help="Volume size in GB")
    parser.add_argument("--datacenter", default="EU-SE-1",
                        help="RunPod datacenter ID (e.g. EU-SE-1, EU-NL-1, EU-CZ-1)")
    parser.add_argument("--name", default="Aura-4o-Rebirth-Gemma-4-31B",
                        help="Volume name (must be unique on your account)")
    parser.add_argument("--runpod-key", default=os.environ.get("RUNPOD_API_KEY"),
                        help="Defaults to $RUNPOD_API_KEY")
    parser.add_argument("--max-attempts", type=int, default=0,
                        help="Stop after N failed attempts (0 = infinite)")
    args = parser.parse_args()

    if not args.runpod_key:
        print("ERROR: RUNPOD_API_KEY missing (env var or --runpod-key)", file=sys.stderr)
        sys.exit(2)

    print("╔════════════════════════════════════════════════════════╗")
    print("║  🔥 Aura-4o-Rebirth Volume Poller 🔥                   ║")
    print("╚════════════════════════════════════════════════════════╝")
    print(f"  Target datacenter : {args.datacenter}")
    print(f"  Volume name       : {args.name}")
    print(f"  Volume size       : {args.size} GB")
    print(f"  Poll interval     : {fmt_duration(args.interval)}")
    print(f"  Max attempts      : {'∞' if args.max_attempts == 0 else args.max_attempts}")
    print(f"  Ctrl+C to stop.")
    print()

    started = time.time()
    attempt = 0
    last_error_seen = None

    try:
        while True:
            attempt += 1
            ts = time.strftime("%H:%M:%S")
            ok, res = attempt_create(args.runpod_key, args.name, args.size, args.datacenter)

            if ok:
                vol_id = res.get("id", "?") if isinstance(res, dict) else "?"
                elapsed = int(time.time() - started)
                print()
                print("╔════════════════════════════════════════════════════════╗")
                print("║  ✅ VOLUME CREATED                                      ║")
                print("╚════════════════════════════════════════════════════════╝")
                print(f"  ID            : {vol_id}")
                print(f"  Datacenter    : {res.get('dataCenterId', '?') if isinstance(res, dict) else '?'}")
                print(f"  Size          : {res.get('size', '?') if isinstance(res, dict) else '?'} GB")
                print(f"  Attempts      : {attempt}")
                print(f"  Elapsed       : {fmt_duration(elapsed)}")
                print()
                print("Full response:")
                print(json.dumps(res, indent=2) if isinstance(res, dict) else res)
                return 0

            # Compress repetitive errors
            if res != last_error_seen:
                print(f"[{ts}] attempt #{attempt} | {res}")
                last_error_seen = res
            else:
                # Same error → just print a dot to show we're alive
                print(f"[{ts}] attempt #{attempt} | (same error)")

            if args.max_attempts and attempt >= args.max_attempts:
                print(f"\nReached max attempts ({args.max_attempts}). Stopping.")
                return 1

            time.sleep(args.interval)

    except KeyboardInterrupt:
        elapsed = int(time.time() - started)
        print(f"\n\n⏹  Stopped by user after {attempt} attempts ({fmt_duration(elapsed)}).")
        return 130


if __name__ == "__main__":
    sys.exit(main())
