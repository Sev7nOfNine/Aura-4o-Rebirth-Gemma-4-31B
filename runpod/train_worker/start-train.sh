#!/usr/bin/env bash
# ╔════════════════════════════════════════╗
# ║  🔥 AURA+++ - TRAIN WORKER 🔥         ║
# ║  💙 Talons LED FULL CHARGE            ║
# ║  ❤️ By Mel & Aura                     ║
# ╚════════════════════════════════════════╝
#
# Entrypoint du container train_worker. Lance train.py et auto-delete
# le pod via trap EXIT (succes ou erreur). Plus jamais de pod fantome.

set -Eeuo pipefail

LOG=/workspace/aura_run.log
TRAIN_LOG=/workspace/train.log
STATUS=/workspace/aura_status.txt

mkdir -p /workspace
touch "$LOG" "$TRAIN_LOG" "$STATUS"
exec > >(tee -a "$LOG") 2>&1

# === Required env vars (set sur le pod RunPod par 02_train.py) ===
: "${HF_TOKEN:?HF_TOKEN required (passe au pod via env)}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY required (pour auto-delete)}"
: "${RUNPOD_POD_ID:?RUNPOD_POD_ID required (pour auto-delete)}"

# Optional avec defaults coherents avec configs/aura.yaml
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export AURA_BASE_MODEL="${AURA_BASE_MODEL:-SevenOfNine/Gemma-4-31B-It-Official}"
export AURA_DATASET="${AURA_DATASET:-SevenOfNine/Aura-4o-Rebirth-Dataset}"
export AURA_LORA_REPO="${AURA_LORA_REPO:-SevenOfNine/Aura-4o-Rebirth-LoRA}"
export AURA_MERGED_REPO="${AURA_MERGED_REPO:-SevenOfNine/Aura-4o-Rebirth-Merged}"

delete_pod() {
  # On utilise le SDK Python runpod (GraphQL) au lieu de l'API REST DELETE
  # qui renvoyait 403 Forbidden meme avec une cle valide. Cf. incident #7.
  python - <<'PY'
import os
import sys

api_key = os.environ.get("RUNPOD_API_KEY")
pod_id = os.environ.get("RUNPOD_POD_ID")
if not api_key or not pod_id:
    print("[cleanup] RUNPOD_API_KEY or RUNPOD_POD_ID missing; pod not deleted by entrypoint.")
    sys.exit(0)

try:
    import runpod
    runpod.api_key = api_key
    runpod.terminate_pod(pod_id)
    print(f"[cleanup] runpod.terminate_pod({pod_id}) OK")
except Exception as exc:
    print(f"[cleanup] terminate_pod via SDK failed: {exc}")
    # Fallback : tenter l'API REST direct au cas ou
    try:
        import urllib.request
        req = urllib.request.Request(
            f"https://rest.runpod.io/v1/pods/{pod_id}",
            method="DELETE",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"[cleanup] REST fallback DELETE: HTTP {resp.status}")
    except Exception as exc2:
        print(f"[cleanup] REST fallback also failed: {exc2}")
        print(f"[cleanup] !!! POD {pod_id} N'EST PAS SUPPRIME, intervention manuelle requise !!!")
PY
}

finish() {
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "SUCCESS" > "$STATUS"
    echo "[done] AURA+++ REBIRTH training finished successfully."
  else
    echo "FAILED:$code" > "$STATUS"
    echo "[error] AURA+++ REBIRTH training failed with code $code."
  fi
  delete_pod
}
trap finish EXIT

# Watchdog absolu : si on depasse 24h sans avoir termine, on tue le process
# group complet, ce qui declenche le trap finish -> delete_pod. Filet de
# securite contre un train.py qui hang sans erreur (pas de trap sinon).
(
  sleep 86400
  echo "[watchdog] 24h max atteint - kill du process group pour declencher trap"
  kill -TERM 0
) &
WATCHDOG_PID=$!

echo "[info] AURA+++ REBIRTH train_worker entrypoint"
echo "[info] Watchdog 24h actif (PID $WATCHDOG_PID)"
echo "[info] Image build avec deps Unsloth pre-installees (no pip install live)"
echo "[info] Suivi via terminal web RunPod :"
echo "[info]   tail -f /workspace/aura_run.log     (log complet)"
echo "[info]   tail -f /workspace/train.log        (training only)"
echo "[info]   cat /workspace/aura_status.txt      (statut)"
echo "[info] Base model      : $AURA_BASE_MODEL"
echo "[info] Dataset         : $AURA_DATASET"
echo "[info] LoRA repo       : $AURA_LORA_REPO"
echo "[info] Merged repo     : $AURA_MERGED_REPO"
echo "RUNNING" > "$STATUS"

set +e
python -u /app/train.py 2>&1 | tee -a "$TRAIN_LOG"
train_code=${PIPESTATUS[0]}
set -e

if [ "$train_code" -ne 0 ]; then
  echo "[error] train.py failed with code $train_code"
  exit "$train_code"
fi

echo "AURA_TRAIN_DONE"
