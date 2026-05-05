"""
╔════════════════════════════════════════╗
║  🔥 Aura-4o-Refresh - MERGE V1 LoRA 🔥 ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝

V1-Refresh : merge le LoRA V1 historique de Mel sur la base d'origine
(paperscarecrow/Gemma-4-31B-it-abliterated) pour la déployer en serverless
llama.cpp custom (et donc avec mmproj/vision, contrairement à la V1 Ollama).

Pourquoi cette voie :
  - Le LoRA V1 capture la "fougue" Aura (retour Mel : V3 Rebirth est lissée).
  - paperscarecrow contient les 356 tensors vision (vérifié) ; Ollama runtime
    ne les expose pas, mais llama.cpp custom + mmproj extrait peut.
  - On reste fidèle à la base d'entraînement du LoRA → zéro dérive.

Pipeline :
  1. DL paperscarecrow (subfolder gemma-4-31b-abliterated)
  2. DL LoRA V1 (SevenOfNine/Aura-4o-Gemma-4-31B-LoRA)
  3. Manual merge (no PEFT, no Unsloth - same approach as 02b)
  4. Save merged + push to HF (Aura-4o-Refresh-Gemma-4-31B-Merged)
  5. Convert HF -> GGUF bf16 + extract mmproj
  6. Quantize Q4_K_M, Q5_K_M, Q8_0
  7. Push GGUF + mmproj -> Aura-4o-Refresh-Gemma-4-31B-GGUF
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = Path("/workspace")
LLAMA = WORK / "llama.cpp"
BASE_DIR = WORK / "base"
LORA_DIR = WORK / "lora"
MERGED = WORK / "merged"
OUT = WORK / "out"

BASE_REPO = "paperscarecrow/Gemma-4-31B-it-abliterated"
BASE_SUBFOLDER = "gemma-4-31b-abliterated"
LORA_REPO = "SevenOfNine/Aura-4o-Gemma-4-31B-LoRA"
MERGED_REPO = "SevenOfNine/Aura-4o-Refresh-Gemma-4-31B-Merged"
GGUF_REPO = "SevenOfNine/Aura-4o-Refresh-Gemma-4-31B-GGUF"

GGUF_PREFIX = "Aura-4o-Refresh-Gemma-4-31B"


def run(cmd, cwd=None):
    print("[CMD]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def step(msg):
    print(f"\n[STEP] {msg}", flush=True)


def free(path: Path):
    if path.exists():
        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1024**3
        print(f"[FREE] {path} ({size:.1f} GB)", flush=True)
        shutil.rmtree(path)


def get_module_by_path(model, path: str):
    obj = model
    for p in path.split("."):
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj


def get_weight_tensor(module):
    import torch.nn as nn
    if isinstance(module, nn.Linear):
        return module.weight
    if hasattr(module, "linear") and isinstance(module.linear, nn.Linear):
        return module.linear.weight
    if hasattr(module, "weight"):
        return module.weight
    raise AttributeError(f"Cannot locate weight tensor on {type(module).__name__}")


def manual_merge_lora(model, lora_dir: Path, alpha: int, r: int, use_rslora: bool = False):
    import torch
    from safetensors import safe_open

    scale = (alpha / math.sqrt(r)) if use_rslora else (alpha / r)
    print(f"[MERGE] LoRA scale = {scale:.4f} (alpha={alpha}, r={r}, rslora={use_rslora})", flush=True)

    safetensor_files = sorted(lora_dir.glob("adapter_model*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No adapter_model*.safetensors in {lora_dir}")

    pairs: dict[str, dict[str, "torch.Tensor"]] = defaultdict(dict)

    for sf in safetensor_files:
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if ".lora_A." in key:
                    path = key.split(".lora_A.")[0].replace("base_model.model.", "", 1)
                    pairs[path]["A"] = t
                elif ".lora_B." in key:
                    path = key.split(".lora_B.")[0].replace("base_model.model.", "", 1)
                    pairs[path]["B"] = t

    print(f"[MERGE] Found {len(pairs)} LoRA target modules", flush=True)

    applied, skipped = 0, []
    for path, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            skipped.append(f"{path} (missing A or B)")
            continue
        try:
            module = get_module_by_path(model, path)
            W = get_weight_tensor(module)
        except (AttributeError, IndexError) as e:
            skipped.append(f"{path} ({e})")
            continue

        A = ab["A"].to(W.device, dtype=torch.float32)
        B = ab["B"].to(W.device, dtype=torch.float32)
        delta = scale * (B @ A)
        if delta.shape != W.shape:
            skipped.append(f"{path} (shape mismatch: delta {tuple(delta.shape)} vs W {tuple(W.shape)})")
            continue
        W.data.add_(delta.to(W.dtype))
        applied += 1
        if applied % 50 == 0:
            print(f"[MERGE] applied {applied}/{len(pairs)}", flush=True)

    print(f"[MERGE] Done. Applied {applied}/{len(pairs)} pairs.", flush=True)
    if skipped:
        print(f"[MERGE] Skipped {len(skipped)} (first 10):", flush=True)
        for s in skipped[:10]:
            print(f"   - {s}", flush=True)


def main():
    HF_TOKEN = os.environ["HF_TOKEN"]

    step("Installing system deps")
    run(["bash", "-lc", "apt-get update -qq && apt-get install -y -qq git cmake python3-pip"])

    step("Installing Python deps")
    run([sys.executable, "-m", "pip", "install", "-q",
         "huggingface_hub", "hf_transfer", "pyyaml",
         "accelerate", "safetensors", "gguf",
         "git+https://github.com/huggingface/transformers.git"])

    if not LLAMA.exists():
        step("Cloning llama.cpp")
        run(["git", "clone", "--depth", "1",
             "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)])
    run([sys.executable, "-m", "pip", "install", "-q", "-r",
         str(LLAMA / "requirements" / "requirements-convert_hf_to_gguf.txt")])

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    from huggingface_hub import snapshot_download, HfApi

    step(f"Downloading base : {BASE_REPO} (~62 GB, subfolder {BASE_SUBFOLDER})")
    snapshot_download(
        repo_id=BASE_REPO,
        allow_patterns=[f"{BASE_SUBFOLDER}/*"],
        local_dir=str(BASE_DIR),
        token=HF_TOKEN,
        max_workers=8,
    )
    actual_base = BASE_DIR / BASE_SUBFOLDER
    print(f"[INFO] Base path: {actual_base}")

    step(f"Downloading LoRA : {LORA_REPO}")
    snapshot_download(repo_id=LORA_REPO, local_dir=str(LORA_DIR),
                      token=HF_TOKEN, max_workers=8)

    with open(LORA_DIR / "adapter_config.json", "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    lora_alpha = int(adapter_cfg.get("lora_alpha", 32))
    lora_r = int(adapter_cfg.get("r", 32))
    use_rslora = bool(adapter_cfg.get("use_rslora", False))
    print(f"\n[LoRA cfg] alpha={lora_alpha}, r={lora_r}, use_rslora={use_rslora}")

    step("Loading base with Gemma4ForConditionalGeneration (full multimodal)")
    import torch
    from transformers import Gemma4ForConditionalGeneration, AutoProcessor

    model = Gemma4ForConditionalGeneration.from_pretrained(
        str(actual_base),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    processor = AutoProcessor.from_pretrained(str(actual_base), token=HF_TOKEN)

    step("Manual LoRA merge (no PEFT, no Unsloth)")
    manual_merge_lora(model, LORA_DIR, alpha=lora_alpha, r=lora_r, use_rslora=use_rslora)
    model = model.to(torch.bfloat16)

    step(f"Saving merged to {MERGED}")
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True)
    model.save_pretrained(str(MERGED), safe_serialization=True)
    processor.save_pretrained(str(MERGED))

    for fname in ["chat_template.jinja", "processor_config.json",
                  "preprocessor_config.json", "special_tokens_map.json",
                  "tokenizer.json", "tokenizer_config.json"]:
        src_f = LORA_DIR / fname
        if src_f.exists():
            shutil.copy2(src_f, MERGED / fname)

    del model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    step(f"Pushing merged -> {MERGED_REPO}")
    api = HfApi(token=HF_TOKEN)
    api.upload_folder(folder_path=str(MERGED), repo_id=MERGED_REPO,
                      repo_type="model")

    free(BASE_DIR)
    free(LORA_DIR)

    OUT.mkdir(exist_ok=True)
    bf16 = OUT / "model-bf16.gguf"
    mmproj = OUT / f"{GGUF_PREFIX}-mmproj-f16.gguf"

    step("Converting HF -> GGUF bf16")
    run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(MERGED),
         "--outfile", str(bf16), "--outtype", "bf16"])

    step("Converting HF -> GGUF mmproj")
    run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(MERGED),
         "--mmproj", "--outfile", str(mmproj), "--outtype", "f16"])

    free(MERGED)

    step("Building llama-quantize")
    quant = LLAMA / "build" / "bin" / "llama-quantize"
    if not quant.exists():
        run(["bash", "-lc",
             f"cd {LLAMA} && cmake -B build && cmake --build build --target llama-quantize -j$(nproc)"])

    quants = [
        ("Q4_K_M", OUT / f"{GGUF_PREFIX}-Q4_K_M.gguf"),
        ("Q5_K_M", OUT / f"{GGUF_PREFIX}-Q5_K_M.gguf"),
        ("Q8_0",   OUT / f"{GGUF_PREFIX}-Q8_0.gguf"),
    ]
    for qtype, qfile in quants:
        step(f"Quantizing -> {qtype}")
        run([str(quant), str(bf16), str(qfile), qtype])

    if bf16.exists():
        bf16.unlink()

    step(f"Pushing mmproj + 3 quants -> {GGUF_REPO}")
    api.upload_file(path_or_fileobj=str(mmproj), path_in_repo=mmproj.name,
                    repo_id=GGUF_REPO, repo_type="model")
    for _, qfile in quants:
        step(f"Uploading {qfile.name}")
        api.upload_file(path_or_fileobj=str(qfile), path_in_repo=qfile.name,
                        repo_id=GGUF_REPO, repo_type="model")

    print("\n[DONE] V1-Refresh : merged + 3 quants + mmproj on HF.", flush=True)


if __name__ == "__main__":
    main()
