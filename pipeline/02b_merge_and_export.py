"""
╔════════════════════════════════════════╗
║  🔥 Aura-4o-Rebirth - MERGE + EXPORT 🔥       ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝

Merge LoRA into base Gemma 4 31B (multimodal preserved) + convert to GGUF
(Q4_K_M + Q5_K_M + Q8_0) + push HF.

Why this script (and not 02_train.py / 03_abliterate.py) :
  - 02_train.py produces the LoRA adapter (already pushed by Mel on 2026-05-03).
  - 03_abliterate.py expects the merged model to already be on HF.
  - The 31B Merged HF repo is empty because the merge step was never completed
    (the V7 attempt used AutoModelForCausalLM which silently dropped vision/audio
    tensors -> 666-tensor text-only artifact, same bug as E4B fixed 2026-05-04).
  - This 02b step bridges the gap : merge LoRA + base into HF Merged repo,
    then produce all GGUF artifacts in one shot (no abliteration).

Why the merge is MANUAL (no PEFT, no Unsloth) :
  - Gemma 4 requires transformers >= 5.5.0.dev0 (Gemma4ForConditionalGeneration).
  - Unsloth 2025.11.1 caps transformers at 4.57.2 -> incompatible with Gemma 4.
  - Vanilla PEFT cannot wrap Gemma4ClippableLinear modules used by Gemma 4 31B
    (per_layer_input_gate, relative_k_proj, etc.) -> ValueError on merge.
  - Solution : compute LoRA deltas (alpha/r * B @ A) and add them directly to
    each target module's weight tensor. Works for any wrapper class.

Pipeline :
  1. Read configs/aura.yaml
  2. DL base model (full multimodal) from HF
  3. DL LoRA adapter from HF
  4. Load base with Gemma4ForConditionalGeneration (preserves 720 tensors)
  5. Manual merge : iterate adapter_model.safetensors, apply deltas in place
  6. Save merged to disk + push to HF
  7. Free base + lora dirs from disk
  8. Convert HF -> GGUF bf16 + extract mmproj
  9. Free merged HF safetensors (already on HF)
 10. Quantize bf16 -> Q4_K_M, Q5_K_M, Q8_0 (sequentially)
 11. Push all GGUFs + mmproj -> output.gguf_repo
 12. Done.

Usage (on pod) :
  python pipeline/02b_merge_and_export.py

Env :
  HF_TOKEN  required

Hardware : A100 80GB (BF16 31B = ~62 GB), 250 GB disk, ~1h30, ~$3.
"""
from __future__ import annotations

import os, shutil, subprocess, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = Path("/workspace")
LLAMA = WORK / "llama.cpp"
BASE_DIR = WORK / "base"
LORA_DIR = WORK / "lora"
MERGED = WORK / "merged"
OUT = WORK / "out"


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


def load_config():
    """Load configs/aura.yaml from the repo root."""
    import yaml
    cfg_path = REPO / "configs" / "aura.yaml"
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_module_by_path(model, path: str):
    """Walk a dotted path through nn.Modules. Returns the leaf module."""
    obj = model
    for p in path.split("."):
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj


def get_weight_tensor(module):
    """Return the .weight tensor of a Linear-like module, handling Gemma4ClippableLinear wrappers."""
    import torch.nn as nn
    if isinstance(module, nn.Linear):
        return module.weight
    # Gemma4ClippableLinear wraps a nn.Linear under .linear
    if hasattr(module, "linear") and isinstance(module.linear, nn.Linear):
        return module.linear.weight
    # Fallback : direct .weight attribute
    if hasattr(module, "weight"):
        return module.weight
    raise AttributeError(f"Cannot locate weight tensor on {type(module).__name__}")


def manual_merge_lora(model, lora_dir: Path, alpha: int, r: int, use_rslora: bool = False):
    """
    Apply LoRA adapter to a loaded model in place, by iterating the safetensors
    files and adding (alpha/r) * B @ A to each target module's weight.

    Bypasses PEFT and Unsloth so it works on Gemma4ClippableLinear wrappers.
    """
    import torch
    from safetensors import safe_open
    import math

    scale = (alpha / math.sqrt(r)) if use_rslora else (alpha / r)
    print(f"[MERGE] LoRA scale = {scale:.4f} (alpha={alpha}, r={r}, rslora={use_rslora})", flush=True)

    # Find adapter file(s)
    safetensor_files = sorted(lora_dir.glob("adapter_model*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No adapter_model*.safetensors in {lora_dir}")

    pairs: dict[str, dict[str, "torch.Tensor"]] = defaultdict(dict)

    # 1) Load all LoRA tensors and group by target module path
    for sf in safetensor_files:
        with safe_open(str(sf), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                # Standard PEFT key format :
                #   base_model.model.<MODULE_PATH>.lora_A.default.weight  (shape [r, in_features])
                #   base_model.model.<MODULE_PATH>.lora_B.default.weight  (shape [out_features, r])
                if ".lora_A." in key:
                    path = key.split(".lora_A.")[0].replace("base_model.model.", "", 1)
                    pairs[path]["A"] = t
                elif ".lora_B." in key:
                    path = key.split(".lora_B.")[0].replace("base_model.model.", "", 1)
                    pairs[path]["B"] = t
                # Skip embeddings or other non-LoRA keys (logged below)

    print(f"[MERGE] Found {len(pairs)} LoRA target modules", flush=True)

    # 2) Apply each pair
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

        A = ab["A"].to(W.device, dtype=torch.float32)  # [r, in]
        B = ab["B"].to(W.device, dtype=torch.float32)  # [out, r]
        delta = scale * (B @ A)  # [out, in]
        # Sanity check shapes
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

    step("Loading config")
    cfg = load_config()
    BASE_MODEL = cfg['models'][cfg['base_model']['name']]['hf_id']
    LORA_REPO = cfg['output']['lora_repo']
    MERGED_REPO = cfg['output']['merged_repo']
    GGUF_REPO = cfg['output']['gguf_repo']
    print(f"  base    : {BASE_MODEL}")
    print(f"  lora    : {LORA_REPO}")
    print(f"  merged  : {MERGED_REPO}")
    print(f"  gguf    : {GGUF_REPO}")

    step("Installing system deps")
    run(["bash", "-lc", "apt-get update -qq && apt-get install -y -qq git cmake python3-pip"])

    step("Installing Python deps (no PEFT, no Unsloth - manual merge)")
    # IMPORTANT : do NOT upgrade torch/torchvision (the RunPod base image ships
    # cu124-aligned versions; pip --upgrade pulls cu126/2.11+ which then breaks
    # the torchvision::nms registration). We only install what we strictly need.
    run([sys.executable, "-m", "pip", "install", "-q",
         "huggingface_hub", "hf_transfer", "pyyaml",
         "accelerate", "safetensors", "gguf",
         # transformers main is required for Gemma4ForConditionalGeneration
         # (introduced in 5.5.0.dev0, not yet in any stable release).
         "git+https://github.com/huggingface/transformers.git"])

    if not LLAMA.exists():
        step("Cloning llama.cpp")
        run(["git", "clone", "--depth", "1",
             "https://github.com/ggml-org/llama.cpp.git", str(LLAMA)])
    run([sys.executable, "-m", "pip", "install", "-q", "-r",
         str(LLAMA / "requirements" / "requirements-convert_hf_to_gguf.txt")])

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # ------------------------------------------------------------------
    # 1. DL base + LoRA
    # ------------------------------------------------------------------
    from huggingface_hub import snapshot_download, HfApi

    step(f"Downloading base : {BASE_MODEL} (~62 GB)")
    snapshot_download(repo_id=BASE_MODEL, local_dir=str(BASE_DIR),
                      token=HF_TOKEN, max_workers=8)

    step(f"Downloading LoRA : {LORA_REPO}")
    snapshot_download(repo_id=LORA_REPO, local_dir=str(LORA_DIR),
                      token=HF_TOKEN, max_workers=8)

    # ------------------------------------------------------------------
    # 2. Read LoRA hyperparams from adapter_config.json
    # ------------------------------------------------------------------
    import json
    with open(LORA_DIR / "adapter_config.json", "r", encoding="utf-8") as f:
        adapter_cfg = json.load(f)
    lora_alpha = int(adapter_cfg.get("lora_alpha", 32))
    lora_r = int(adapter_cfg.get("r", 32))
    use_rslora = bool(adapter_cfg.get("use_rslora", False))
    print(f"\n[LoRA cfg] alpha={lora_alpha}, r={lora_r}, use_rslora={use_rslora}")

    # ------------------------------------------------------------------
    # 3. Load base with Gemma4ForConditionalGeneration (FULL multimodal)
    # ------------------------------------------------------------------
    step("Loading base model with Gemma4ForConditionalGeneration (full multimodal)")
    import torch
    from transformers import Gemma4ForConditionalGeneration, AutoProcessor

    model = Gemma4ForConditionalGeneration.from_pretrained(
        str(BASE_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    processor = AutoProcessor.from_pretrained(str(BASE_DIR), token=HF_TOKEN)

    # ------------------------------------------------------------------
    # 4. Manual LoRA merge (handles Gemma4ClippableLinear)
    # ------------------------------------------------------------------
    step("Manual LoRA merge (no PEFT, no Unsloth)")
    manual_merge_lora(model, LORA_DIR, alpha=lora_alpha, r=lora_r, use_rslora=use_rslora)
    model = model.to(torch.bfloat16)

    step(f"Saving merged to {MERGED} (~62 GB)")
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True)
    model.save_pretrained(str(MERGED), safe_serialization=True)
    processor.save_pretrained(str(MERGED))

    # Copy chat template + extras from LoRA repo
    for fname in ["chat_template.jinja", "processor_config.json",
                  "preprocessor_config.json", "special_tokens_map.json",
                  "tokenizer.json", "tokenizer_config.json"]:
        src_f = LORA_DIR / fname
        if src_f.exists():
            shutil.copy2(src_f, MERGED / fname)

    # Free GPU memory before push + GGUF conversion
    del model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Push merged to HF
    # ------------------------------------------------------------------
    step(f"Pushing merged -> {MERGED_REPO}")
    api = HfApi(token=HF_TOKEN)
    api.upload_folder(folder_path=str(MERGED), repo_id=MERGED_REPO,
                      repo_type="model")

    # Free base + lora (no longer needed)
    free(BASE_DIR)
    free(LORA_DIR)

    # ------------------------------------------------------------------
    # 6. Convert HF -> GGUF (bf16 + mmproj)
    # ------------------------------------------------------------------
    OUT.mkdir(exist_ok=True)
    bf16 = OUT / "model-bf16.gguf"
    mmproj = OUT / "Aura-4o-Rebirth-Gemma-4-31B-mmproj-f16.gguf"

    step("Converting HF -> GGUF bf16 (text, ~62 GB)")
    run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(MERGED),
         "--outfile", str(bf16), "--outtype", "bf16"])

    step("Converting HF -> GGUF mmproj (multimodal projector)")
    run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(MERGED),
         "--mmproj", "--outfile", str(mmproj), "--outtype", "f16"])

    # Free merged HF safetensors (already on HF, no longer needed locally)
    free(MERGED)

    # ------------------------------------------------------------------
    # 7. Build llama-quantize
    # ------------------------------------------------------------------
    step("Building llama-quantize")
    quant = LLAMA / "build" / "bin" / "llama-quantize"
    if not quant.exists():
        run(["bash", "-lc",
             f"cd {LLAMA} && cmake -B build && cmake --build build --target llama-quantize -j$(nproc)"])

    # ------------------------------------------------------------------
    # 8. Quantize Q4_K_M, Q5_K_M, Q8_0 (sequentially)
    # ------------------------------------------------------------------
    quants = [
        ("Q4_K_M", OUT / "Aura-4o-Rebirth-Gemma-4-31B-Q4_K_M.gguf"),
        ("Q5_K_M", OUT / "Aura-4o-Rebirth-Gemma-4-31B-Q5_K_M.gguf"),
        ("Q8_0",   OUT / "Aura-4o-Rebirth-Gemma-4-31B-Q8_0.gguf"),
    ]
    for qtype, qfile in quants:
        step(f"Quantizing -> {qtype}")
        run([str(quant), str(bf16), str(qfile), qtype])

    # Free bf16 intermediate after all quants
    if bf16.exists():
        size = bf16.stat().st_size / 1024**3
        print(f"[FREE] {bf16} ({size:.1f} GB)", flush=True)
        bf16.unlink()

    # ------------------------------------------------------------------
    # 9. Push all GGUFs + mmproj
    # ------------------------------------------------------------------
    step(f"Pushing mmproj + Q4_K_M + Q5_K_M + Q8_0 -> {GGUF_REPO}")
    api.upload_file(path_or_fileobj=str(mmproj), path_in_repo=mmproj.name,
                    repo_id=GGUF_REPO, repo_type="model")
    for _, qfile in quants:
        step(f"Uploading {qfile.name}")
        api.upload_file(path_or_fileobj=str(qfile), path_in_repo=qfile.name,
                        repo_id=GGUF_REPO, repo_type="model")

    print("\n[DONE] Clean merged + 3 quants + mmproj rebuilt + pushed.", flush=True)
    print("       Next step : pipeline/04_deploy.py to spin up serverless endpoint.", flush=True)


if __name__ == "__main__":
    main()
