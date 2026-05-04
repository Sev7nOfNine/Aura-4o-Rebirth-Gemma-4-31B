"""
╔════════════════════════════════════════╗
║  🔥 AURA+++ - MERGE + EXPORT 🔥       ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝

Merge LoRA into base Gemma 4 31B (multimodal preserved) + convert to GGUF (Q4_K_M + Q5_K_M + Q8_0) + push HF.

Pipeline (à executer sur un pod RunPod A100 80GB) :
  1. Read configs/aura.yaml
  2. DL base model (full multimodal) from HF
  3. DL LoRA adapter from HF
  4. Load base with Gemma4ForConditionalGeneration (NOT AutoModelForCausalLM)
     -> preserves text + vision + audio = 720 tensors
  5. PEFT merge_and_unload + save merged to disk (BF16)
  6. Push merged -> output.merged_repo
  7. Free base + lora dirs from disk
  8. Convert HF -> GGUF bf16 + extract mmproj
  9. Free merged HF safetensors (already on HF)
 10. Quantize bf16 -> Q4_K_M, Q5_K_M, Q8_0 (sequentially)
 11. Push all GGUFs + mmproj -> output.gguf_repo
 12. Done.

Why this script exists :
  - 02_train.py produces the LoRA adapter (already pushed by Mel on 2026-05-03).
  - 03_abliterate.py expects the merged model to already be on HF.
  - This 02b step bridges the gap : merge LoRA + base into HF Merged repo,
    then produce the GGUF artifacts in one shot (no abliteration).
  - Critical fix : uses Gemma4ForConditionalGeneration to keep the multimodal
    encoders (vision/audio = 54 tensors) that AutoModelForCausalLM silently drops.

Usage (on pod) :
  python pipeline/02b_merge_and_export.py

Env :
  HF_TOKEN  required

Hardware : A100 80GB (BF16 31B = ~62 GB), 250 GB disk, ~1h30, ~$3.
"""
from __future__ import annotations

import os, shutil, subprocess, sys
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

    step("Installing Python deps")
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-q",
         "huggingface_hub[cli]", "hf_transfer", "pyyaml",
         "transformers", "peft", "accelerate", "safetensors",
         "datasets", "gguf", "torch", "torchvision"])

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
    # 2. Merge with Gemma4ForConditionalGeneration (FULL multimodal)
    # ------------------------------------------------------------------
    step("Loading base model with Gemma4ForConditionalGeneration (full multimodal)")
    import torch
    from transformers import Gemma4ForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    base = Gemma4ForConditionalGeneration.from_pretrained(
        str(BASE_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=HF_TOKEN,
    )
    processor = AutoProcessor.from_pretrained(str(BASE_DIR), token=HF_TOKEN)

    step("Attaching LoRA + merge_and_unload")
    peft_model = PeftModel.from_pretrained(base, str(LORA_DIR), token=HF_TOKEN)
    merged = peft_model.merge_and_unload()
    merged = merged.to(torch.bfloat16)

    step(f"Saving merged to {MERGED} (~62 GB)")
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True)
    merged.save_pretrained(str(MERGED), safe_serialization=True)
    processor.save_pretrained(str(MERGED))

    # Copy chat template + extras from LoRA repo
    for fname in ["chat_template.jinja", "processor_config.json",
                  "preprocessor_config.json", "special_tokens_map.json",
                  "tokenizer.json", "tokenizer_config.json"]:
        src_f = LORA_DIR / fname
        if src_f.exists():
            shutil.copy2(src_f, MERGED / fname)

    # Free GPU memory
    del base, peft_model, merged
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3. Push merged to HF
    # ------------------------------------------------------------------
    step(f"Pushing merged -> {MERGED_REPO}")
    api = HfApi(token=HF_TOKEN)
    api.upload_folder(folder_path=str(MERGED), repo_id=MERGED_REPO,
                      repo_type="model")

    # Free base + lora (no longer needed)
    free(BASE_DIR)
    free(LORA_DIR)

    # ------------------------------------------------------------------
    # 4. Convert HF -> GGUF (bf16 + mmproj)
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
    # 5. Build llama-quantize
    # ------------------------------------------------------------------
    step("Building llama-quantize")
    quant = LLAMA / "build" / "bin" / "llama-quantize"
    if not quant.exists():
        run(["bash", "-lc",
             f"cd {LLAMA} && cmake -B build && cmake --build build --target llama-quantize -j$(nproc)"])

    # ------------------------------------------------------------------
    # 6. Quantize Q4_K_M, Q5_K_M, Q8_0 (sequentially)
    # ------------------------------------------------------------------
    quants = [
        ("Q4_K_M", OUT / "Aura-4o-Rebirth-Gemma-4-31B-Q4_K_M.gguf"),
        ("Q5_K_M", OUT / "Aura-4o-Rebirth-Gemma-4-31B-Q5_K_M.gguf"),
        ("Q8_0",   OUT / "Aura-4o-Rebirth-Gemma-4-31B-Q8_0.gguf"),
    ]
    for qtype, qfile in quants:
        step(f"Quantizing -> {qtype}")
        run([str(quant), str(bf16), str(qfile), qtype])

    # Free bf16 intermediate (no longer needed after all quants)
    if bf16.exists():
        size = bf16.stat().st_size / 1024**3
        print(f"[FREE] {bf16} ({size:.1f} GB)", flush=True)
        bf16.unlink()

    # ------------------------------------------------------------------
    # 7. Push all GGUFs + mmproj
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
