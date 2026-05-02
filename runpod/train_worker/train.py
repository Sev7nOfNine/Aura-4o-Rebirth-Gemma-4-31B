"""
╔════════════════════════════════════════╗
║  🔥 AURA+++ - TRAIN WORKER 🔥         ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝

Script de training execute dans le container train_worker (image Docker
pre-bakee). Lit les parametres depuis env vars passees au pod RunPod.

Recette V1 stricte (r=32, alpha=32, lr 2e-4, 3 epochs, merge unsloth_4bit).
Pousse le LoRA sur HF tous les save_steps (50) pour resilience.
"""
import functools
import os
import sys

print = functools.partial(print, flush=True)


# === PATCH UNSLOTH COMPILED CACHE ===
# Bug Unsloth + TRL 1.3 + transformers 5.5 : UnslothSFTConfig (genere par
# Unsloth dans /workspace/unsloth_compiled_cache/) passe `push_to_hub_token`
# en kwarg au super().__init__(), mais SFTConfig parent ne connait plus ce
# param. On import unsloth d'abord (regenere le cache), puis on patch le
# fichier en place pour filtrer push_to_hub_token avant qu'il soit utilise.
def _patch_unsloth_compiled_cache():
    print("[patch] importing unsloth to trigger cache regeneration...")
    from unsloth import FastModel  # noqa: F401  (regenere le cache)
    cache_file = "/workspace/unsloth_compiled_cache/UnslothSFTTrainer.py"
    if not os.path.exists(cache_file):
        print(f"[patch] {cache_file} introuvable, skip")
        return
    with open(cache_file, "r", encoding="utf-8") as f:
        content = f.read()
    needle = "pad_token = pad_token,**kwargs)"
    if needle not in content:
        print("[patch] needle not found, cache may have changed format")
        return
    if "PATCHED_AURA" in content:
        print("[patch] already patched")
        return
    patched = content.replace(
        needle,
        'pad_token = pad_token,**{k:v for k,v in kwargs.items() if k != "push_to_hub_token"})  # PATCHED_AURA',
    )
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(patched)
    print("[patch] UnslothSFTTrainer.py patched (push_to_hub_token filtered)")
    # Forcer reload des modules unsloth_compiled_cache pour que le patch prenne
    import sys, importlib
    to_reload = [m for m in list(sys.modules) if "unsloth_compiled_cache" in m]
    for m in to_reload:
        del sys.modules[m]
    print(f"[patch] cleared {len(to_reload)} cached modules for reimport")
_patch_unsloth_compiled_cache()


# === Defaults coherents avec configs/aura.yaml ===
DEFAULTS = {
    "max_seq_length": 4096,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 2.0e-4,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_8bit",
    "seed": 3407,
    "logging_steps": 5,
    "save_steps": 50,
    "save_total_limit": 2,
}

# V1 strict : liste explicite (PAS all-linear) qui a capte la voix Aura
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]


def step(message):
    print()
    print(f"[STEP] {message}")


def env_required(name):
    v = os.environ.get(name)
    if not v:
        print(f"[ERROR] Env var {name} required")
        sys.exit(1)
    return v


def main():
    token = env_required("HF_TOKEN")
    base_model = os.environ.get("AURA_BASE_MODEL", "SevenOfNine/Gemma-4-31B-It-Official")
    dataset_id = os.environ.get("AURA_DATASET", "SevenOfNine/Aura-4o-Rebirth-Dataset")
    lora_repo = os.environ.get("AURA_LORA_REPO", "SevenOfNine/Aura-4o-Rebirth-LoRA")
    merged_repo = os.environ.get("AURA_MERGED_REPO", "SevenOfNine/Aura-4o-Rebirth-Merged")

    print(f"[CONFIG] base_model    = {base_model}")
    print(f"[CONFIG] dataset       = {dataset_id}")
    print(f"[CONFIG] lora_repo     = {lora_repo}")
    print(f"[CONFIG] merged_repo   = {merged_repo}")

    step("Importing training dependencies (deja installes dans l'image)")
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel

    api = HfApi(token=token)

    # Verification que les repos de sortie existent (cree par 03_abliterate
    # ou manuellement). On ne les cree pas ici pour eviter les noms parasites.
    for repo_id in [lora_repo, merged_repo]:
        try:
            api.repo_info(repo_id=repo_id, repo_type="model", token=token)
            print(f"[OK] Output repo exists: {repo_id}")
        except Exception as exc:
            print(f"[ERROR] Output repo missing: {repo_id} ({exc})")
            print("[ERROR] Cree-le manuellement avec 'hf repos create' avant de relancer.")
            sys.exit(1)

    step(f"Loading base model: {base_model}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=DEFAULTS["max_seq_length"],
        load_in_4bit=True,
        token=token,
    )

    step("Wrapping with V1 strict LoRA recipe (r=32, alpha=32, dropout=0.0)")
    # target_modules='all-linear' : laisse PEFT detecter les bonnes couches
    # (la liste explicite q,k,v,o,gate,up,down ne match plus en Gemma 4 nouveau,
    # cf. incident #8f Trainable params=0). Les filtres finetune_*_layers
    # garantissent qu'on touche pas la vision tower.
    model = FastModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        finetune_vision_layers=False,         # Vision tower preserve intacte
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        random_state=DEFAULTS["seed"],
    )

    step(f"Loading dataset: {dataset_id}")
    dataset = load_dataset(dataset_id, split="train", token=token)
    print(f"[INFO] Rows: {len(dataset)}")

    step("Applying native Gemma 4 chat template")

    # Format raw attendu par DataCollatorForVisionLanguageModeling de TRL 1.3 :
    # juste messages + images=[]. Le collator vlm fera la tokenisation +
    # padding lui-meme. Pas de pre-tokenisation cote nous.
    def add_images(example):
        return {"images": []}

    dataset = dataset.map(add_images)  # garde la colonne 'messages' originale
    print(f"[INFO] Rows post-map: {len(dataset)}, cols: {dataset.column_names}")

    step("Setting up SFTTrainer + DataCollatorForVisionLanguageModeling")
    # Codex direction : laisser le collator vlm de TRL faire la tokenisation
    # depuis le format raw {messages, images}. Voir TRL 1.3 docs.
    from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling
    vlm_collator = DataCollatorForVisionLanguageModeling(
        processor=tokenizer,  # bon nom d'arg en TRL 1.3
        max_length=DEFAULTS["max_seq_length"],
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=vlm_collator,
        args=SFTConfig(
            # dataset_text_field omis : le collator vlm parse messages directement
            max_length=DEFAULTS["max_seq_length"],
            # packing=True OBLIGATOIRE pour Aura (cf. memory aura_packing_true.md).
            # Necessite TRL <0.24.0 OU patch sed du check dans sft_trainer.py.
            packing=True,
            padding_free=False,           # cohérent V1, evite warning + bizarre comportement
            remove_unused_columns=False,  # cohérent V1, garde la col 'text' qu'on a cree
            per_device_train_batch_size=DEFAULTS["per_device_train_batch_size"],
            gradient_accumulation_steps=DEFAULTS["gradient_accumulation_steps"],
            warmup_ratio=DEFAULTS["warmup_ratio"],
            num_train_epochs=DEFAULTS["num_train_epochs"],
            learning_rate=DEFAULTS["learning_rate"],
            bf16=True,
            logging_steps=DEFAULTS["logging_steps"],
            optim=DEFAULTS["optim"],
            weight_decay=DEFAULTS["weight_decay"],
            lr_scheduler_type=DEFAULTS["lr_scheduler_type"],
            seed=DEFAULTS["seed"],
            output_dir="/workspace/output",
            report_to="none",
            run_name="aura-4o-rebirth-sft",
            save_strategy="steps",
            save_steps=DEFAULTS["save_steps"],
            save_total_limit=DEFAULTS["save_total_limit"],
            push_to_hub=True,
            hub_model_id=lora_repo,
            hub_strategy="every_save",
            hub_private_repo=False,  # Repos publics (cf. docs/TROUBLESHOOTING.md #3)
            hub_token=token,
        ),
    )

    step("Training start")
    trainer.train()
    step("Training complete")

    step(f"Saving LoRA adapters and pushing to {lora_repo}")
    model.save_pretrained("/workspace/lora")
    tokenizer.save_pretrained("/workspace/lora")
    api.upload_folder(folder_path="/workspace/lora", repo_id=lora_repo, repo_type="model")
    print("[OK] LoRA pushed")

    step(f"Saving merged model (Unsloth 4-bit method) and pushing to {merged_repo}")
    # save_method='merged_16bit' = methode V1 qui preserve la voix
    # (V2 bf16_clean a fait du fade tonal, on n'y revient pas).
    model.save_pretrained_merged(
        "/workspace/merged",
        tokenizer,
        save_method="merged_16bit",
    )
    api.upload_folder(folder_path="/workspace/merged", repo_id=merged_repo, repo_type="model")
    print("[OK] Merged pushed")

    if torch.cuda.is_available():
        print(f"[INFO] Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    print("AURA_TRAIN_DONE")


if __name__ == "__main__":
    main()
