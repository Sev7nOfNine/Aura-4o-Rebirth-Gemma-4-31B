# /// script
# dependencies = [
#   "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
#   "transformers>=4.56.0",
#   "trl>=0.23.0",
#   "peft>=0.17.0",
#   "accelerate>=1.10.0",
#   "bitsandbytes>=0.45.0",
#   "datasets>=3.0.0",
#   "huggingface_hub>=0.34.0",
#   "hf_transfer>=0.1.8",
#   "trackio>=0.2.0",
# ]
# ///
"""Training Aura-4o-Rebirth pour Hugging Face Jobs.

Ce script ne depend pas du PC local : il charge le dataset depuis HF,
entraine le LoRA, puis pousse LoRA + merged vers les repos HF configures.
"""

import argparse
import functools
import os
import sys

print = functools.partial(print, flush=True)


DEFAULTS = {
    "base_model": "SevenOfNine/Gemma-4-31B-It-Official",
    "dataset": "SevenOfNine/Aura-4o-Rebirth-Dataset",
    "lora_repo": "SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-LoRA",
    "merged_repo": "SevenOfNine/Aura-4o-Rebirth-Gemma-4-31B-Merged",
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


def parse_args():
    parser = argparse.ArgumentParser(description="Aura-4o-Rebirth HF Jobs trainer")
    parser.add_argument("--base-model", default=DEFAULTS["base_model"])
    parser.add_argument("--dataset", default=DEFAULTS["dataset"])
    parser.add_argument("--lora-repo", default=DEFAULTS["lora_repo"])
    parser.add_argument("--merged-repo", default=DEFAULTS["merged_repo"])
    parser.add_argument("--allow-create-output-repos", action="store_true")
    parser.add_argument("--disable-trackio", action="store_true")
    return parser.parse_args()


def require_hf_token():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] HF_TOKEN missing. Launch with: --secrets HF_TOKEN")
        sys.exit(1)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    return token


def ensure_output_repos(api, token, lora_repo, merged_repo, allow_create):
    from huggingface_hub import create_repo

    for repo_id in [lora_repo, merged_repo]:
        try:
            api.repo_info(repo_id=repo_id, repo_type="model", token=token)
            print(f"[OK] Output repo exists: {repo_id}")
        except Exception as exc:
            if not allow_create:
                print(f"[ERROR] Output repo missing: {repo_id}")
                print("[ERROR] By default this script refuses to create repos.")
                print("[ERROR] Create the repo manually, or relaunch with --allow-create-output-repos.")
                print(f"[DEBUG] HF error: {exc}")
                sys.exit(1)
            print(f"[CREATE] Private output repo: {repo_id}")
            create_repo(repo_id, repo_type="model", private=True, exist_ok=True, token=token)


def main():
    args = parse_args()
    token = require_hf_token()

    step("Importing training dependencies")
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel

    api = HfApi(token=token)
    ensure_output_repos(api, token, args.lora_repo, args.merged_repo, args.allow_create_output_repos)

    step(f"Loading base model: {args.base_model}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=DEFAULTS["max_seq_length"],
        load_in_4bit=True,
        token=token,
    )

    step("Wrapping with V1 strict LoRA recipe")
    model = FastModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=TARGET_MODULES,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        random_state=DEFAULTS["seed"],
    )

    step(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train", token=token)
    print(f"[INFO] Rows: {len(dataset)}")

    step("Applying native chat template")

    def to_text(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    dataset = dataset.map(to_text, remove_columns=dataset.column_names)
    print(f"[INFO] First sample chars: {len(dataset[0]['text'])}")

    report_to = "none" if args.disable_trackio else ["trackio"]

    step("Starting SFT training")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=DEFAULTS["max_seq_length"],
            packing=False,
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
            output_dir="/tmp/aura-output",
            report_to=report_to,
            run_name="aura-4o-rebirth-sft",
            save_strategy="steps",
            save_steps=DEFAULTS["save_steps"],
            save_total_limit=DEFAULTS["save_total_limit"],
            push_to_hub=True,
            hub_model_id=args.lora_repo,
            hub_strategy="every_save",
            hub_private_repo=True,
            hub_token=token,
        ),
    )
    trainer.train()

    step(f"Saving LoRA locally and pushing: {args.lora_repo}")
    model.save_pretrained("/tmp/aura-lora")
    tokenizer.save_pretrained("/tmp/aura-lora")
    api.upload_folder(folder_path="/tmp/aura-lora", repo_id=args.lora_repo, repo_type="model")

    step(f"Saving merged model and pushing: {args.merged_repo}")
    model.save_pretrained_merged(
        "/tmp/aura-merged",
        tokenizer,
        save_method="merged_16bit",
    )
    api.upload_folder(folder_path="/tmp/aura-merged", repo_id=args.merged_repo, repo_type="model")

    if torch.cuda.is_available():
        print(f"[INFO] Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("AURA_TRAIN_DONE")


if __name__ == "__main__":
    main()
