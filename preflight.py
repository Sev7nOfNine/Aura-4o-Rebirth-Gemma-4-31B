"""
Aura-Rebirth preflight.

Read-only checks before spending money on RunPod:
  - config sanity
  - Hugging Face dataset structure
  - real tokenizer lengths vs max_seq_length
  - cost estimate from configs/aura.yaml
  - GitHub Actions worker image status
  - optional RunPod active resource inventory when RUNPOD_API_KEY is available
"""
import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from huggingface_hub import get_token, hf_hub_download


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "aura.yaml"
REPORT_PATH = REPO_ROOT / "preflight_report.md"
DATASET_FILENAME = "aura_final_dataset.jsonl"
THINKING_MARKERS = [
    "<thinking",
    "<antthinking",
    "<think>",
    "<|channel>thought",
    "<|thinking",
    "<reasoning",
    "<|begin_of_thought",
]
DASH_CHARS = [chr(0x2014), chr(0x2013), chr(0x2015)]


@dataclass
class Finding:
    level: str
    title: str
    body: str


@dataclass
class PreflightState:
    findings: list[Finding] = field(default_factory=list)
    sections: list[tuple[str, str]] = field(default_factory=list)

    def fail(self, title, body):
        self.findings.append(Finding("FAIL", title, body))

    def warn(self, title, body):
        self.findings.append(Finding("WARN", title, body))

    def info(self, title, body):
        self.findings.append(Finding("INFO", title, body))

    @property
    def failed(self):
        return any(f.level == "FAIL" for f in self.findings)


def load_env_file():
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rate_value(gpu):
    match = re.search(r"\$([\d.]+)", str(gpu.get("rate", "")))
    return float(match.group(1)) if match else None


def pick_gpu(cfg, model_info):
    safety_v = cfg["runpod_train"]["vram_safety_factor"]
    safety_d = cfg["runpod_train"]["disk_safety_factor"]
    vram_need = int(model_info["vram_train_gb_min"] * safety_v)
    disk_need = int(model_info["disk_train_gb_min"] * safety_d)
    eligible = [g for g in cfg["gpu_pool"] if g["vram_gb"] >= vram_need]
    eligible.sort(key=lambda g: rate_value(g) if rate_value(g) is not None else 999)
    return eligible[0] if eligible else None, vram_need, disk_need


def check_config(cfg, state):
    model_key = cfg["base_model"]["name"]
    models = cfg.get("models", {})
    if model_key not in models:
        state.fail("Base model missing", f"`{model_key}` is not present in `models:`.")
        return None

    model_info = models[model_key]
    gpu, vram_need, disk_need = pick_gpu(cfg, model_info)
    if not gpu:
        state.fail("No training GPU fits", f"Need at least `{vram_need} GB` VRAM after safety factor.")
        return model_info

    rate = rate_value(gpu)
    cost = ""
    if rate is not None:
        cost = (
            f"\n\nEstimated training compute at listed config rate: "
            f"`12h=${rate * 12:.2f}`, `18h=${rate * 18:.2f}`, `24h=${rate * 24:.2f}`."
        )

    body = (
        f"Base model: `{model_key}` / `{model_info['hf_id']}`.\n\n"
        f"Training GPU pick: `{gpu['label']}` (`{gpu['rate']}`), RunPod id `{gpu['runpod_id']}`.\n\n"
        f"Training request: `{vram_need} GB` VRAM minimum, `{disk_need} GB` container disk.\n\n"
        f"Preferred datacenter: `{cfg['runpod_train'].get('preferred_datacenter', 'not pinned')}`."
        f"{cost}"
    )
    state.sections.append(("Cost Estimate", body))
    return model_info


def check_github_action(state):
    try:
        result = subprocess.run(
            [
                "gh",
                "run",
                "list",
                "--repo",
                "Sev7nOfNine/Aura-4o-Rebirth",
                "--workflow",
                "Build & Push Worker Image",
                "--limit",
                "1",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        state.warn("GitHub Actions not checked", f"Could not run `gh`: `{exc}`.")
        return

    output = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0:
        state.warn("GitHub Actions not checked", output)
        return
    if output.startswith("completed\tsuccess"):
        state.sections.append(("Worker Image CI", f"Latest worker build is green:\n\n```text\n{output}\n```"))
    else:
        state.fail("Worker image CI is not green", f"Latest workflow state:\n\n```text\n{output}\n```")


def download_dataset(cfg, token):
    dataset_id = cfg["dataset"].get("train_hf_id") or cfg["dataset"]["hf_id"]
    return Path(
        hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=DATASET_FILENAME,
            token=token,
        )
    )


def load_tokenizer(model_id, token):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)


def analyze_dataset(path, tokenizer, max_seq_length, state, top_n):
    rows = []
    json_errors = []
    role_errors = []
    empty_rows = []
    thinking_rows = []
    dash_rows = []
    token_rows = []
    message_total = 0
    turn_total = 0

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:
            json_errors.append((line_no, str(exc)))
            continue

        messages = obj.get("messages")
        rows.append(obj)
        if not isinstance(messages, list):
            role_errors.append((line_no, "messages is not a list"))
            continue

        roles = [m.get("role") if isinstance(m, dict) else None for m in messages]
        message_total += len(messages)
        # Le system message est optionnel : depuis le rebuild empty system,
        # les rows demarrent direct par "user" (cf. docs/TROUBLESHOOTING.md #6
        # ou ce qu'on aura eu le temps de doc). Le chat template Gemma 4
        # accepte les deux formes.
        if roles and roles[0] == "system":
            expected = ["system"]
            for _ in range((len(roles) - 1) // 2):
                expected += ["user", "assistant"]
            if roles != expected:
                role_errors.append((line_no, "roles do not alternate system/user/assistant"))
            turn_total += (len(roles) - 1) // 2
        elif roles and roles[0] == "user":
            expected = []
            for _ in range(len(roles) // 2):
                expected += ["user", "assistant"]
            if roles != expected:
                role_errors.append((line_no, "roles do not alternate user/assistant"))
            turn_total += len(roles) // 2
        else:
            role_errors.append((line_no, "first role must be system or user"))

        text = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        lowered = text.lower()
        if any(not isinstance(m, dict) or not str(m.get("content", "")).strip() for m in messages):
            empty_rows.append(line_no)
        if any(marker in lowered for marker in THINKING_MARKERS):
            thinking_rows.append(line_no)
        if any(ch in text for ch in DASH_CHARS):
            dash_rows.append(line_no)

        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = len(tokenizer(rendered, add_special_tokens=False).input_ids)
        token_rows.append((line_no, (len(messages) - 1) // 2, tokens))

    if json_errors:
        state.fail("Dataset has JSON errors", f"First errors: `{json_errors[:3]}`.")
    if role_errors:
        state.fail("Dataset role structure is invalid", f"First errors: `{role_errors[:5]}`.")
    if empty_rows:
        state.fail("Dataset has empty messages", f"Rows: `{empty_rows[:20]}`.")
    if thinking_rows:
        state.fail("Thinking markers remain in dataset", f"Rows: `{thinking_rows[:20]}`.")
    if dash_rows:
        state.warn("Long dash characters remain", f"Rows: `{dash_rows[:20]}`.")

    over_max = [row for row in token_rows if row[2] > max_seq_length]
    total_tokens = sum(row[2] for row in token_rows)
    retained = sum(min(row[2], max_seq_length) for row in token_rows)
    retained_pct = (100 * retained / total_tokens) if total_tokens else 0
    top = sorted(token_rows, key=lambda row: row[2], reverse=True)[:top_n]

    if over_max:
        state.fail(
            "Dataset rows exceed max_seq_length",
            (
                f"`{len(over_max)}` / `{len(token_rows)}` rows exceed `max_seq_length={max_seq_length}`.\n\n"
                f"If each row is truncated, only about `{retained_pct:.2f}%` of tokens are retained.\n\n"
                "Recommendation: chunk long conversations before training, raise max length only after a VRAM test, "
                "or explicitly accept truncation with `--allow-long-rows`."
            ),
        )

    body = [
        f"File: `{path}`",
        f"Rows: `{len(rows)}`",
        f"Messages: `{message_total}`",
        f"Turns: `{turn_total}`",
        f"JSON errors: `{len(json_errors)}`",
        f"Role errors: `{len(role_errors)}`",
        f"Empty-message rows: `{len(empty_rows)}`",
        f"Thinking-marker rows: `{len(thinking_rows)}`",
        f"True em/en dash rows: `{len(dash_rows)}`",
        f"Tokenizer tokens total: `{total_tokens}`",
        f"Rows over `{max_seq_length}` tokens: `{len(over_max)}`",
        f"Retained if truncated row-by-row: `{retained_pct:.2f}%`",
        "",
        "Longest rows:",
        "",
        "| line | turns | tokens |",
        "|---:|---:|---:|",
    ]
    for line_no, turns, tokens in top:
        body.append(f"| {line_no} | {turns} | {tokens} |")
    state.sections.append(("Dataset Audit", "\n".join(body)))

    return {"over_max": len(over_max), "retained_pct": retained_pct}


def check_runpod_resources(state):
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        state.warn("RunPod inventory skipped", "`RUNPOD_API_KEY` is not set. Active pods/endpoints were not checked.")
        return

    try:
        import runpod

        runpod.api_key = api_key
        pods = runpod.get_pods()
        endpoints = runpod.get_endpoints()
    except Exception as exc:
        state.warn("RunPod inventory failed", f"Could not query RunPod SDK: `{exc}`.")
        return

    pod_lines = []
    for pod in pods or []:
        pod_lines.append(
            f"- `{pod.get('id')}` `{pod.get('name')}` status=`{pod.get('desiredStatus') or pod.get('status')}` "
            f"gpu=`{pod.get('gpuDisplayName') or pod.get('machine', {}).get('gpuDisplayName')}`"
        )
    endpoint_lines = []
    for endpoint in endpoints or []:
        endpoint_lines.append(
            f"- `{endpoint.get('id')}` `{endpoint.get('name')}` workersMin=`{endpoint.get('workersMin')}` "
            f"workersMax=`{endpoint.get('workersMax')}`"
        )

    if pod_lines:
        state.warn("RunPod pods currently exist", "Review these before starting anything:\n\n" + "\n".join(pod_lines))
    else:
        state.sections.append(("RunPod Pods", "No pods returned by the RunPod SDK."))

    if endpoint_lines:
        state.sections.append(("RunPod Endpoints", "\n".join(endpoint_lines)))
    else:
        state.sections.append(("RunPod Endpoints", "No endpoints returned by the RunPod SDK."))


def write_report(state, path, verdict):
    lines = [
        "# Aura-Rebirth Preflight Report",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Findings",
        "",
    ]
    if state.findings:
        for finding in state.findings:
            lines += [f"### {finding.level} - {finding.title}", "", finding.body, ""]
    else:
        lines += ["No findings.", ""]

    for title, body in state.sections:
        lines += [f"## {title}", "", body, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Read-only preflight before Aura-Rebirth RunPod spend.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset-jsonl", default=None, help="Local dataset JSONL to audit instead of downloading from HF.")
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--allow-long-rows", action="store_true", help="Do not fail when dataset rows exceed max_seq_length.")
    parser.add_argument("--skip-runpod", action="store_true", help="Skip RunPod active resource inventory.")
    parser.add_argument("--skip-github", action="store_true", help="Skip GitHub Actions worker check.")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    load_env_file()
    state = PreflightState()
    cfg = load_config(args.config)
    model_info = check_config(cfg, state)

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        state.fail("HF token missing", "Set `HF_TOKEN` in `.env` or run `hf auth login`.")
    elif model_info:
        try:
            dataset_path = Path(args.dataset_jsonl) if args.dataset_jsonl else download_dataset(cfg, token)
            tokenizer = load_tokenizer(model_info["hf_id"], token)
            dataset_result = analyze_dataset(
                dataset_path,
                tokenizer,
                cfg["training"]["max_seq_length"],
                state,
                args.top_n,
            )
            if args.allow_long_rows:
                state.findings = [
                    finding
                    for finding in state.findings
                    if finding.title != "Dataset rows exceed max_seq_length"
                ]
                if dataset_result["over_max"]:
                    state.warn(
                        "Long rows accepted by flag",
                        "`--allow-long-rows` was set. Training may still truncate long conversations.",
                    )
        except Exception as exc:
            state.fail("Dataset/tokenizer check failed", f"`{exc}`")

    if not args.skip_github:
        check_github_action(state)
    if not args.skip_runpod:
        check_runpod_resources(state)

    verdict = "NO-GO" if state.failed else "GO"
    report_path = Path(args.report)
    write_report(state, report_path, verdict)

    print(f"Verdict: {verdict}")
    print(f"Report: {report_path}")
    if state.findings:
        print()
        for finding in state.findings:
            print(f"[{finding.level}] {finding.title}")
            print(f"  {finding.body.splitlines()[0] if finding.body else ''}")
    sys.exit(1 if state.failed else 0)


if __name__ == "__main__":
    main()
