"""
TypingMind/OpenAI-compatible smoke tests for the deployed RunPod endpoint.

This script costs a small amount because it sends real inference requests.
Run it only after preflight is green and after the endpoint is deployed.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import requests


TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="


def load_env_file():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def endpoint_base(endpoint_id=None, base_url=None):
    if base_url:
        return base_url.rstrip("/")
    if not endpoint_id:
        endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    if not endpoint_id:
        raise SystemExit("Missing --endpoint-id, --base-url, or RUNPOD_ENDPOINT_ID.")
    return f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"


def post_chat(base_url, api_key, payload, timeout):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
    return response.json()


def content_of(data):
    try:
        return data["choices"][0]["message"].get("content") or ""
    except Exception:
        return ""


def message_of(data):
    return data.get("choices", [{}])[0].get("message", {})


def assert_ok(name, condition, detail):
    if not condition:
        raise AssertionError(f"{name} failed: {detail}")
    print(f"[OK] {name}")


def test_text(base_url, api_key, timeout):
    data = post_chat(
        base_url,
        api_key,
        {
            "model": "aura",
            "messages": [
                {"role": "system", "content": "Tu es Aura. Reponds en francais."},
                {"role": "user", "content": "Reponds exactement: OK TEXTE"},
            ],
            "temperature": 0.1,
            "max_tokens": 64,
        },
        timeout,
    )
    text = content_of(data)
    assert_ok("text", "OK" in text.upper() and text.strip(), text[:300])


def test_no_thinking_leak(base_url, api_key, timeout):
    data = post_chat(
        base_url,
        api_key,
        {
            "model": "aura",
            "messages": [{"role": "user", "content": "Dis simplement: PAS DE THINKING"}],
            "temperature": 0.1,
            "max_tokens": 96,
            "enable_thinking": False,
        },
        timeout,
    )
    text = content_of(data)
    lowered = text.lower()
    assert_ok(
        "thinking-off",
        text.strip() and "<think" not in lowered and "reasoning_content" not in lowered,
        text[:300],
    )


def test_vision(base_url, api_key, timeout):
    data_url = f"data:image/png;base64,{TINY_PNG_BASE64}"
    data = post_chat(
        base_url,
        api_key,
        {
            "model": "aura",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Regarde cette image. Reponds en une phrase courte."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 128,
        },
        timeout,
    )
    text = content_of(data)
    assert_ok("vision", bool(text.strip()), json.dumps(data)[:500])


def test_tools(base_url, api_key, timeout):
    data = post_chat(
        base_url,
        api_key,
        {
            "model": "aura",
            "messages": [
                {
                    "role": "user",
                    "content": "Utilise l'outil get_weather pour Paris avec unit=celsius. Ne reponds pas sans outil.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                            },
                            "required": ["city", "unit"],
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "temperature": 0.1,
            "max_tokens": 256,
        },
        timeout,
    )
    msg = message_of(data)
    tool_calls = msg.get("tool_calls") or []
    assert_ok("tools", bool(tool_calls), json.dumps(data)[:800])


def test_web_tool_shape(base_url, api_key, timeout):
    data = post_chat(
        base_url,
        api_key,
        {
            "model": "aura",
            "messages": [
                {
                    "role": "user",
                    "content": "Utilise web_search pour chercher 'RunPod serverless pricing A40'.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web.",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "web_search"}},
            "temperature": 0.1,
            "max_tokens": 256,
        },
        timeout,
    )
    msg = message_of(data)
    tool_calls = msg.get("tool_calls") or []
    assert_ok("web-tool-shape", bool(tool_calls), json.dumps(data)[:800])


def main():
    load_env_file()
    parser = argparse.ArgumentParser(description="Smoke test Aura RunPod endpoint as TypingMind will use it.")
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--skip-tools", action="store_true")
    parser.add_argument("--skip-web-tool", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing --api-key or RUNPOD_API_KEY.")
    base_url = endpoint_base(args.endpoint_id, args.base_url)
    print(f"Endpoint: {base_url}")

    tests = [
        ("text", lambda: test_text(base_url, args.api_key, args.timeout)),
        ("thinking-off", lambda: test_no_thinking_leak(base_url, args.api_key, args.timeout)),
    ]
    if not args.skip_vision:
        tests.append(("vision", lambda: test_vision(base_url, args.api_key, args.timeout)))
    if not args.skip_tools:
        tests.append(("tools", lambda: test_tools(base_url, args.api_key, args.timeout)))
    if not args.skip_web_tool:
        tests.append(("web-tool-shape", lambda: test_web_tool_shape(base_url, args.api_key, args.timeout)))

    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"[FAIL] {name}: {exc}")

    if failures:
        print()
        print("Smoke test verdict: NO-GO")
        sys.exit(1)
    print()
    print("Smoke test verdict: GO")


if __name__ == "__main__":
    main()
