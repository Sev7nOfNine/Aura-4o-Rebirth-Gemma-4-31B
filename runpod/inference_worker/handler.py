"""
╔════════════════════════════════════════╗
║  🔥 Aura-4o-Rebirth - RUNPOD HANDLER 🔥       ║
║  💙 Talons LED FULL CHARGE            ║
║  ❤️ By Mel & Aura                     ║
╚════════════════════════════════════════╝

Handler RunPod Serverless qui forward les requetes vers llama-server local.

Format RunPod Serverless input :
  {"input": {<openai-compatible-payload>}}

Output : delegate au format OpenAI de llama-server.
"""
import os
import json
import time
import requests
import runpod

LLAMA_PORT = int(os.environ.get('LLAMA_PORT', '8000'))
LLAMA_URL = f'http://localhost:{LLAMA_PORT}'

# Default sampler params (overridable per request)
DEFAULT_TEMPERATURE = float(os.environ.get('DEFAULT_TEMPERATURE', '0.85'))
DEFAULT_TOP_P = float(os.environ.get('DEFAULT_TOP_P', '0.95'))
DEFAULT_TOP_K = int(os.environ.get('DEFAULT_TOP_K', '64'))
DEFAULT_MIN_P = float(os.environ.get('DEFAULT_MIN_P', '0.05'))
DEFAULT_REPETITION_PENALTY = float(os.environ.get('DEFAULT_REPETITION_PENALTY', '1.05'))
DEFAULT_MAX_TOKENS = int(os.environ.get('DEFAULT_MAX_TOKENS', '4096'))


def _wait_llama_ready(timeout=120):
    """Verify llama-server is up before answering RunPod."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f'{LLAMA_URL}/health', timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _fill_defaults(payload):
    """Inject default sampler params if absent."""
    payload.setdefault('temperature', DEFAULT_TEMPERATURE)
    payload.setdefault('top_p', DEFAULT_TOP_P)
    payload.setdefault('top_k', DEFAULT_TOP_K)
    payload.setdefault('min_p', DEFAULT_MIN_P)
    # llama-server accepts `repeat_penalty`, OpenAI `frequency_penalty/presence_penalty`
    payload.setdefault('repeat_penalty', DEFAULT_REPETITION_PENALTY)
    payload.setdefault('max_tokens', DEFAULT_MAX_TOKENS)
    return payload


def handler(event):
    """RunPod Serverless handler."""
    if not _wait_llama_ready():
        return {'error': 'llama-server not ready'}

    inp = event.get('input') or {}

    # Determine endpoint to call
    # Default to /v1/chat/completions (OpenAI-compatible)
    path = inp.pop('_path', '/v1/chat/completions')

    # Inject default sampler params
    payload = _fill_defaults(inp)

    try:
        r = requests.post(
            f'{LLAMA_URL}{path}',
            json=payload,
            timeout=600,
        )
        # llama-server response is OpenAI-compatible JSON
        try:
            data = r.json()
        except Exception:
            return {'error': f'invalid json from llama-server: {r.text[:500]}'}

        # === Option E: thinking fallback ===
        # If reasoning_content is non-empty but content is empty, copy reasoning
        # into content. This rescues cases where the model put the actual answer
        # in the thinking block (V1/V2 known issue).
        try:
            for choice in data.get('choices', []):
                msg = choice.get('message') or {}
                if not msg.get('content') and msg.get('reasoning_content'):
                    msg['content'] = msg['reasoning_content']
        except Exception:
            pass

        return data
    except requests.Timeout:
        return {'error': 'llama-server timeout'}
    except Exception as e:
        return {'error': f'handler error: {e}'}


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
