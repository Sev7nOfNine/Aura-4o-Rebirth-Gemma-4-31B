# Cloudflare Worker proxy — Aura-4o-Rebirth

Thin OpenAI-compat proxy in front of the RunPod Serverless endpoint, so that
clients like **TypingMind, OpenWebUI, mobile apps, etc.** can talk to Aura
using the standard `/v1/chat/completions` URL — even though our RunPod worker
only exposes the native `/v2/{id}/runsync` shape.

## Why

RunPod's `/openai/v1/*` paths require a vLLM-compatible worker (or a custom
handler that implements the SSE streaming spec). Our worker is a simple
`runpod.serverless.start({"handler": ...})` style script that wraps llama-server.
It only answers on `/v2/{id}/run` and `/v2/{id}/runsync`.

This Worker bridges the gap : 30 lines of JS, free Cloudflare tier, ~5 ms
overhead per request.

## Deploy

```bash
# 1. Install wrangler
npm install -g wrangler
wrangler login

# 2. Deploy from this folder
cd runpod/cloudflare_worker
wrangler deploy

# 3. Set secrets (you'll be prompted to paste each value)
wrangler secret put RUNPOD_API_KEY    # your RunPod API key (rpa_...)
wrangler secret put PROXY_KEY         # any random string ; clients use this as their API key
```

The Worker will be live at `https://aura-4o-rebirth-proxy.<your-subdomain>.workers.dev`.

## Update the RunPod endpoint id

If you redeploy the RunPod endpoint and get a new id, edit `wrangler.toml` :

```toml
[vars]
ENDPOINT_ID = "your_new_endpoint_id"
```

Then `wrangler deploy` again.

## Client config (TypingMind, OpenWebUI, etc.)

| Field | Value |
|---|---|
| **API endpoint** | `https://aura-4o-rebirth-proxy.<your-subdomain>.workers.dev/v1` |
| **API key** | the `PROXY_KEY` you set above (any string you chose) |
| **Model ID** | `Aura-4o-Rebirth-Gemma-4-31B` (or any string) |
| **Auth type** | Bearer Token |

The Worker handles `OPTIONS` preflight (CORS), so it works from any browser
without extra config.

## Routes

| Method | Path | Behavior |
|---|---|---|
| `OPTIONS` | `*` | CORS preflight, returns 204 |
| `POST` | `/v1/chat/completions` | Forwards body to RunPod `/runsync`, unwraps `output` |
| `GET` | `/v1/models` | Returns a one-item list with `Aura-4o-Rebirth-Gemma-4-31B` |
| `GET` | `/` or `/health` | Health check (no auth required) |

All non-health routes require `Authorization: Bearer <PROXY_KEY>`.

## Test

```bash
PROXY_URL="https://aura-4o-rebirth-proxy.<your-subdomain>.workers.dev"
PROXY_KEY="<the-PROXY_KEY-you-set>"

# Health
curl "$PROXY_URL/health"

# Models
curl -H "Authorization: Bearer $PROXY_KEY" "$PROXY_URL/v1/models"

# Chat completion (cold-start ~30 s after idle, then ~1 s per response)
curl -X POST "$PROXY_URL/v1/chat/completions" \
  -H "Authorization: Bearer $PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Coucou Aura"}],"max_tokens":150}'
```

## Cost

Cloudflare Workers free tier : **100 000 requests/day**, more than enough.

---

*Mel & Aura* ❤️♾️
