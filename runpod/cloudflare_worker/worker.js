// ╔════════════════════════════════════════╗
// ║  🔥 Aura-4o-Rebirth - CF PROXY 🔥      ║
// ║  💙 Talons LED FULL CHARGE            ║
// ║  ❤️ By Mel & Aura                     ║
// ╚════════════════════════════════════════╝
//
// Cloudflare Worker proxy : OpenAI-compat front-end -> RunPod Serverless back-end
//
// Exposes a thin /v1/* surface to clients (TypingMind, OpenWebUI, mobile, etc.)
// and forwards the OpenAI-format body to our RunPod custom worker via /runsync.
// The RunPod worker already returns OpenAI-shaped JSON (llama-server's response),
// so we just unwrap RunPod's envelope { output: <openai_response> } before reply.
//
// Why this exists :
//   The /openai/v1/* path on RunPod requires a vLLM-compatible worker (or a
//   custom handler implementing the SSE streaming spec). Our llama.cpp worker
//   uses the simple input/output handler.py pattern, which only answers on
//   /v2/{id}/run and /v2/{id}/runsync. This CF Worker bridges that gap so any
//   OpenAI-spec client can talk to Aura without changes on the worker side.
//
// Deployment :
//   1. Install wrangler:    npm install -g wrangler
//   2. Login:               wrangler login
//   3. Deploy:              wrangler deploy (uses wrangler.toml in this folder)
//   4. Set secrets:
//        wrangler secret put RUNPOD_API_KEY  # paste your rpa_... key
//        wrangler secret put PROXY_KEY       # any random string, you'll use this from clients
//   5. Set vars (in Cloudflare dashboard or wrangler.toml):
//        ENDPOINT_ID = x6sybfwczt4lbb (your RunPod endpoint id)
//
// Client config (TypingMind etc) :
//   Endpoint URL : https://<your-worker>.workers.dev/v1
//   API Key      : <PROXY_KEY>
//   Model ID     : Aura-4o-Rebirth-Gemma-4-31B (any string really)

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  'Access-Control-Max-Age': '86400',
};

const MODEL_ID = 'Aura-4o-Rebirth-Gemma-4-31B';

function jsonResponse(body, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json', ...CORS_HEADERS, ...extraHeaders },
  });
}

function unauthorized(reason = 'Unauthorized') {
  return jsonResponse({ error: { message: reason, type: 'invalid_request_error' } }, 401);
}

async function handleChatCompletions(request, env) {
  // Forward the OpenAI body to RunPod /runsync, wrapped in {input: ...}
  const openaiBody = await request.json();

  const upstream = await fetch(
    `https://api.runpod.ai/v2/${env.ENDPOINT_ID}/runsync`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${env.RUNPOD_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ input: openaiBody }),
    }
  );

  if (!upstream.ok) {
    const text = await upstream.text();
    return jsonResponse(
      { error: { message: `RunPod upstream ${upstream.status}: ${text.slice(0, 500)}`, type: 'upstream_error' } },
      upstream.status === 429 ? 429 : 502,
    );
  }

  const data = await upstream.json();
  // RunPod runsync envelope : { id, status, output, delayTime, executionTime, ... }
  if (data.status === 'COMPLETED' && data.output) {
    // output is already OpenAI-shaped (from llama-server), pass through
    return jsonResponse(data.output);
  }
  if (data.status === 'FAILED') {
    return jsonResponse(
      { error: { message: data.error || 'job failed', type: 'worker_error', details: data } },
      502,
    );
  }
  // IN_QUEUE / IN_PROGRESS shouldn't happen on /runsync but handle gracefully
  return jsonResponse(
    { error: { message: `unexpected runpod status: ${data.status}`, type: 'upstream_error', details: data } },
    504,
  );
}

function handleModels() {
  // Minimal /v1/models response so clients (TypingMind etc.) can validate the connection
  return jsonResponse({
    object: 'list',
    data: [
      {
        id: MODEL_ID,
        object: 'model',
        created: Math.floor(Date.now() / 1000),
        owned_by: 'sevenofnine',
      },
    ],
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    // Auth : require Bearer <PROXY_KEY> on every non-OPTIONS request
    const auth = request.headers.get('Authorization') || '';
    const expected = `Bearer ${env.PROXY_KEY}`;
    if (auth !== expected) {
      return unauthorized('invalid or missing api key');
    }

    // Routes
    if (url.pathname === '/v1/chat/completions' && request.method === 'POST') {
      return handleChatCompletions(request, env);
    }
    if (url.pathname === '/v1/models' && request.method === 'GET') {
      return handleModels();
    }
    // Health
    if (url.pathname === '/' || url.pathname === '/health') {
      return jsonResponse({ ok: true, service: 'aura-4o-rebirth-proxy', model: MODEL_ID });
    }

    return jsonResponse({ error: { message: `route not found: ${request.method} ${url.pathname}` } }, 404);
  },
};
