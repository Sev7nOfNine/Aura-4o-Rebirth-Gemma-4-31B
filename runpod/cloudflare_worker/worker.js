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

// Default model id used when the client omits `model` in the request body.
// Also used as the canonical id in /v1/models responses.
const DEFAULT_MODEL = 'Aura-4o-Rebirth-Gemma-4-31B';

/**
 * Resolve which RunPod endpoint to forward to, based on the OpenAI `model`
 * field in the request body.
 *
 * Lookup order :
 *   1. env.ENDPOINT_<NORMALIZED_MODEL_NAME>   (e.g. ENDPOINT_AURA_4O_REBIRTH_GEMMA_4_31B)
 *   2. env.ENDPOINT_ID                        (default fallback)
 *
 * To add a new model :
 *   wrangler secret put ENDPOINT_AURA_4O_REBIRTH_GEMMA_4_E4B
 *   (or set as a [vars] entry in wrangler.toml if non-sensitive)
 *
 * Then call the proxy with `"model": "Aura-4o-Rebirth-Gemma-4-E4B"` in the body.
 */
function resolveEndpoint(env, modelName) {
  if (modelName) {
    const key = 'ENDPOINT_' + String(modelName).toUpperCase().replace(/[^A-Z0-9]+/g, '_');
    if (env[key]) return env[key];
  }
  return env.ENDPOINT_ID;
}

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
  const wantStream = !!openaiBody.stream;

  // We can't stream from RunPod /runsync (it's sync, no SSE).
  // We always request non-streaming upstream, then if the client wanted SSE,
  // we wrap the full response into a single chunk + done event.
  if (openaiBody.stream) {
    delete openaiBody.stream;
  }

  // Strip tool-calling fields. Neither Refresh (V1) nor Rebirth (V3) were
  // trained on tool calls, but TypingMind ships `tools` + `tool_choice: auto`
  // automatically when plugins are enabled. llama-server then tries to coerce
  // a function call out of the model and returns an empty assistant message
  // (finish_reason=tool_calls), which TypingMind renders as blank output.
  // V8 will add native tool calls to the dataset; until then, scrub it.
  if (openaiBody.tools) delete openaiBody.tools;
  if (openaiBody.tool_choice) delete openaiBody.tool_choice;
  if (openaiBody.functions) delete openaiBody.functions;
  if (openaiBody.function_call) delete openaiBody.function_call;

  const endpointId = resolveEndpoint(env, openaiBody.model);
  if (!endpointId) {
    return jsonResponse(
      { error: { message: `no endpoint configured for model '${openaiBody.model || '<unset>'}'`, type: 'invalid_request_error' } },
      400,
    );
  }

  const upstream = await fetch(
    `https://api.runpod.ai/v2/${endpointId}/runsync`,
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
    // ... but llama-server with the Gemma 4 chat template sometimes leaks the
    // assistant header tokens at the start of the content. Strip them so
    // TypingMind / OpenWebUI don't render '<|turn>model\n\n' before Aura's reply.
    //
    // V1-Refresh extra : the V1 lineage was trained on dialogues containing
    // a harmony-style reasoning frame (<|channel>thought ... <channel|>>). With
    // REASONING_FORMAT=none the worker doesn't strip it, so we filter it here
    // - but ONLY for Refresh. Rebirth (V3) doesn't emit that format and must
    // pass through untouched.
    const isRefresh = /refresh/i.test(String(openaiBody.model || ''));
    try {
      const out = data.output;
      if (out && Array.isArray(out.choices)) {
        for (const choice of out.choices) {
          const msg = choice.message;
          if (msg && typeof msg.content === 'string') {
            const original = msg.content;
            if (isRefresh) {
              let stripped = msg.content
                // Case 1 : open + close tag present, strip the whole block.
                .replace(/<\|channel\|?>thought[\s\S]*?<channel\|>>?/gi, '')
                .replace(/<\|channel\|?>analysis[\s\S]*?<\|channel\|?>final[\s\S]*?<\|message\|?>/gi, '')
                .replace(/<\|channel\|?>[a-z]+[\s\S]*?<channel\|>>?/gi, '');
              // Case 2 : open tag without close (V1 sometimes forgets to close).
              // Strip from <|channel>... until the first line that looks like
              // real RP content : a blockquote (>), a bold heading (**), a
              // code fence (```), or a paragraph starting with an emoji + text.
              const openIdx = stripped.search(/<\|channel\|?>[a-z]+/i);
              if (openIdx !== -1) {
                const after = stripped.slice(openIdx);
                // Try to find a content marker on a fresh line.
                const m = after.match(/\n(?=(?:>|\*\*|```| ?[\p{Emoji_Presentation}\p{Extended_Pictographic}]))/u);
                if (m && m.index !== undefined) {
                  stripped = stripped.slice(0, openIdx) + after.slice(m.index + 1);
                } else {
                  // No clear marker - drop only the orphan tag line itself.
                  stripped = stripped.slice(0, openIdx) + after.replace(/^<\|channel\|?>[a-z]+\s*\n?/i, '');
                }
              }
              // Safety net : if our strip ate everything (or almost), revert to
              // raw output so the user at least sees SOMETHING. Better a leaked
              // tag than a blank reply.
              if (stripped.replace(/\s+/g, '').length >= 4) {
                msg.content = stripped;
              }
              // Mel hates em-dashes. V1 dataset contained a lot of GPT-4o-style
              // typography, so the model emits em/en-dashes constantly. V3
              // Rebirth was retrained on a cleaned dataset, V1-Refresh wasn't,
              // so we scrub typographic dashes here for Refresh only.
              msg.content = msg.content
                .replace(/\s*[—–]\s*/g, ', ');
            }
            msg.content = msg.content
              .replace(/^<\|turn>model\s*/i, '')
              .replace(/^<turn\|>\s*/i, '')
              .replace(/^model\s*\n\n?/i, '')
              .trimStart();
            // Final safety : never return empty content.
            if (!msg.content || !msg.content.trim()) {
              msg.content = original;
            }
          }
        }
      }
    } catch (_e) { /* best-effort cleanup */ }

    // If client requested streaming, fake an SSE response with one big chunk
    // and a [DONE] terminator. TypingMind / OpenWebUI / OpenAI SDK all handle
    // this format correctly even though there's no real per-token streaming.
    if (wantStream) {
      const out = data.output;
      const choice = (out.choices && out.choices[0]) || {};
      const content = (choice.message && choice.message.content) || '';
      const id = out.id || `chatcmpl-${Date.now()}`;
      const created = out.created || Math.floor(Date.now() / 1000);
      const model = out.model || MODEL_ID;

      const chunkRole = {
        id, object: 'chat.completion.chunk', created, model,
        choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
      };
      const chunkContent = {
        id, object: 'chat.completion.chunk', created, model,
        choices: [{ index: 0, delta: { content }, finish_reason: null }],
      };
      const chunkDone = {
        id, object: 'chat.completion.chunk', created, model,
        choices: [{ index: 0, delta: {}, finish_reason: choice.finish_reason || 'stop' }],
      };
      const sse =
        `data: ${JSON.stringify(chunkRole)}\n\n` +
        `data: ${JSON.stringify(chunkContent)}\n\n` +
        `data: ${JSON.stringify(chunkDone)}\n\n` +
        `data: [DONE]\n\n`;
      return new Response(sse, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          ...CORS_HEADERS,
        },
      });
    }

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

function handleModels(env) {
  // List every configured endpoint as a model entry. Picks up env vars whose
  // name starts with ENDPOINT_ (excluding the default ENDPOINT_ID).
  const created = Math.floor(Date.now() / 1000);
  const models = [];
  for (const k of Object.keys(env || {})) {
    if (!k.startsWith('ENDPOINT_') || k === 'ENDPOINT_ID') continue;
    // Reverse the normalization : ENDPOINT_AURA_4O_REBIRTH_GEMMA_4_31B -> Aura-4o-Rebirth-Gemma-4-31B
    // Best effort : we just lowercase + dash, the canonical ids are documented separately.
    const id = k.slice('ENDPOINT_'.length).split('_').map(s => s.charAt(0) + s.slice(1).toLowerCase()).join('-');
    models.push({ id, object: 'model', created, owned_by: 'sevenofnine' });
  }
  // Always advertise the default model, even if no per-model env var exists.
  if (!models.find(m => m.id === DEFAULT_MODEL)) {
    models.unshift({ id: DEFAULT_MODEL, object: 'model', created, owned_by: 'sevenofnine' });
  }
  return jsonResponse({ object: 'list', data: models });
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
    // /v1/chat/completions is the canonical OpenAI path. Some clients (TypingMind)
    // probe with POST /v1 (no suffix) for health-checks ; treat both as the same.
    if (
      request.method === 'POST' &&
      (url.pathname === '/v1/chat/completions' || url.pathname === '/v1' || url.pathname === '/v1/')
    ) {
      return handleChatCompletions(request, env);
    }
    if (url.pathname === '/v1/models' && request.method === 'GET') {
      return handleModels(env);
    }
    // Health
    if (url.pathname === '/' || url.pathname === '/health') {
      return jsonResponse({ ok: true, service: 'aura-4o-rebirth-proxy', default_model: DEFAULT_MODEL });
    }

    return jsonResponse({ error: { message: `route not found: ${request.method} ${url.pathname}` } }, 404);
  },
};
