/**
 * rftmw_native_serve.js
 *
 * Ollama-compatible HTTP server that bridges to the native C++ inference
 * binary (rftmw_qwen25_native --serve) via stdin/stdout.
 *
 * Zero Python in the loop.  The native binary mmaps the GGUF, runs
 * AVX2-accelerated Q4_K/Q6_K matmul, and streams tokens back.
 *
 * Endpoints:
 *   GET  /api/tags   — list models
 *   POST /api/chat   — streaming or non-streaming chat
 *   GET  /health     — liveness
 *
 * Usage:
 *   GGUF_PATH=models/qwen2.5-coder-3b-instruct-q4_k_m.gguf \
 *   NATIVE_BIN=src/rftmw_native/build_qwen25/rftmw_qwen25_native \
 *   node mobile/server/rftmw_native_serve.js
 */

const http = require("http");
const { spawn } = require("child_process");
const path = require("path");
const readline = require("readline");

const fs = require("fs");

const PORT = parseInt(process.env.RFTMW_PORT || "8000", 10);
const HOST = process.env.RFTMW_HOST || "0.0.0.0";
const MAX_SEQ = parseInt(process.env.RFTMW_MAX_SEQ || "2048", 10);
const MAX_TOKENS = parseInt(process.env.RFTMW_MAX_TOKENS || "256", 10);
const MIMOSA_CACHE = parseInt(process.env.RFTMW_MIMOSA_CACHE || "512", 10);

const PROJECT_ROOT = path.resolve(__dirname, "../..");
const GGUF_PATH = process.env.GGUF_PATH
  || path.join(PROJECT_ROOT, "models", "qwen2.5-coder-3b-instruct-q4_k_m.gguf");
const NATIVE_BIN = process.env.NATIVE_BIN
  || path.join(PROJECT_ROOT, "src", "rftmw_native", "build_qwen25", "rftmw_qwen25_native");

// Derive a human-readable model name from the GGUF filename
function deriveModelName(ggufPath) {
  const base = path.basename(ggufPath, ".gguf").replace(/-q\d.*$/i, "").replace(/[_-]+/g, " ");
  return base || "Unknown Model";
}
const MODEL_DISPLAY_NAME = deriveModelName(GGUF_PATH);
const MODEL_TAG = path.basename(GGUF_PATH, ".gguf").toLowerCase();

// ── Hex encode/decode (match the native protocol) ────────────────

function hexEncode(str) {
  return Buffer.from(str, "utf-8").toString("hex");
}
function hexDecode(hex) {
  return Buffer.from(hex, "hex").toString("utf-8");
}

// ── Native process management ────────────────────────────────────

let nativeProc = null;
let nativeRL = null;
let pendingResolve = null;
let lineBuffer = [];
let ready = false;

function startNative() {
  return new Promise((resolve, reject) => {
    console.log(`[native-serve] spawning: ${NATIVE_BIN}`);
    console.log(`[native-serve] GGUF: ${GGUF_PATH}`);

    nativeProc = spawn(NATIVE_BIN, [
      "--gguf", GGUF_PATH,
      "--max-seq", String(MAX_SEQ),
      ...(MIMOSA_CACHE > 0 ? ["--mimosa-cache", String(MIMOSA_CACHE)] : []),
      "--serve",
    ], {
      stdio: ["pipe", "pipe", "inherit"],
      cwd: PROJECT_ROOT,
    });

    nativeProc.on("error", (err) => {
      console.error(`[native-serve] process error: ${err.message}`);
      reject(err);
    });

    nativeProc.on("exit", (code) => {
      console.error(`[native-serve] process exited with code ${code}`);
      nativeProc = null;
      ready = false;
    });

    nativeRL = readline.createInterface({ input: nativeProc.stdout });

    nativeRL.on("line", (line) => {
      if (!ready) {
        if (line.trim() === "READY") {
          ready = true;
          console.log("[native-serve] engine READY");
          resolve();
        }
        return;
      }
      if (pendingResolve) {
        lineBuffer.push(line);
        // Check if DONE or ERROR
        if (line.startsWith("DONE\t") || line.startsWith("ERROR\t") || line === "BYE") {
          const cb = pendingResolve;
          pendingResolve = null;
          const lines = lineBuffer.splice(0);
          cb(lines);
        }
      }
    });
  });
}

function sendGenerate(prompt, maxTokens, templateMode) {
  return new Promise((resolve, reject) => {
    if (!nativeProc || !ready) {
      return reject(new Error("native engine not ready"));
    }
    lineBuffer = [];
    pendingResolve = resolve;

    const hex = hexEncode(prompt);
    const cmd = `GENERATE\t${maxTokens}\t${templateMode}\t${hex}\n`;
    nativeProc.stdin.write(cmd);
  });
}

// Streaming variant: calls onToken for each TOKEN line
function sendGenerateStreaming(prompt, maxTokens, templateMode, onToken) {
  return new Promise((resolve, reject) => {
    if (!nativeProc || !ready) {
      return reject(new Error("native engine not ready"));
    }

    // Temporarily replace the line handler
    const origHandler = nativeRL;
    let tokenLines = [];

    const tempHandler = (line) => {
      if (line.startsWith("TOKEN\t")) {
        const parts = line.split("\t");
        const tokenText = hexDecode(parts[2] || "");
        onToken(tokenText);
        tokenLines.push(line);
      } else if (line.startsWith("DONE\t") || line.startsWith("ERROR\t")) {
        nativeRL.removeListener("line", tempHandler);
        nativeRL.on("line", defaultLineHandler);
        resolve(line);
      } else if (line.startsWith("STAGE\t")) {
        // ignore stage updates
      }
    };

    // Remove the default handler, add temp
    nativeRL.removeAllListeners("line");
    nativeRL.on("line", tempHandler);

    const hex = hexEncode(prompt);
    const cmd = `GENERATE\t${maxTokens}\t${templateMode}\t${hex}\n`;
    nativeProc.stdin.write(cmd);
  });
}

function defaultLineHandler(line) {
  if (pendingResolve) {
    lineBuffer.push(line);
    if (line.startsWith("DONE\t") || line.startsWith("ERROR\t") || line === "BYE") {
      const cb = pendingResolve;
      pendingResolve = null;
      cb(lineBuffer.splice(0));
    }
  }
}

// ── Build ChatML prompt ──────────────────────────────────────────

function buildPrompt(messages) {
  let prompt = "";
  for (const msg of messages) {
    prompt += `<|im_start|>${msg.role || "user"}\n${msg.content || ""}<|im_end|>\n`;
  }
  prompt += "<|im_start|>assistant\n";
  return prompt;
}

// ── HTTP server ──────────────────────────────────────────────────

function createServer() {
  const server = http.createServer(async (req, res) => {
    // CORS
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");

    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    // ── Serve the web chat UI ──
    if (req.method === "GET" && (req.url === "/" || req.url === "/index.html")) {
      const uiPath = path.join(__dirname, "chat.html");
      try {
        const html = fs.readFileSync(uiPath, "utf-8");
        res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
        res.end(html);
      } catch {
        res.writeHead(500, { "Content-Type": "text/plain" });
        res.end("chat.html not found");
      }
      return;
    }

    if (req.method === "GET" && req.url === "/health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: ready ? "ok" : "loading" }));
      return;
    }

    if (req.method === "GET" && req.url === "/api/tags") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        models: [{
          name: MODEL_TAG,
          model: MODEL_TAG,
          display_name: MODEL_DISPLAY_NAME,
          modified_at: new Date().toISOString(),
          size: 0,
          digest: "native-gguf",
          details: {
            format: "gguf-q4_k_m",
            family: MODEL_DISPLAY_NAME.split(" ")[0],
            quantization_level: "Q4_K_M",
          },
        }],
      }));
      return;
    }

    if (req.method === "POST" && req.url === "/api/chat") {
      let body = "";
      for await (const chunk of req) body += chunk;

      let parsed;
      try {
        parsed = JSON.parse(body);
      } catch {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "invalid JSON" }));
        return;
      }

      const messages = parsed.messages || [];
      if (!messages.length) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "no messages" }));
        return;
      }

      const stream = parsed.stream !== false;
      const options = parsed.options || {};
      const maxTok = Math.min(
        parseInt(options.max_new_tokens || options.num_predict || MAX_TOKENS, 10),
        MAX_TOKENS
      );
      const modelName = parsed.model || "qwen2.5-coder:3b";
      const prompt = buildPrompt(messages);

      if (!ready) {
        res.writeHead(503, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "engine not ready" }));
        return;
      }

      if (stream) {
        res.writeHead(200, {
          "Content-Type": "application/x-ndjson",
          "Transfer-Encoding": "chunked",
        });

        try {
          const doneLine = await sendGenerateStreaming(prompt, maxTok, "raw", (tokenText) => {
            // Skip stop-sequence tokens in stream
            if (tokenText.includes("<|im_end|>") || tokenText.includes("<|endoftext|>")) return;
            const chunk = JSON.stringify({
              model: modelName,
              created_at: new Date().toISOString(),
              message: { role: "assistant", content: tokenText },
              done: false,
            }) + "\n";
            res.write(chunk);
          });

          // Final done chunk
          res.write(JSON.stringify({
            model: modelName,
            created_at: new Date().toISOString(),
            message: { role: "assistant", content: "" },
            done: true,
            done_reason: "stop",
          }) + "\n");
        } catch (err) {
          res.write(JSON.stringify({
            model: modelName,
            created_at: new Date().toISOString(),
            message: { role: "assistant", content: `\n[error: ${err.message}]` },
            done: true,
          }) + "\n");
        }
        res.end();
      } else {
        try {
          let fullText = "";
          const doneLine = await sendGenerateStreaming(prompt, maxTok, "raw", (tokenText) => {
            fullText += tokenText;
          });
          // Prefer DONE line text (has complete output with stop seq already stripped by native binary)
          if (doneLine && doneLine.startsWith("DONE\t")) {
            const parts = doneLine.split("\t");
            if (parts[1]) fullText = hexDecode(parts[1]);
          }
          // Belt-and-suspenders: also strip stop sequences and partial trailing fragments
          fullText = fullText.replace(/<\|im_end\|>/g, "").replace(/<\|endoftext\|>/g, "");
          fullText = fullText.replace(/<\|im_end\|?$/, "").replace(/<\|im_end?$/, "")
                             .replace(/<\|im_?$/, "").replace(/<\|i?$/, "")
                             .replace(/<\|endoftext\|?$/, "").replace(/<\|endoftex?$/, "")
                             .replace(/<\|endoft?$/, "").replace(/<\|endo?$/, "")
                             .replace(/<\|end?$/, "").replace(/<\|en?$/, "").replace(/<\|?$/, "");
          fullText = fullText.trim();
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({
            model: modelName,
            created_at: new Date().toISOString(),
            message: { role: "assistant", content: fullText.trim() },
            done: true,
            done_reason: "stop",
          }));
        } catch (err) {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: err.message }));
        }
      }
      return;
    }

    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "not found" }));
  });

  server.listen(PORT, HOST, () => {
    console.log(`[native-serve] HTTP listening on ${HOST}:${PORT}`);
    console.log(`[native-serve] Web UI:    http://localhost:${PORT}/`);
    console.log(`[native-serve] Endpoints:`);
    console.log(`[native-serve]   GET  /          — chat web UI`);
    console.log(`[native-serve]   GET  /api/tags  — list models (Ollama compat)`);
    console.log(`[native-serve]   POST /api/chat  — streaming chat (Ollama compat)`);
    console.log(`[native-serve]   GET  /health    — liveness check`);
  });
}

// ── Main ─────────────────────────────────────────────────────────

async function main() {
  console.log("[native-serve] RFTMW Native Inference Server (C++/AVX2, dense Q4_K + Mimosa KV)");
  console.log(`[native-serve] GGUF: ${GGUF_PATH}`);
  console.log(`[native-serve] Binary: ${NATIVE_BIN}`);
  console.log(`[native-serve] Max seq: ${MAX_SEQ}, Max tokens: ${MAX_TOKENS}, Mimosa cache: ${MIMOSA_CACHE}`);

  await startNative();
  createServer();
}

main().catch((err) => {
  console.error(`[native-serve] FATAL: ${err.message}`);
  process.exit(1);
});
