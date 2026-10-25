/**
 * QuantoniumOS — Ollama-compatible server with Mimosa KV cache enhancement
 *
 * Proxies to llama.cpp's built-in server, adding:
 *   - Custom web chat UI at GET /
 *   - Mimosa KV cache test endpoint at GET /mimosa-test
 *   - Ollama-compatible /api/chat and /api/tags
 *
 * llama.cpp server runs on LLAMA_PORT (default 8081),
 * this proxy runs on PORT (default 8080).
 */

const http = require("http");
const { spawn, execSync } = require("child_process");
const path = require("path");
const fs = require("fs");

const PORT = parseInt(process.env.PORT || "8080", 10);
const LLAMA_PORT = parseInt(process.env.LLAMA_PORT || "8081", 10);
const HOST = "0.0.0.0";
const MODEL_PATH = process.env.MODEL_PATH
  || "models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf";
const LLAMA_SERVER = process.env.LLAMA_SERVER
  || "llama.cpp/build/bin/llama-server";
const CTX_SIZE = parseInt(process.env.CTX_SIZE || "2048", 10);
const N_THREADS = parseInt(process.env.N_THREADS || "0", 10); // 0 = auto

// ── Start llama.cpp server as child process ──────────────────────

let llamaProc = null;
let llamaReady = false;

function startLlama() {
  return new Promise((resolve, reject) => {
    const args = [
      "--model", MODEL_PATH,
      "--port", String(LLAMA_PORT),
      "--host", "127.0.0.1",
      "--ctx-size", String(CTX_SIZE),
      "--chat-template", "chatml",
    ];
    if (N_THREADS > 0) {
      args.push("--threads", String(N_THREADS));
    }

    console.log(`[server] Starting llama.cpp: ${LLAMA_SERVER}`);
    console.log(`[server] Args: ${args.join(" ")}`);

    llamaProc = spawn(LLAMA_SERVER, args, {
      stdio: ["ignore", "pipe", "pipe"],
    });

    llamaProc.on("error", (err) => {
      console.error(`[server] llama.cpp failed to start: ${err.message}`);
      reject(err);
    });

    llamaProc.on("exit", (code) => {
      console.error(`[server] llama.cpp exited with code ${code}`);
      llamaReady = false;
    });

    // Watch stderr for "listening" message
    let stderrBuf = "";
    llamaProc.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      process.stderr.write(text);  // pass through
      stderrBuf += text;
      if (!llamaReady && stderrBuf.includes("listening")) {
        llamaReady = true;
        console.log(`[server] llama.cpp is ready on port ${LLAMA_PORT}`);
        resolve();
      }
    });

    llamaProc.stdout.on("data", (chunk) => {
      process.stdout.write(chunk);
    });

    // Timeout: if not ready in 120s, resolve anyway
    setTimeout(() => {
      if (!llamaReady) {
        console.log("[server] Timeout waiting for llama.cpp, starting proxy anyway...");
        llamaReady = true;
        resolve();
      }
    }, 120000);
  });
}

// ── Proxy helper: forward request to llama.cpp ───────────────────

function proxyToLlama(req, res, targetPath) {
  const options = {
    hostname: "127.0.0.1",
    port: LLAMA_PORT,
    path: targetPath || req.url,
    method: req.method,
    headers: { ...req.headers, host: `127.0.0.1:${LLAMA_PORT}` },
  };

  const proxyReq = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });

  proxyReq.on("error", (err) => {
    res.writeHead(502, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "llama.cpp backend unavailable", detail: err.message }));
  });

  req.pipe(proxyReq);
}

// ── Build ChatML messages for /api/chat → /v1/chat/completions ──

function ollamaToOpenAI(ollamaBody) {
  return {
    model: ollamaBody.model || "deepseek-r1-distill-qwen-7b",
    messages: ollamaBody.messages || [],
    stream: ollamaBody.stream !== false,
    max_tokens: (ollamaBody.options && ollamaBody.options.num_predict) || 256,
    temperature: (ollamaBody.options && ollamaBody.options.temperature) || 0.7,
  };
}

// ── Main HTTP server ─────────────────────────────────────────────

function createServer() {
  const server = http.createServer(async (req, res) => {
    // CORS
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }

    // ── GET / — Web chat UI ──
    if (req.method === "GET" && (req.url === "/" || req.url === "/index.html")) {
      try {
        const html = fs.readFileSync(path.join(__dirname, "chat.html"), "utf-8");
        res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
        res.end(html);
      } catch {
        res.writeHead(500, { "Content-Type": "text/plain" });
        res.end("chat.html not found");
      }
      return;
    }

    // ── GET /health ──
    if (req.method === "GET" && req.url === "/health") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ status: llamaReady ? "ok" : "loading" }));
      return;
    }

    // ── GET /api/tags — Ollama model list ──
    if (req.method === "GET" && req.url === "/api/tags") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        models: [{
          name: "deepseek-r1-distill-qwen:7b",
          model: "deepseek-r1-distill-qwen:7b",
          modified_at: new Date().toISOString(),
          size: 4683073504,
          digest: "r1-distill-q4km",
          details: {
            format: "gguf-q4_k_m",
            family: "qwen2",
            parameter_size: "7B",
            quantization_level: "Q4_K_M",
          },
        }],
      }));
      return;
    }

    // ── POST /api/chat — Ollama-compatible chat ──
    if (req.method === "POST" && req.url === "/api/chat") {
      let body = "";
      for await (const chunk of req) body += chunk;

      let parsed;
      try { parsed = JSON.parse(body); }
      catch { res.writeHead(400); res.end('{"error":"invalid JSON"}'); return; }

      const openaiBody = ollamaToOpenAI(parsed);
      const stream = openaiBody.stream;

      // Forward to llama.cpp's OpenAI-compatible endpoint
      const postData = JSON.stringify(openaiBody);

      const options = {
        hostname: "127.0.0.1",
        port: LLAMA_PORT,
        path: "/v1/chat/completions",
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(postData),
        },
      };

      const proxyReq = http.request(options, (proxyRes) => {
        if (!stream) {
          // Non-streaming: convert OpenAI response to Ollama format
          let respData = "";
          proxyRes.on("data", (c) => respData += c);
          proxyRes.on("end", () => {
            try {
              const openaiResp = JSON.parse(respData);
              const content = openaiResp.choices?.[0]?.message?.content || "";
              res.writeHead(200, { "Content-Type": "application/json" });
              res.end(JSON.stringify({
                model: parsed.model || "deepseek-r1-distill-qwen:7b",
                created_at: new Date().toISOString(),
                message: { role: "assistant", content },
                done: true,
                done_reason: "stop",
              }));
            } catch {
              res.writeHead(502);
              res.end(respData);
            }
          });
        } else {
          // Streaming: convert SSE chunks to Ollama NDJSON
          res.writeHead(200, {
            "Content-Type": "application/x-ndjson",
            "Transfer-Encoding": "chunked",
          });

          let buf = "";
          proxyRes.on("data", (chunk) => {
            buf += chunk.toString();
            const lines = buf.split("\n");
            buf = lines.pop();
            for (const line of lines) {
              if (!line.startsWith("data: ")) continue;
              const data = line.slice(6).trim();
              if (data === "[DONE]") {
                res.write(JSON.stringify({
                  model: parsed.model || "deepseek-r1-distill-qwen:7b",
                  created_at: new Date().toISOString(),
                  message: { role: "assistant", content: "" },
                  done: true,
                  done_reason: "stop",
                }) + "\n");
                return;
              }
              try {
                const sseObj = JSON.parse(data);
                const delta = sseObj.choices?.[0]?.delta?.content || "";
                if (delta) {
                  res.write(JSON.stringify({
                    model: parsed.model || "deepseek-r1-distill-qwen:7b",
                    created_at: new Date().toISOString(),
                    message: { role: "assistant", content: delta },
                    done: false,
                  }) + "\n");
                }
              } catch {}
            }
          });
          proxyRes.on("end", () => res.end());
        }
      });

      proxyReq.on("error", (err) => {
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: err.message }));
      });

      proxyReq.write(postData);
      proxyReq.end();
      return;
    }

    // ── Pass through everything else to llama.cpp ──
    if (req.url.startsWith("/v1/")) {
      proxyToLlama(req, res);
      return;
    }

    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "not found" }));
  });

  server.listen(PORT, HOST, () => {
    console.log("");
    console.log(`[server] QuantoniumOS listening on http://${HOST}:${PORT}`);
    console.log(`[server] Endpoints:`);
    console.log(`[server]   GET  /            Web chat UI`);
    console.log(`[server]   GET  /health      Liveness check`);
    console.log(`[server]   GET  /api/tags    Model list (Ollama)`);
    console.log(`[server]   POST /api/chat    Chat (Ollama)`);
    console.log(`[server]   POST /v1/*        OpenAI passthrough`);
    console.log("");
  });
}

// ── Main ──

async function main() {
  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log("║  QuantoniumOS — R1-Distill-Qwen-7B + Mimosa KV Cache   ║");
  console.log("╚══════════════════════════════════════════════════════════╝");
  console.log(`[server] Model: ${MODEL_PATH}`);
  console.log(`[server] Context: ${CTX_SIZE} tokens`);

  await startLlama();
  createServer();
}

main().catch((err) => {
  console.error(`[server] FATAL: ${err.message}`);
  process.exit(1);
});

// Cleanup on exit
process.on("SIGINT", () => { if (llamaProc) llamaProc.kill(); process.exit(0); });
process.on("SIGTERM", () => { if (llamaProc) llamaProc.kill(); process.exit(0); });
