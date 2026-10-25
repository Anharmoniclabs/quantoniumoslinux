import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./BootScreen.css";

interface BootScreenProps {
  onReady: () => void;
}

interface OllamaStatus {
  running: boolean;
  model: string;
  models: string[];
  backend: string;
}

interface RustEngineStatus {
  available_7b: boolean;
  available_35b: boolean;
  backend: string;
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatRustStatus(status: RustEngineStatus): string {
  if (status.available_35b) {
    return "Rust 35B pack ready";
  }
  if (status.available_7b) {
    return "Rust local pack ready";
  }
  return "Loading local model packs";
}

function formatRustInitResult(result: string): string {
  if (/qwen35-opus-distil/i.test(result)) {
    return "Rust 35B pack ready";
  }
  if (/qwen2\.5-coder_3b|deepseek-r1-7b/i.test(result)) {
    return "Rust local pack ready";
  }
  return "Rust engine ready";
}

export function BootScreen({ onReady }: BootScreenProps) {
  const [stage, setStage] = useState("loading");
  const [bootStage, setBootStage] = useState("Starting local runtimes");
  const [substage, setSubstage] = useState("Wiring Ollama and local model packs");
  const [dots, setDots] = useState("");
  const [modelName, setModelName] = useState("");
  const [fadeOut, setFadeOut] = useState(false);

  // Animate dots
  useEffect(() => {
    const iv = setInterval(() => {
      setDots((d) => (d.length >= 3 ? "" : d + "."));
    }, 400);
    return () => clearInterval(iv);
  }, []);

  // Wait for Ollama and warm the Rust pack engine before leaving boot.
  useEffect(() => {
    if (stage !== "loading") return;

    let cancelled = false;
    const waitForOllama = async () => {
      const startTime = Date.now();
      setBootStage("Connecting to Ollama");

      while (!cancelled) {
        try {
          const status = await invoke<OllamaStatus>("ollama_status");
          if (status.running && status.model) {
            setModelName(status.model);
            return;
          }
        } catch {
          // Rust command surface may still be starting.
        }

        if (Date.now() - startTime > 20000) {
          setModelName("eldrae:smartweights");
          setSubstage("Ollama startup is slow; continuing with engine warmup");
          return;
        }

        await delay(800);
      }
    };

    const warmRustEngine = async () => {
      setSubstage("Loading local model packs");

      try {
        const status = await invoke<RustEngineStatus>("rust_engine_status");
        if (status.available_7b || status.available_35b) {
          setSubstage(formatRustStatus(status));
          return;
        }

        const result = await invoke<string>("rust_engine_init");
        setSubstage(formatRustInitResult(result));
      } catch {
        setSubstage("Rust pack unavailable, using Ollama fallback");
      }
    };

    const boot = async () => {
      await Promise.allSettled([waitForOllama(), warmRustEngine()]);
      if (!cancelled) {
        setStage("ready");
      }
    };

    boot();
    return () => { cancelled = true; };
  }, [stage]);

  // When ready, fade out and transition
  useEffect(() => {
    if (stage !== "ready") return;
    const t1 = setTimeout(() => setFadeOut(true), 600);
    const t2 = setTimeout(() => onReady(), 1200);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [stage, onReady]);

  const displayName = modelName
    .replace(/-/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    || "Eldrae SmartWeights";

  return (
    <div className={`boot-screen ${fadeOut ? "fade-out" : ""}`}>
      <div className="boot-content">
        {/* Q Logo */}
        <div className="boot-logo">
          <span className="boot-q">Q</span>
        </div>

        {/* Title */}
        <h1 className="boot-title">QuantoniumOS</h1>

        {/* Status */}
        <div className="boot-status">
          {stage === "loading" && (
            <>
              <span className="boot-stage">{bootStage}{dots}</span>
              <div className="boot-progress-track">
                <div className="boot-progress-bar" />
              </div>
              <span className="boot-substage">
                {substage}
              </span>
            </>
          )}
          {stage === "ready" && (
            <>
              <span className="boot-stage boot-ready">
                {displayName}
              </span>
              <span className="boot-substage">{substage || "Engines ready"}</span>
            </>
          )}
        </div>
      </div>

      {/* Bottom branding */}
      <div className="boot-footer">
        <span>Local AI · Private by default</span>
      </div>
    </div>
  );
}
