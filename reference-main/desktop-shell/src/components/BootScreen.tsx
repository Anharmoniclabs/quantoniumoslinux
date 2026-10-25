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

export function BootScreen({ onReady }: BootScreenProps) {
  const [stage, setStage] = useState("loading");
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

  // Poll Ollama status via Rust middleware
  useEffect(() => {
    if (stage !== "loading") return;

    let cancelled = false;
    const poll = async () => {
      while (!cancelled) {
        try {
          const s = await invoke<OllamaStatus>("ollama_status");
          if (s.running && s.model) {
            setModelName(s.model);
            setStage("ready");
            return;
          }
        } catch {
          // Rust command not ready yet
        }
        await new Promise((r) => setTimeout(r, 800));
      }
    };
    poll();
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
    || "DeepSeek R1 7B";

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
              <span className="boot-stage">Connecting to Ollama{dots}</span>
              <div className="boot-progress-track">
                <div className="boot-progress-bar" />
              </div>
              <span className="boot-substage">
                Rust middleware · Local inference
              </span>
            </>
          )}
          {stage === "ready" && (
            <>
              <span className="boot-stage boot-ready">
                {displayName}
              </span>
              <span className="boot-substage">Engine ready</span>
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
