import { useRef, useState } from "react";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import "./ImageGenView.css";

const STYLES = ["pixel", "photo", "anime", "sketch", "oil_paint", "neon", "abstract"] as const;
type Style = typeof STYLES[number];

const RESOLUTIONS = [
  { label: "512", width: 512, height: 512 },
  { label: "768", width: 768, height: 768 },
  { label: "1024", width: 1024, height: 1024 },
] as const;

const QUALITY_MODES = ["draft", "balanced", "ultra"] as const;
type QualityMode = typeof QUALITY_MODES[number];

// Resonart is the only image engine. Auto picks diffusion-first with
// procedural fallback; the explicit modes force one path.
const BACKENDS = ["auto", "diffusion", "procedural"] as const;
type BackendMode = typeof BACKENDS[number];

interface GenResult {
  success: boolean;
  path: string;
  prompt: string;
  style: string;
  backend: string;
  seed: number;
  width: number;
  height: number;
  error: string;
}

export function ImageGenView() {
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("blurry, low resolution, watermark, distorted");
  const [style, setStyle] = useState<Style>("photo");
  const [steps, setSteps] = useState(20);
  const [resolution, setResolution] = useState<(typeof RESOLUTIONS)[number]>(RESOLUTIONS[2]);
  const [quality, setQuality] = useState<QualityMode>("balanced");
  const [backend, setBackend] = useState<BackendMode>("auto");
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState<GenResult | null>(null);
  const [history, setHistory] = useState<GenResult[]>([]);
  const [error, setError] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  async function handleGenerate() {
    if (!prompt.trim() || generating) return;
    setGenerating(true);
    setError("");
    setResult(null);

    try {
      const res = await invoke<GenResult>("generate_image", {
        req: {
          prompt: prompt.trim(),
          negative_prompt: negative.trim(),
          steps,
          style,
          width: resolution.width,
          height: resolution.height,
          quality,
          backend,
        },
      });

      if (res.success) {
        setResult(res);
        setHistory((prev) => [res, ...prev].slice(0, 12));
      } else {
        setError(res.error || "Generation failed.");
      }
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setGenerating(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      handleGenerate();
    }
  }

  return (
    <div className="imagegen-root">
      <div className="imagegen-controls">
        <div className="imagegen-label">PROMPT</div>
        <textarea
          ref={textareaRef}
          className="imagegen-textarea"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={"Describe what to generate...\nfor example: neon city skyline at night, rain-slicked streets"}
          rows={4}
        />

        <div className="imagegen-label">NEGATIVE PROMPT</div>
        <input
          className="imagegen-input"
          value={negative}
          onChange={(e) => setNegative(e.target.value)}
          placeholder="Things to avoid..."
        />

        <div className="imagegen-row">
          <div className="imagegen-field">
            <div className="imagegen-label">BACKEND</div>
            <select
              className="imagegen-select"
              value={backend}
              onChange={(e) => setBackend(e.target.value as BackendMode)}
            >
              {BACKENDS.map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </select>
          </div>
          <div className="imagegen-field">
            <div className="imagegen-label">STYLE</div>
            <select
              className="imagegen-select"
              value={style}
              onChange={(e) => setStyle(e.target.value as Style)}
            >
              {STYLES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
          <div className="imagegen-field">
            <div className="imagegen-label">QUALITY</div>
            <select
              className="imagegen-select"
              value={quality}
              onChange={(e) => setQuality(e.target.value as QualityMode)}
            >
              {QUALITY_MODES.map((q) => (
                <option key={q} value={q}>
                  {q}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="imagegen-row">
          <div className="imagegen-field">
            <div className="imagegen-label">RESOLUTION</div>
            <select
              className="imagegen-select"
              value={resolution.label}
              onChange={(e) => {
                const next = RESOLUTIONS.find((r) => r.label === e.target.value) ?? RESOLUTIONS[2];
                setResolution(next);
              }}
            >
              {RESOLUTIONS.map((r) => (
                <option key={r.label} value={r.label}>
                  {r.width}x{r.height}
                </option>
              ))}
            </select>
          </div>
          <div className="imagegen-field">
            <div className="imagegen-label">STEPS: {steps}</div>
            <input
              type="range"
              min={5}
              max={50}
              value={steps}
              onChange={(e) => setSteps(Number(e.target.value))}
              className="imagegen-slider"
            />
          </div>
        </div>

        <button
          className={`imagegen-btn ${generating ? "imagegen-btn--busy" : ""}`}
          onClick={handleGenerate}
          disabled={generating || !prompt.trim()}
        >
          {generating ? (
            <>
              <span className="imagegen-spinner" /> Generating...
            </>
          ) : (
            "Generate Image"
          )}
        </button>

        <div className="imagegen-hint">
          Ctrl+Enter to generate | backend-aware local render | 1024-first on-device render
        </div>

        {error && <div className="imagegen-error">{error}</div>}

        {result && (
          <div className="imagegen-meta">
            <div className="imagegen-meta-row">
              <span className="imagegen-meta-label">Seed</span>
              <span className="imagegen-meta-val">{result.seed}</span>
            </div>
            <div className="imagegen-meta-row">
              <span className="imagegen-meta-label">Size</span>
              <span className="imagegen-meta-val">
                {result.width}x{result.height}
              </span>
            </div>
            <div className="imagegen-meta-row">
              <span className="imagegen-meta-label">Style</span>
              <span className="imagegen-meta-val">{result.style}</span>
            </div>
            <div className="imagegen-meta-row">
              <span className="imagegen-meta-label">Backend</span>
              <span className="imagegen-meta-val">{result.backend}</span>
            </div>
            <div className="imagegen-meta-row">
              <span className="imagegen-meta-label">Path</span>
              <span className="imagegen-meta-val imagegen-path" title={result.path}>
                {result.path.split(/[\\/]/).pop()}
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="imagegen-canvas">
        {generating && (
          <div className="imagegen-generating">
            <div className="imagegen-pulse" />
            <div className="imagegen-gen-label">Synthesizing...</div>
          </div>
        )}
        {!generating && result && (
          <img
            key={result.path}
            src={convertFileSrc(result.path)}
            alt={result.prompt}
            className="imagegen-output"
          />
        )}
        {!generating && !result && (
          <div className="imagegen-placeholder">
            <div className="imagegen-placeholder-icon">IMG</div>
            <div className="imagegen-placeholder-text">Resonart</div>
            <div className="imagegen-placeholder-sub">
              local image engine with procedural and diffusion backends
            </div>
          </div>
        )}

        {history.length > 1 && (
          <div className="imagegen-history">
            {history.slice(1).map((h, i) => (
              <img
                key={i}
                src={convertFileSrc(h.path)}
                alt={h.prompt}
                className="imagegen-thumb"
                title={h.prompt}
                onClick={() => setResult(h)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
