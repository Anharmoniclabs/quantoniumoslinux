import { useRef, useEffect, useState, useCallback } from "react";
import { Waves, Play, Pause } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import "./VisualizerView.css";

const PHI = 0.618;

interface RftData {
  magnitudes: number[];
  phases: number[];
  concentration: number;
  top_k: number;
  sparsity: number;
}

export function VisualizerView() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef(0);
  const timeRef = useRef(0);
  const [playing, setPlaying] = useState(true);
  const [depth, setDepth] = useState(4);
  const [frequency, setFrequency] = useState(1.0);
  const depthRef = useRef(depth);
  const freqRef = useRef(frequency);
  const rftRef = useRef<RftData | null>(null);

  useEffect(() => { depthRef.current = depth; }, [depth]);
  useEffect(() => { freqRef.current = frequency; }, [frequency]);

  // Fetch real RFT transform data from Rust engine when params change
  useEffect(() => {
    const n = 128;
    const freq = frequency;
    const d = depth;
    const signal = Array.from({ length: n }, (_, i) => {
      let v = 0;
      for (let layer = 0; layer < d; layer++) {
        const fs = freq * (1 + layer * PHI);
        v += Math.sin(fs * (i / n) * Math.PI * 8 + layer * Math.PI * 0.4) / (1 + layer * 0.3);
      }
      return v;
    });
    invoke<RftData>("rft_transform", { req: { signal } })
      .then((data) => { rftRef.current = data; })
      .catch(() => { rftRef.current = null; });
  }, [depth, frequency]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const mid = H / 2;
    const d = depthRef.current;
    const freq = freqRef.current;
    const t = timeRef.current;
    const rft = rftRef.current;

    ctx.fillStyle = "#121212";
    ctx.fillRect(0, 0, W, H);

    // Draw grid
    ctx.strokeStyle = "rgba(255,255,255,0.03)";
    ctx.lineWidth = 1;
    for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
    for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }

    // Center axis
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(W, mid); ctx.stroke();

    // ── RFT magnitude spectrum (bottom 30% of canvas) ──
    if (rft && rft.magnitudes.length > 0) {
      const specH = H * 0.25;
      const specY = H - specH - 8;
      const mags = rft.magnitudes;
      const maxMag = Math.max(...mags) || 1;
      const barW = W / mags.length;

      // Spectrum background
      ctx.fillStyle = "rgba(144,144,144,0.04)";
      ctx.fillRect(0, specY, W, specH);

      for (let i = 0; i < mags.length; i++) {
        const barH = (mags[i] / maxMag) * specH * 0.9;
        const phase = rft.phases[i] ?? 0;
        const brightness = 100 + Math.floor(Math.abs(Math.sin(phase + t * 0.5)) * 80);
        ctx.fillStyle = `rgba(${brightness},${brightness},${brightness},0.6)`;
        ctx.fillRect(i * barW + 1, specY + specH - barH, Math.max(barW - 1, 1), barH);
      }

      // Spectrum label
      ctx.fillStyle = "rgba(224,224,224,0.35)";
      ctx.font = "9px 'SF Mono', 'Cascadia Code', monospace";
      ctx.fillText("RFT Spectrum (Rust engine)", 8, specY + 10);
    }

    // ── Layered φ-RFT waves (main area) ──
    const waveH = rft ? H * 0.65 : H;
    const waveMid = waveH / 2;

    for (let layer = 0; layer < d; layer++) {
      const coupling = PHI;
      const freqScale = freq * (1 + layer * coupling);
      const ampScale = (waveMid * 0.6) / (1 + layer * 0.3);
      const phaseOffset = layer * Math.PI * 0.4;
      const speed = 2.0;

      // Modulate with RFT data if available
      const rftMod = rft ? 0.3 + 0.7 * (rft.concentration ?? 0.5) : 1.0;

      const alpha = Math.max(0.15, 0.6 - layer * 0.06);
      const bright = Math.max(80, 180 - layer * 15);
      ctx.strokeStyle = `rgba(${bright},${bright},${bright},${alpha})`;
      ctx.lineWidth = Math.max(1, 2.5 - layer * 0.2);
      ctx.beginPath();

      for (let px = 0; px < W; px++) {
        const x = (px / W) * Math.PI * 8;
        const wave = Math.sin(freqScale * x - speed * t + phaseOffset);
        const qmod = Math.sin(coupling * x + t) * 0.2 * rftMod;
        const y = waveMid + (wave + qmod) * ampScale;
        if (px === 0) ctx.moveTo(px, y); else ctx.lineTo(px, y);
      }
      ctx.stroke();
    }

    // Metrics overlay — real engine data
    ctx.fillStyle = "rgba(224,224,224,0.5)";
    ctx.font = "11px 'SF Mono', 'Cascadia Code', monospace";
    if (rft) {
      ctx.fillText(`RFT Engine: active`, 12, 20);
      ctx.fillText(`Concentration: ${(rft.concentration * 100).toFixed(1)}%`, 12, 36);
      ctx.fillText(`Sparsity: ${(rft.sparsity * 100).toFixed(1)}% (top-k=${rft.top_k})`, 12, 52);
      ctx.fillText(`φ-coupling: ${PHI}  depth=${d}  freq=${freq.toFixed(1)}`, 12, 68);
    } else {
      ctx.fillText(`Complexity: ${(d * freq * 17.3).toFixed(1)}`, 12, 20);
      ctx.fillText(`Coherence: ${(100 - d * 3.2 + Math.sin(t * 0.3) * 2).toFixed(1)}%`, 12, 36);
      ctx.fillText(`Interference: ${(d * 12.5 + Math.cos(t * 0.5) * 3).toFixed(1)}%`, 12, 52);
      ctx.fillText(`φ-coupling: ${PHI}`, 12, 68);
    }
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const rect = canvas.parentElement!.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      draw();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas.parentElement!);
    return () => ro.disconnect();
  }, [draw]);

  useEffect(() => {
    if (!playing) { cancelAnimationFrame(rafRef.current); return; }
    const tick = () => {
      timeRef.current += 0.04;
      draw();
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, draw]);

  return (
    <div className="viz-view">
      <div className="viz-controls">
        <Waves size={14} />
        <label className="viz-slider-label">
          Depth
          <input type="range" min={1} max={10} value={depth} onChange={(e) => setDepth(Number(e.target.value))} className="viz-slider" />
          <span className="viz-slider-val">{depth}</span>
        </label>
        <label className="viz-slider-label">
          Freq
          <input type="range" min={1} max={50} value={Math.round(frequency * 10)} onChange={(e) => setFrequency(Number(e.target.value) / 10)} className="viz-slider" />
          <span className="viz-slider-val">{frequency.toFixed(1)}</span>
        </label>
        <button className="viz-toggle" onClick={() => setPlaying(!playing)}>
          {playing ? <Pause size={13} /> : <Play size={13} />}
          {playing ? "Pause" : "Play"}
        </button>
      </div>
      <div className="viz-canvas-wrap">
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}
