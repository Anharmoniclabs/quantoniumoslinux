import { useState, useEffect, useRef, useCallback } from "react";
import { Cpu, HardDrive, MemoryStick, RefreshCw, Zap } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import "./MonitorView.css";

interface SystemStats {
  cpu_percent: number;
  mem_used_gb: number;
  mem_total_gb: number;
  mem_percent: number;
  disk_used_gb: number;
  disk_total_gb: number;
  disk_percent: number;
  processes: { name: string; cpu: number; mem_mb: number }[];
}

interface EngineStatus {
  active: boolean;
  engines: string[];
  rft_variants: number;
  fft_ready: boolean;
  crypto_ready: boolean;
  compression_ready: boolean;
}

function useSystemStats(intervalMs = 2000) {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval>>();

  useEffect(() => {
    const poll = async () => {
      try {
        const data = await invoke<SystemStats>("get_system_stats");
        setStats(data);
      } catch {
        // Fallback when command is unavailable
        setStats((prev) => prev ?? {
          cpu_percent: 0,
          mem_used_gb: 0, mem_total_gb: 0, mem_percent: 0,
          disk_used_gb: 0, disk_total_gb: 0, disk_percent: 0,
          processes: [],
        });
      }
    };
    poll();
    timerRef.current = setInterval(poll, intervalMs);
    return () => clearInterval(timerRef.current);
  }, [intervalMs]);

  return stats;
}

function ProgressRing({ value, color, size = 80 }: { value: number; color: string; size?: number }) {
  const r = (size - 8) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - Math.min(value, 100) / 100);

  return (
    <svg width={size} height={size} className="progress-ring">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none"
        stroke="var(--surface)" strokeWidth={6} />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none"
        stroke={color} strokeWidth={6} strokeLinecap="round"
        strokeDasharray={circ} strokeDashoffset={offset}
        style={{ transition: "stroke-dashoffset 400ms var(--ease-out)", transform: "rotate(-90deg)", transformOrigin: "center" }} />
      <text x="50%" y="50%" textAnchor="middle" dy="0.35em"
        fill="var(--fg)" fontSize="14" fontWeight="700">
        {Math.round(value)}%
      </text>
    </svg>
  );
}

/* ── Sparkline ── */
function Sparkline({ values, color = "rgba(144,144,144,0.7)" }: { values: number[]; color?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || values.length < 2) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const W = c.width;
    const H = c.height;
    ctx.clearRect(0, 0, W, H);

    // baseline
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, H - 1); ctx.lineTo(W, H - 1); ctx.stroke();

    const step = W / (values.length - 1);
    const pts = values.map((v, i) => ({ x: i * step, y: H - (v / 100) * (H - 4) - 2 }));

    // fill
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    pts.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
    ctx.lineTo(pts[pts.length - 1].x, H);
    ctx.lineTo(pts[0].x, H);
    ctx.closePath();
    ctx.fillStyle = color.replace(/[\d.]+\)$/, "0.08)");
    ctx.fill();

    // line
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    pts.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }, [values, color]);

  return <canvas ref={canvasRef} width={200} height={36} className="sparkline-canvas" />;
}

/* ── Vertex Resonance Graph (φ-modulated sinusoid) ── */
const PHI = 1.618033988749895;
const VERTEX_COUNT = 32;

function VertexGraph() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef(0);
  const tRef = useRef(0);

  const draw = useCallback(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const W = c.width;
    const H = c.height;
    const t = tRef.current;

    ctx.fillStyle = "#0f1216";
    ctx.fillRect(0, 0, W, H);

    // grid
    ctx.strokeStyle = "rgba(255,255,255,0.03)";
    ctx.lineWidth = 1;
    for (let y = 0; y < H; y += 30) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    // compute amplitudes: |sin(ω·(1+0.2·sin(t))) · cos(ω·φ+t)|
    const freqs = Array.from({ length: VERTEX_COUNT }, (_, i) => (i / VERTEX_COUNT) * 2 * Math.PI);
    const amps = freqs.map((w) =>
      Math.abs(Math.sin(w * (1 + 0.2 * Math.sin(t))) * Math.cos(w * PHI + t))
    );
    const maxAmp = Math.max(...amps) || 1;

    const spacing = W / (VERTEX_COUNT - 1);
    const pts = amps.map((a, i) => ({
      x: i * spacing,
      y: H - (a / maxAmp) * (H * 0.7) - H * 0.1,
    }));

    // filled area
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    pts.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
    ctx.lineTo(pts[pts.length - 1].x, H);
    ctx.lineTo(pts[0].x, H);
    ctx.closePath();
    ctx.fillStyle = "rgba(144,144,144,0.12)";
    ctx.fill();

    // line
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    pts.slice(1).forEach((p) => ctx.lineTo(p.x, p.y));
    ctx.strokeStyle = "rgba(180,180,180,0.6)";
    ctx.lineWidth = 2;
    ctx.stroke();

    // vertex dots
    ctx.fillStyle = "rgba(200,200,200,0.5)";
    pts.forEach((p) => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    });

    // labels
    ctx.fillStyle = "rgba(224,224,224,0.4)";
    ctx.font = "10px 'SF Mono', 'Cascadia Code', monospace";
    ctx.fillText("Vertex Resonance (RFT streaming)", 8, 14);
    ctx.fillText(`t = ${t.toFixed(1)}`, W - 60, 14);
  }, []);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const resize = () => {
      const r = c.parentElement!.getBoundingClientRect();
      c.width = r.width;
      c.height = r.height;
      draw();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(c.parentElement!);
    return () => ro.disconnect();
  }, [draw]);

  useEffect(() => {
    const tick = () => {
      tRef.current += 0.06;
      draw();
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  return <canvas ref={canvasRef} className="vertex-canvas" />;
}

export function MonitorView() {
  const stats = useSystemStats();
  const [cpuHistory, setCpuHistory] = useState<number[]>([]);
  const [memHistory, setMemHistory] = useState<number[]>([]);
  const [engineStatus, setEngineStatus] = useState<EngineStatus | null>(null);

  useEffect(() => {
    invoke<EngineStatus>("rft_engine_status")
      .then(setEngineStatus)
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!stats) return;
    setCpuHistory((h) => [...h.slice(-59), stats.cpu_percent]);
    setMemHistory((h) => [...h.slice(-59), stats.mem_percent]);
  }, [stats]);

  return (
    <div className="monitor-view">
      <header className="monitor-header">
        <h1>System Monitor</h1>
        <span className="monitor-live">
          <span className="live-dot" /> Live
        </span>
      </header>

      <div className="monitor-grid">
        {/* CPU */}
        <div className="stat-card">
          <div className="stat-card-header">
            <Cpu size={16} />
            <span>CPU</span>
          </div>
          <ProgressRing value={stats?.cpu_percent ?? 0} color="var(--accent)" />
          <Sparkline values={cpuHistory} color="rgba(144,144,144,0.7)" />
        </div>

        {/* Memory */}
        <div className="stat-card">
          <div className="stat-card-header">
            <MemoryStick size={16} />
            <span>Memory</span>
          </div>
          <ProgressRing value={stats?.mem_percent ?? 0} color="var(--accent)" />
          <span className="stat-detail">
            {stats ? `${stats.mem_used_gb.toFixed(1)} / ${stats.mem_total_gb.toFixed(1)} GB` : "—"}
          </span>
          <Sparkline values={memHistory} color="rgba(144,144,144,0.7)" />
        </div>

        {/* Disk */}
        <div className="stat-card">
          <div className="stat-card-header">
            <HardDrive size={16} />
            <span>Disk</span>
          </div>
          <ProgressRing value={stats?.disk_percent ?? 0} color="var(--accent)" />
          <span className="stat-detail">
            {stats ? `${stats.disk_used_gb.toFixed(0)} / ${stats.disk_total_gb.toFixed(0)} GB` : "—"}
          </span>
        </div>
      </div>

      {/* Vertex Resonance Graph */}
      <div className="vertex-section">
        <div className="vertex-header">
          <Zap size={14} />
          <h2>Vertex Resonance Graph</h2>
          <span className="rft-badge">{engineStatus?.active ? "RFT Online" : "RFT Offline"}</span>
        </div>
        <div className="vertex-canvas-wrap">
          <VertexGraph />
        </div>
      </div>

      {/* RFT Engine Status */}
      {engineStatus && (
        <div className="engine-status-section" style={{ padding: "12px 16px", marginBottom: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <Zap size={13} />
            <span style={{ fontSize: 12, fontWeight: 600, color: "var(--fg)" }}>Rust RFT Engines</span>
            <span style={{
              fontSize: 10, padding: "2px 6px", borderRadius: 4,
              background: engineStatus.active ? "rgba(144,144,144,0.2)" : "rgba(255,80,80,0.2)",
              color: engineStatus.active ? "var(--accent)" : "#f55",
            }}>
              {engineStatus.active ? "ACTIVE" : "OFFLINE"}
            </span>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {engineStatus.engines.map((e) => (
              <span key={e} style={{
                fontSize: 10, padding: "2px 6px", borderRadius: 3,
                background: "rgba(144,144,144,0.1)", color: "var(--fg-dim, #999)",
                fontFamily: "'SF Mono', 'Cascadia Code', monospace",
              }}>{e}</span>
            ))}
          </div>
          <div style={{ fontSize: 10, color: "var(--fg-dim, #777)", marginTop: 6, fontFamily: "'SF Mono', 'Cascadia Code', monospace" }}>
            {engineStatus.rft_variants} RFT variants · FFT {engineStatus.fft_ready ? "✓" : "✗"} · Crypto {engineStatus.crypto_ready ? "✓" : "✗"} · Compression {engineStatus.compression_ready ? "✓" : "✗"}
          </div>
        </div>
      )}

      {/* Process table */}
      <div className="proc-section">
        <div className="proc-header">
          <h2>Processes</h2>
          <RefreshCw size={14} className="spin" />
        </div>
        <div className="proc-table">
          <div className="proc-row proc-row-header">
            <span>Name</span>
            <span>CPU %</span>
            <span>Mem (MB)</span>
          </div>
          {(stats?.processes ?? []).slice(0, 12).map((p, i) => (
            <div key={i} className="proc-row">
              <span className="truncate">{p.name}</span>
              <span>{p.cpu.toFixed(1)}</span>
              <span>{p.mem_mb.toFixed(0)}</span>
            </div>
          ))}
          {(!stats || stats.processes.length === 0) && (
            <div className="proc-row proc-empty">
              <span>Waiting for system data…</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
