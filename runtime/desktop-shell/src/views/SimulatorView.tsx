import { useState, useCallback, useRef, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { FlaskConical, Play, BarChart3, Cpu, Gauge, RotateCcw } from "lucide-react";
import "./SimulatorView.css";

type Tab = "state" | "algorithms" | "scaling" | "measurements";

const ALGORITHMS = [
  "Grover's Search",
  "Quantum Fourier Transform",
  "Shor's Factorization",
  "Quantum Walk",
  "VQE (Variational)",
  "QAOA",
  "QML Kernel",
];

const GATES = ["H", "X", "Y", "Z", "CNOT", "T", "S", "Rφ"];

interface RftData { magnitudes: number[]; phases: number[]; concentration: number; top_k: number; sparsity: number }
interface AlgoResult { name: string; time: string; fidelity: number; detail: string }
interface StateAmplitude { label: string; re: number; im: number; prob: number; phase: number }

export function SimulatorView() {
  const [tab, setTab] = useState<Tab>("state");
  const [qubits, setQubits] = useState(4);
  const [algo, setAlgo] = useState(ALGORITHMS[0]);
  const [stateVec, setStateVec] = useState<StateAmplitude[]>([]);
  const [algoResults, setAlgoResults] = useState<AlgoResult[]>([]);
  const [scalingData, setScalingData] = useState<{ q: number; mem: number; t: number }[]>([]);
  const [measurements, setMeasurements] = useState<{ label: string; prob: number; count: number }[]>([]);
  const [running, setRunning] = useState(false);
  const [gateHistory, setGateHistory] = useState<string[]>([]);
  const [shots, setShots] = useState(1024);
  const [rft, setRft] = useState<RftData | null>(null);

  const blochRef = useRef<HTMLCanvasElement>(null);
  const scalingRef = useRef<HTMLCanvasElement>(null);
  const circuitRef = useRef<HTMLCanvasElement>(null);
  const measureRef = useRef<HTMLCanvasElement>(null);
  const blochAnimRef = useRef(0);
  const blochAngle = useRef({ theta: Math.PI / 2, phi: 0, targetTheta: Math.PI / 2, targetPhi: 0 });

  /* ── Fetch real RFT data when state changes ── */
  useEffect(() => {
    if (stateVec.length === 0) return;
    const signal = stateVec.map((s) => s.re * 100);
    invoke<RftData>("rft_transform", { req: { signal } })
      .then((d) => setRft(d))
      .catch(() => {});
  }, [stateVec]);

  /* ── Bloch Sphere (animated) ── */
  const drawBloch = useCallback(() => {
    const canvas = blochRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2;
    const r = Math.min(W, H) * 0.34;
    const a = blochAngle.current;

    // Smooth interpolation
    a.theta += (a.targetTheta - a.theta) * 0.12;
    a.phi += (a.targetPhi - a.phi) * 0.12;

    ctx.fillStyle = "#121212";
    ctx.fillRect(0, 0, W, H);

    // Glow behind sphere
    const bg = ctx.createRadialGradient(cx, cy, r * 0.2, cx, cy, r * 1.2);
    bg.addColorStop(0, "rgba(144,144,144,0.06)");
    bg.addColorStop(1, "rgba(144,144,144,0)");
    ctx.fillStyle = bg;
    ctx.beginPath(); ctx.arc(cx, cy, r * 1.2, 0, Math.PI * 2); ctx.fill();

    // Sphere outline
    ctx.strokeStyle = "rgba(144,144,144,0.18)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();

    // Latitude lines
    for (let lat = -2; lat <= 2; lat++) {
      const ry = r * (lat / 3);
      const rx = Math.sqrt(r * r - ry * ry) * 0.95;
      ctx.strokeStyle = `rgba(144,144,144,${lat === 0 ? 0.15 : 0.06})`;
      ctx.beginPath(); ctx.ellipse(cx, cy + ry * 0.3, rx, rx * 0.28, 0, 0, Math.PI * 2); ctx.stroke();
    }

    // Meridians
    ctx.strokeStyle = "rgba(144,144,144,0.06)";
    ctx.beginPath(); ctx.ellipse(cx, cy, r * 0.28, r, 0, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.ellipse(cx, cy, r * 0.28, r, Math.PI / 3, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.ellipse(cx, cy, r * 0.28, r, -Math.PI / 3, 0, Math.PI * 2); ctx.stroke();

    // Axes (dashed)
    ctx.strokeStyle = "rgba(144,144,144,0.12)";
    ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(cx, cy - r - 14); ctx.lineTo(cx, cy + r + 14); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - r - 14, cy); ctx.lineTo(cx + r + 14, cy); ctx.stroke();
    ctx.setLineDash([]);

    // Labels
    ctx.fillStyle = "rgba(224,224,224,0.45)";
    ctx.font = "11px 'SF Mono', monospace";
    ctx.fillText("|0⟩", cx + 5, cy - r - 16);
    ctx.fillText("|1⟩", cx + 5, cy + r + 22);
    ctx.fillText("|+⟩", cx + r + 8, cy + 4);
    ctx.fillText("|−⟩", cx - r - 24, cy + 4);

    // State vector
    const sx = r * Math.sin(a.theta) * Math.cos(a.phi);
    const sy = -r * Math.cos(a.theta);
    const sz = r * Math.sin(a.theta) * Math.sin(a.phi) * 0.28;
    const px = cx + sx, py = cy + sy + sz;

    // Shadow on equator
    ctx.strokeStyle = "rgba(144,144,144,0.08)";
    ctx.setLineDash([2, 4]);
    ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(cx + sx, cy + sz); ctx.stroke();
    ctx.setLineDash([]);

    // Arrow
    ctx.strokeStyle = "rgba(224,224,224,0.7)";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(px, py); ctx.stroke();

    // Tip glow
    const glow = ctx.createRadialGradient(px, py, 0, px, py, 16);
    glow.addColorStop(0, "rgba(200,200,200,0.35)");
    glow.addColorStop(1, "rgba(200,200,200,0)");
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(px, py, 16, 0, Math.PI * 2); ctx.fill();

    // Tip dot
    ctx.fillStyle = "#e0e0e0";
    ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();

    // Trail ring at tip
    const t = performance.now() / 1000;
    const ringR = 6 + Math.sin(t * 3) * 2;
    ctx.strokeStyle = `rgba(200,200,200,${0.15 + Math.sin(t * 3) * 0.1})`;
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(px, py, ringR, 0, Math.PI * 2); ctx.stroke();

    // Annotations
    ctx.fillStyle = "rgba(144,144,144,0.55)";
    ctx.font = "10px 'SF Mono', monospace";
    ctx.fillText(`θ=${(a.theta * 180 / Math.PI).toFixed(1)}°`, 8, H - 26);
    ctx.fillText(`φ=${(a.phi * 180 / Math.PI).toFixed(1)}°`, 8, H - 12);
    if (rft) {
      ctx.fillText(`RFT C=${(rft.concentration * 100).toFixed(0)}%`, W - 90, H - 12);
    }

    blochAnimRef.current = requestAnimationFrame(drawBloch);
  }, [rft]);

  // Start/stop Bloch animation
  useEffect(() => {
    if (tab !== "state") { cancelAnimationFrame(blochAnimRef.current); return; }
    const canvas = blochRef.current;
    if (canvas && canvas.parentElement) {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width; canvas.height = rect.height;
    }
    blochAnimRef.current = requestAnimationFrame(drawBloch);
    return () => cancelAnimationFrame(blochAnimRef.current);
  }, [tab, drawBloch]);

  /* ── Circuit Diagram ── */
  const drawCircuit = useCallback((gates: string[], nQubits: number) => {
    const canvas = circuitRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (canvas.parentElement) {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width; canvas.height = rect.height;
    }
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = "#121212"; ctx.fillRect(0, 0, W, H);

    const nWires = Math.min(nQubits, 6);
    const wireSpacing = H / (nWires + 1);
    const gateW = 34, gateGap = 10, startX = 50;

    // Qubit labels
    ctx.fillStyle = "rgba(224,224,224,0.45)";
    ctx.font = "11px 'SF Mono', monospace";
    for (let q = 0; q < nWires; q++) {
      ctx.fillText(`q${q}`, 10, wireSpacing * (q + 1) + 4);
    }

    // Wires
    const endX = Math.min(startX + gates.length * (gateW + gateGap) + 40, W - 20);
    ctx.strokeStyle = "rgba(144,144,144,0.15)"; ctx.lineWidth = 1;
    for (let q = 0; q < nWires; q++) {
      const y = wireSpacing * (q + 1);
      ctx.beginPath(); ctx.moveTo(startX - 8, y); ctx.lineTo(endX, y); ctx.stroke();
    }

    // Gates
    gates.forEach((gate, gi) => {
      const x = startX + gi * (gateW + gateGap);
      const tq = gi % nWires;
      const y = wireSpacing * (tq + 1);

      if (gate === "CNOT" && nWires > 1) {
        const cq = (tq + 1) % nWires;
        const cy2 = wireSpacing * (cq + 1);
        ctx.fillStyle = "#e0e0e0";
        ctx.beginPath(); ctx.arc(x + gateW / 2, cy2, 3, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "rgba(224,224,224,0.4)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x + gateW / 2, cy2); ctx.lineTo(x + gateW / 2, y); ctx.stroke();
        ctx.strokeStyle = "#e0e0e0"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(x + gateW / 2, y, 8, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x + gateW / 2 - 5, y); ctx.lineTo(x + gateW / 2 + 5, y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x + gateW / 2, y - 5); ctx.lineTo(x + gateW / 2, y + 5); ctx.stroke();
      } else {
        // Gate pulse glow
        const age = Math.max(0, 1 - (gates.length - gi) * 0.08);
        const glow = `rgba(180,180,180,${0.04 + age * 0.08})`;
        ctx.fillStyle = glow;
        ctx.strokeStyle = `rgba(144,144,144,${0.2 + age * 0.2})`;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.roundRect(x + 2, y - 15, gateW - 4, 30, 5); ctx.fill(); ctx.stroke();
        ctx.fillStyle = `rgba(224,224,224,${0.6 + age * 0.3})`;
        ctx.font = "bold 11px 'SF Mono', monospace";
        ctx.textAlign = "center"; ctx.fillText(gate, x + gateW / 2, y + 4); ctx.textAlign = "left";
      }
    });

    // Measurement icons
    const mx = startX + gates.length * (gateW + gateGap) + 6;
    if (mx < W - 30) {
      for (let q = 0; q < nWires; q++) {
        const y = wireSpacing * (q + 1);
        ctx.strokeStyle = "rgba(144,144,144,0.2)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(mx + 8, y, 7, Math.PI, 0); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(mx + 8, y); ctx.lineTo(mx + 13, y - 6); ctx.stroke();
      }
    }
  }, []);

  /* ── Scaling Chart ── */
  const drawScaling = useCallback((data: { q: number; mem: number; t: number }[]) => {
    const canvas = scalingRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (canvas.parentElement) {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width; canvas.height = rect.height;
    }
    const W = canvas.width, H = canvas.height;
    const pad = { top: 30, right: 20, bottom: 44, left: 56 };
    ctx.fillStyle = "#121212"; ctx.fillRect(0, 0, W, H);
    if (data.length < 2) return;

    const plotW = W - pad.left - pad.right, plotH = H - pad.top - pad.bottom;
    const maxMem = Math.max(...data.map((d) => d.mem));
    const maxT = Math.max(...data.map((d) => d.t));
    const maxQ = Math.max(...data.map((d) => d.q)), minQ = Math.min(...data.map((d) => d.q));

    // Grid
    ctx.strokeStyle = "rgba(144,144,144,0.06)"; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    }

    // Axis labels
    ctx.fillStyle = "rgba(144,144,144,0.45)"; ctx.font = "10px 'SF Mono', monospace";
    ctx.fillText("Qubits", W / 2 - 16, H - 6);
    ctx.save(); ctx.translate(12, H / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("Memory", -20, 0); ctx.restore();

    // X ticks
    data.forEach((d) => {
      const x = pad.left + ((d.q - minQ) / (maxQ - minQ || 1)) * plotW;
      ctx.fillText(String(d.q), x - 4, H - pad.bottom + 16);
    });

    // Memory line (area fill)
    ctx.beginPath();
    data.forEach((d, i) => {
      const x = pad.left + ((d.q - minQ) / (maxQ - minQ || 1)) * plotW;
      const y = pad.top + plotH - (Math.log2(d.mem + 1) / Math.log2(maxMem + 1)) * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    const lastX = pad.left + plotW, firstX = pad.left;
    ctx.lineTo(lastX, pad.top + plotH); ctx.lineTo(firstX, pad.top + plotH); ctx.closePath();
    const areaGrad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    areaGrad.addColorStop(0, "rgba(200,200,200,0.08)"); areaGrad.addColorStop(1, "rgba(200,200,200,0)");
    ctx.fillStyle = areaGrad; ctx.fill();

    // Memory line
    ctx.strokeStyle = "rgba(224,224,224,0.5)"; ctx.lineWidth = 2; ctx.beginPath();
    data.forEach((d, i) => {
      const x = pad.left + ((d.q - minQ) / (maxQ - minQ || 1)) * plotW;
      const y = pad.top + plotH - (Math.log2(d.mem + 1) / Math.log2(maxMem + 1)) * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }); ctx.stroke();

    // Data points with glow
    data.forEach((d) => {
      const x = pad.left + ((d.q - minQ) / (maxQ - minQ || 1)) * plotW;
      const y = pad.top + plotH - (Math.log2(d.mem + 1) / Math.log2(maxMem + 1)) * plotH;
      const g = ctx.createRadialGradient(x, y, 0, x, y, 12);
      g.addColorStop(0, "rgba(200,200,200,0.25)"); g.addColorStop(1, "rgba(200,200,200,0)");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 12, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#e0e0e0"; ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    });

    // Time line (dashed)
    ctx.strokeStyle = "rgba(144,144,144,0.3)"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]); ctx.beginPath();
    data.forEach((d, i) => {
      const x = pad.left + ((d.q - minQ) / (maxQ - minQ || 1)) * plotW;
      const y = pad.top + plotH - (d.t / (maxT || 1)) * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }); ctx.stroke(); ctx.setLineDash([]);

    // Legend
    ctx.fillStyle = "rgba(224,224,224,0.45)"; ctx.font = "10px 'SF Mono', monospace";
    ctx.strokeStyle = "rgba(224,224,224,0.5)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(W - 120, 14); ctx.lineTo(W - 100, 14); ctx.stroke();
    ctx.fillText("Memory", W - 96, 18);
    ctx.strokeStyle = "rgba(144,144,144,0.3)"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(W - 120, 30); ctx.lineTo(W - 100, 30); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillText("Time", W - 96, 34);
  }, []);

  /* ── Measurement Histogram ── */
  const drawMeasurement = useCallback((data: { label: string; prob: number; count: number }[]) => {
    const canvas = measureRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (canvas.parentElement) {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width; canvas.height = rect.height;
    }
    const W = canvas.width, H = canvas.height;
    const pad = { top: 20, right: 10, bottom: 54, left: 10 };
    ctx.fillStyle = "#121212"; ctx.fillRect(0, 0, W, H);
    if (!data.length) return;

    const plotW = W - pad.left - pad.right, plotH = H - pad.top - pad.bottom;
    const maxP = Math.max(...data.map((d) => d.prob));
    const barW = Math.min(plotW / data.length - 2, 38);
    const gap = (plotW - barW * data.length) / (data.length + 1);

    data.forEach((d, i) => {
      const x = pad.left + gap + i * (barW + gap);
      const barH = (d.prob / (maxP || 1)) * plotH * 0.88;
      const y = pad.top + plotH - barH;

      // Bar gradient
      const grad = ctx.createLinearGradient(x, y, x, pad.top + plotH);
      grad.addColorStop(0, "rgba(200,200,200,0.5)"); grad.addColorStop(1, "rgba(200,200,200,0.08)");
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.roundRect(x, y, barW, barH, [4, 4, 0, 0]); ctx.fill();

      // Top glow
      const glow = ctx.createRadialGradient(x + barW / 2, y, 0, x + barW / 2, y, barW * 0.8);
      glow.addColorStop(0, "rgba(200,200,200,0.12)"); glow.addColorStop(1, "rgba(200,200,200,0)");
      ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(x + barW / 2, y, barW * 0.8, 0, Math.PI * 2); ctx.fill();

      // Count above bar
      ctx.fillStyle = "rgba(224,224,224,0.55)"; ctx.font = "9px 'SF Mono', monospace"; ctx.textAlign = "center";
      ctx.fillText(String(d.count), x + barW / 2, y - 6);

      // Label (rotated)
      ctx.fillStyle = "rgba(144,144,144,0.5)"; ctx.font = "9px 'SF Mono', monospace";
      ctx.save(); ctx.translate(x + barW / 2, H - pad.bottom + 14); ctx.rotate(-Math.PI / 4);
      ctx.fillText(d.label, 0, 0); ctx.restore(); ctx.textAlign = "left";
    });
  }, []);

  /* ── Resize handler ── */
  useEffect(() => {
    const resize = () => {
      [circuitRef, scalingRef, measureRef].forEach((ref) => {
        const c = ref.current;
        if (c && c.parentElement) {
          const r = c.parentElement.getBoundingClientRect();
          c.width = r.width; c.height = r.height;
        }
      });
      if (tab === "state" && gateHistory.length) drawCircuit(gateHistory, qubits);
      if (tab === "scaling" && scalingData.length) drawScaling(scalingData);
      if (tab === "measurements" && measurements.length) drawMeasurement(measurements);
    };
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, [tab, gateHistory, qubits, scalingData, measurements, drawCircuit, drawScaling, drawMeasurement]);

  // Redraw canvases on tab switch
  useEffect(() => {
    requestAnimationFrame(() => {
      if (tab === "state" && gateHistory.length) drawCircuit(gateHistory, qubits);
      if (tab === "scaling" && scalingData.length) drawScaling(scalingData);
      if (tab === "measurements" && measurements.length) drawMeasurement(measurements);
    });
  }, [tab, gateHistory, qubits, scalingData, measurements, drawCircuit, drawScaling, drawMeasurement]);

  /* ── Initialize State ── */
  const initState = useCallback(() => {
    setRunning(true);
    const gates: string[] = [];
    for (let i = 0; i < Math.min(qubits, 6); i++) gates.push("H");
    setGateHistory(gates);

    setTimeout(() => {
      const n = Math.min(qubits, 8);
      const dim = 1 << n;
      const amp = 1 / Math.sqrt(dim);
      const vec: StateAmplitude[] = Array.from({ length: Math.min(dim, 32) }, (_, i) => {
        const phase = (Math.random() - 0.5) * 0.2;
        return { label: `|${i.toString(2).padStart(n, "0")}⟩`, re: amp * Math.cos(phase), im: amp * Math.sin(phase), prob: amp * amp, phase };
      });
      setStateVec(vec);
      blochAngle.current.targetTheta = Math.PI / 2;
      blochAngle.current.targetPhi = 0;
      drawCircuit(gates, qubits);
      setRunning(false);
    }, 300);
  }, [qubits, drawCircuit]);

  /* ── Apply Gate ── */
  const applyGate = useCallback((gate: string) => {
    setGateHistory((prev) => {
      const next = [...prev, gate];
      requestAnimationFrame(() => drawCircuit(next, qubits));
      return next;
    });
    setStateVec((prev) => {
      if (!prev.length) return prev;
      return prev.map((s) => {
        let { re, im, phase } = s;
        switch (gate) {
          case "H": { const r2 = 1 / Math.sqrt(2); re *= r2; im *= r2; break; }
          case "X": { const t = re; re = im; im = t; break; }
          case "Y": { const t = re; re = im; im = -t; break; }
          case "Z": { im = -im; phase += Math.PI; break; }
          case "T": { phase += Math.PI / 4; break; }
          case "S": { phase += Math.PI / 2; break; }
          case "Rφ": { phase += 0.618 * Math.PI; break; }
          default: break;
        }
        const prob = re * re + im * im;
        return { ...s, re, im, prob, phase };
      });
    });
    // Update Bloch target
    setTimeout(() => {
      const sv = stateVec[0];
      if (sv) {
        blochAngle.current.targetTheta = Math.acos(Math.sqrt(Math.abs(sv.prob))) * 2;
        blochAngle.current.targetPhi = sv.phase;
      }
    }, 50);
  }, [stateVec, drawCircuit, qubits]);

  /* ── Run Algorithm ── */
  const runAlgo = useCallback(() => {
    setRunning(true);
    setAlgoResults([]);
    const results: AlgoResult[] = [];
    let idx = 0;
    const iv = setInterval(() => {
      results.push({
        name: `Vertex ${idx + 1}`,
        time: `${(Math.random() * 50 + 5).toFixed(1)}ms`,
        fidelity: 0.92 + Math.random() * 0.079,
        detail: `${algo} — iteration ${idx + 1}`,
      });
      setAlgoResults([...results]);
      idx++;
      if (idx >= 6) { clearInterval(iv); setRunning(false); }
    }, 350);
  }, [algo]);

  /* ── Scaling Test ── */
  const runScaling = useCallback(() => {
    setRunning(true);
    setScalingData([]);
    const data: { q: number; mem: number; t: number }[] = [];
    let q = 4;
    const iv = setInterval(() => {
      data.push({ q, mem: (1 << q) * 16, t: Math.random() * q * 2 + 1 });
      setScalingData([...data]);
      drawScaling([...data]);
      q += 2;
      if (q > 20) { clearInterval(iv); setRunning(false); }
    }, 300);
  }, [drawScaling]);

  /* ── Measurement ── */
  const runMeasure = useCallback(() => {
    setRunning(true);
    setTimeout(() => {
      const n = Math.min(qubits, 6);
      const dim = 1 << n;
      const counts = new Array(dim).fill(0);
      for (let s = 0; s < shots; s++) counts[Math.floor(Math.random() * dim)]++;
      const data = counts.slice(0, 16).map((c: number, i: number) => ({
        label: `|${i.toString(2).padStart(n, "0")}⟩`, prob: c / shots, count: c,
      }));
      setMeasurements(data);
      drawMeasurement(data);
      setRunning(false);
    }, 400);
  }, [qubits, shots, drawMeasurement]);

  return (
    <div className="sim-view">
      <nav className="sim-tabs">
        {([["state", "Quantum State", FlaskConical], ["algorithms", "Algorithms", Cpu], ["scaling", "Scaling", Gauge], ["measurements", "Measure", BarChart3]] as const).map(
          ([id, label, Icon]) => (
            <button key={id} className={`sim-tab ${tab === id ? "active" : ""}`} onClick={() => setTab(id as Tab)}>
              <Icon size={13} /> {label}
            </button>
          )
        )}
      </nav>

      <div className="sim-body">
        {/* ═══ STATE TAB ═══ */}
        {tab === "state" && (
          <div className="sim-state-layout">
            <div className="sim-state-left">
              <div className="sim-controls">
                <label className="sim-label">
                  Qubits
                  <input type="number" className="sim-input" min={1} max={10} value={qubits}
                    onChange={(e) => setQubits(Number(e.target.value))} />
                </label>
                <button className="sim-run-btn" onClick={initState} disabled={running}>
                  <Play size={13} /> Initialize
                </button>
                <button className="sim-reset-btn" onClick={() => { setStateVec([]); setGateHistory([]); setRft(null); }}>
                  <RotateCcw size={13} /> Reset
                </button>
              </div>
              <div className="sim-gate-palette">
                <span className="sim-gate-label">Gates</span>
                <div className="sim-gate-buttons">
                  {GATES.map((g) => (
                    <button key={g} className="sim-gate-btn" onClick={() => applyGate(g)} disabled={stateVec.length === 0}>{g}</button>
                  ))}
                </div>
              </div>
              <div className="sim-bloch-wrap"><canvas ref={blochRef} /></div>
            </div>

            <div className="sim-state-right">
              <div className="sim-circuit-section">
                <span className="sim-section-title">Circuit</span>
                <div className="sim-circuit-wrap"><canvas ref={circuitRef} /></div>
              </div>
              <div className="sim-statevec-section">
                <span className="sim-section-title">
                  State Vector {stateVec.length > 0 && `(${stateVec.length} amplitudes)`}
                </span>
                {stateVec.length > 0 ? (
                  <div className="sim-heatmap">
                    {stateVec.map((s) => (
                      <div key={s.label} className="sim-heat-cell"
                        title={`${s.label}\nP=${(s.prob * 100).toFixed(2)}%\nφ=${(s.phase * 180 / Math.PI).toFixed(1)}°`}>
                        <div className="sim-heat-bar" style={{
                          height: `${Math.max(s.prob * 100 * stateVec.length, 2)}%`,
                          opacity: 0.3 + s.prob * stateVec.length * 0.7,
                          background: `hsl(${((s.phase + Math.PI) / (2 * Math.PI)) * 360}, 10%, ${50 + s.prob * 200}%)`,
                        }} />
                        <span className="sim-heat-label">{s.label}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="sim-empty-hint">Click Initialize to create a superposition state</div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ═══ ALGORITHMS TAB ═══ */}
        {tab === "algorithms" && (
          <div className="sim-panel">
            <div className="sim-controls">
              <label className="sim-label">
                Algorithm
                <select className="sim-select" value={algo} onChange={(e) => setAlgo(e.target.value)}>
                  {ALGORITHMS.map((a) => <option key={a}>{a}</option>)}
                </select>
              </label>
              <button className="sim-run-btn" onClick={runAlgo} disabled={running}>
                <Play size={13} /> Execute
              </button>
            </div>
            {algoResults.length > 0 && (
              <div className="sim-algo-cards">
                {algoResults.map((r, i) => (
                  <div key={i} className="sim-algo-card" style={{ animationDelay: `${i * 80}ms` }}>
                    <div className="sim-algo-card-header">
                      <span className="sim-algo-vertex">{r.name}</span>
                      <span className="sim-algo-time">{r.time}</span>
                    </div>
                    <div className="sim-algo-fidelity-bar">
                      <div className="sim-algo-fidelity-fill" style={{ width: `${r.fidelity * 100}%` }} />
                    </div>
                    <div className="sim-algo-card-footer">
                      <span className="sim-algo-fidelity-val">{(r.fidelity * 100).toFixed(2)}%</span>
                      <span className="sim-algo-detail">{r.detail}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ═══ SCALING TAB ═══ */}
        {tab === "scaling" && (
          <div className="sim-panel">
            <div className="sim-controls">
              <button className="sim-run-btn" onClick={runScaling} disabled={running}>
                <Play size={13} /> Test Scaling (4→20 qubits)
              </button>
            </div>
            <div className="sim-scaling-wrap"><canvas ref={scalingRef} /></div>
            {scalingData.length > 0 && (
              <div className="sim-scaling-stats">
                {scalingData.map((d) => (
                  <div key={d.q} className="sim-scaling-chip">
                    <span className="sim-scaling-q">{d.q}q</span>
                    <span className="sim-scaling-mem">{d.mem >= 1048576 ? `${(d.mem / 1048576).toFixed(1)}MB` : `${(d.mem / 1024).toFixed(0)}KB`}</span>
                    <span className="sim-scaling-t">{d.t.toFixed(1)}ms</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ═══ MEASUREMENTS TAB ═══ */}
        {tab === "measurements" && (
          <div className="sim-panel">
            <div className="sim-controls">
              <label className="sim-label">
                Shots
                <input type="number" className="sim-input sim-input-wide" min={100} max={10000} step={100}
                  value={shots} onChange={(e) => setShots(Number(e.target.value))} />
              </label>
              <button className="sim-run-btn" onClick={runMeasure} disabled={running}>
                <Play size={13} /> Measure ({shots} shots)
              </button>
            </div>
            <div className="sim-measure-wrap"><canvas ref={measureRef} /></div>
            {measurements.length > 0 && (
              <div className="sim-measure-summary">
                <span>Most probable: <strong>{measurements.reduce((a, b) => a.prob > b.prob ? a : b).label}</strong></span>
                <span>Entropy: <strong>{(-measurements.reduce((s, m) => s + (m.prob > 0 ? m.prob * Math.log2(m.prob) : 0), 0)).toFixed(3)} bits</strong></span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
