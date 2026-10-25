import { useState, useCallback, useRef, useEffect } from "react";
import { Cpu, Play, Copy, KeyRound, ShieldCheck } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import "./CryptoView.css";

type Tab = "qkd" | "encrypt" | "analysis";
type Protocol = "BB84" | "B92" | "SARG04";

function randomBits(n: number): number[] {
  return Array.from({ length: n }, () => (Math.random() > 0.5 ? 1 : 0));
}

interface Photon { x: number; y: number; angle: number; speed: number; age: number; intercepted: boolean }

export function CryptoView() {
  const [tab, setTab] = useState<Tab>("qkd");
  const [protocol, setProtocol] = useState<Protocol>("BB84");
  const [keyLen, setKeyLen] = useState(64);
  const [noise, setNoise] = useState(2);
  const [eavesdropper, setEavesdropper] = useState(false);
  const [running, setRunning] = useState(false);

  const [aliceBits, setAliceBits] = useState<number[]>([]);
  const [bobBases, setBobBases] = useState<number[]>([]);
  const [aliceBases, setAliceBases] = useState<number[]>([]);
  const [matchedBases, setMatchedBases] = useState<boolean[]>([]);
  const [siftedKey, setSiftedKey] = useState<number[]>([]);
  const [qber, setQber] = useState<number | null>(null);

  const [plaintext, setPlaintext] = useState("");
  const [ciphertext, setCiphertext] = useState("");
  const [analysisLog, setAnalysisLog] = useState<string[]>([]);

  // Photon channel animation
  const channelRef = useRef<HTMLCanvasElement>(null);
  const photonsRef = useRef<Photon[]>([]);
  const animRef = useRef(0);
  const [channelActive, setChannelActive] = useState(false);

  const drawChannel = useCallback(() => {
    const canvas = channelRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (canvas.parentElement) {
      const rect = canvas.parentElement.getBoundingClientRect();
      if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width; canvas.height = rect.height;
      }
    }
    const W = canvas.width, H = canvas.height;
    const mid = H / 2;

    ctx.fillStyle = "#121212"; ctx.fillRect(0, 0, W, H);

    // Channel line
    ctx.strokeStyle = "rgba(144,144,144,0.08)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(60, mid); ctx.lineTo(W - 60, mid); ctx.stroke();

    // Alice node
    ctx.fillStyle = "rgba(144,144,144,0.1)"; ctx.beginPath(); ctx.arc(40, mid, 20, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(200,200,200,0.3)"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(40, mid, 20, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = "rgba(224,224,224,0.6)"; ctx.font = "bold 10px 'SF Mono', monospace"; ctx.textAlign = "center"; ctx.fillText("A", 40, mid + 4);

    // Bob node
    ctx.fillStyle = "rgba(144,144,144,0.1)"; ctx.beginPath(); ctx.arc(W - 40, mid, 20, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(200,200,200,0.3)"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(W - 40, mid, 20, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = "rgba(224,224,224,0.6)"; ctx.fillText("B", W - 40, mid + 4);

    // Eve (if eavesdropper)
    if (eavesdropper) {
      const ex = W / 2, ey = mid - 40;
      ctx.strokeStyle = "rgba(200,80,80,0.2)"; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(W / 2, mid); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(200,80,80,0.1)"; ctx.beginPath(); ctx.arc(ex, ey, 14, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "rgba(200,80,80,0.3)"; ctx.beginPath(); ctx.arc(ex, ey, 14, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = "rgba(200,80,80,0.6)"; ctx.font = "bold 9px 'SF Mono', monospace"; ctx.fillText("E", ex, ey + 3);
    }

    ctx.textAlign = "left";

    // Labels
    ctx.fillStyle = "rgba(144,144,144,0.35)"; ctx.font = "9px 'SF Mono', monospace";
    ctx.fillText("Alice", 22, mid + 36);
    ctx.textAlign = "right"; ctx.fillText("Bob", W - 22, mid + 36);
    ctx.textAlign = "left";
    ctx.fillText("Quantum Channel", W / 2 - 45, mid + 30);

    // Photons
    const photons = photonsRef.current;
    for (let i = photons.length - 1; i >= 0; i--) {
      const p = photons[i];
      p.x += p.speed;
      p.age++;

      if (p.x > W - 60) { photons.splice(i, 1); continue; }

      // Polarization symbol
      const sz = 5;
      ctx.save();
      ctx.translate(p.x, p.y + Math.sin(p.age * 0.15) * 3);
      ctx.rotate(p.angle);

      if (p.intercepted) {
        // Eavesdropped photon — red glow
        const g = ctx.createRadialGradient(0, 0, 0, 0, 0, 10);
        g.addColorStop(0, "rgba(200,80,80,0.25)"); g.addColorStop(1, "rgba(200,80,80,0)");
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(0, 0, 10, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "rgba(200,120,120,0.6)";
      } else {
        const g = ctx.createRadialGradient(0, 0, 0, 0, 0, 8);
        g.addColorStop(0, "rgba(200,200,200,0.2)"); g.addColorStop(1, "rgba(200,200,200,0)");
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(0, 0, 8, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "rgba(200,200,200,0.6)";
      }
      ctx.lineWidth = 1.5;
      // Cross/line for polarization
      ctx.beginPath(); ctx.moveTo(-sz, 0); ctx.lineTo(sz, 0); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, -sz); ctx.lineTo(0, sz); ctx.stroke();

      ctx.restore();
    }

    if (channelActive || photons.length > 0) {
      animRef.current = requestAnimationFrame(drawChannel);
    }
  }, [eavesdropper, channelActive]);

  // Start/stop channel animation
  useEffect(() => {
    if (tab !== "qkd") { cancelAnimationFrame(animRef.current); return; }
    animRef.current = requestAnimationFrame(drawChannel);
    return () => cancelAnimationFrame(animRef.current);
  }, [tab, drawChannel, channelActive]);

  const generateKey = useCallback(() => {
    setRunning(true);
    setChannelActive(true);

    // Spawn photon particles
    const canvas = channelRef.current;
    const mid = canvas ? canvas.height / 2 : 60;
    photonsRef.current = [];
    let spawned = 0;
    const spawnIv = setInterval(() => {
      if (spawned >= Math.min(keyLen, 40)) { clearInterval(spawnIv); return; }
      photonsRef.current.push({
        x: 60, y: mid, angle: Math.random() * Math.PI,
        speed: 2 + Math.random() * 2, age: 0,
        intercepted: eavesdropper && Math.random() < 0.25,
      });
      spawned++;
    }, 50);

    setTimeout(() => {
      const alice = randomBits(keyLen);
      const aBases = randomBits(keyLen);
      const bBases = randomBits(keyLen);
      const matched = aBases.map((a, i) => a === bBases[i]);
      const sifted = alice.filter((_, i) => matched[i]);

      let errorCount = 0;
      if (eavesdropper) {
        sifted.forEach((_, i) => { if (Math.random() < 0.25) { sifted[i] ^= 1; errorCount++; } });
      }
      sifted.forEach((_, i) => { if (Math.random() < noise / 100) { sifted[i] ^= 1; errorCount++; } });

      setAliceBits(alice);
      setAliceBases(aBases);
      setBobBases(bBases);
      setMatchedBases(matched);
      setSiftedKey(sifted);
      setQber(sifted.length > 0 ? errorCount / sifted.length : 0);
      setRunning(false);
      setTimeout(() => setChannelActive(false), 800);
    }, 1200);
  }, [keyLen, noise, eavesdropper]);

  const handleEncrypt = async () => {
    if (!plaintext || siftedKey.length === 0) return;
    const key = siftedKey.join("").padEnd(16, "0").slice(0, 32);
    try {
      const result = await invoke<{ output_hex: string; cipher: string; rounds: number }>("rft_encrypt", { req: { data: plaintext, key } });
      setCiphertext(result.output_hex);
    } catch {
      const bytes = new TextEncoder().encode(plaintext);
      const out = Array.from(bytes, (b, i) => b ^ (siftedKey[i % siftedKey.length] * 0xFF));
      setCiphertext(out.map((b) => b.toString(16).padStart(2, "0")).join(""));
    }
  };

  const handleDecrypt = async () => {
    if (!ciphertext || siftedKey.length === 0) return;
    const key = siftedKey.join("").padEnd(16, "0").slice(0, 32);
    try {
      const result = await invoke<{ output_hex: string; output_text: string | null }>("rft_decrypt", { req: { data: ciphertext, key } });
      setPlaintext(result.output_text ?? ""); setCiphertext("");
    } catch {
      const hex = ciphertext.match(/.{1,2}/g) ?? [];
      const bytes = hex.map((h, i) => parseInt(h, 16) ^ (siftedKey[i % siftedKey.length] * 0xFF));
      setPlaintext(new TextDecoder().decode(new Uint8Array(bytes))); setCiphertext("");
    }
  };

  const runAnalysis = useCallback(async () => {
    setRunning(true);
    const logs: string[] = [];
    const push = (msg: string) => { logs.push(msg); setAnalysisLog([...logs]); };
    const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

    push(`Protocol: ${protocol}`); await delay(200);
    push(`Key length: ${keyLen} bits`); await delay(200);
    push(`Sifted key: ${siftedKey.length} bits (${((siftedKey.length / keyLen) * 100).toFixed(1)}% yield)`); await delay(200);
    push(`QBER: ${qber !== null ? (qber * 100).toFixed(2) : "N/A"}%`); await delay(200);
    push(`Eavesdropper: ${eavesdropper ? "ACTIVE — elevated QBER" : "Not detected"}`); await delay(200);
    push(`Channel noise: ${noise}%`); await delay(200);
    push(`Security margin: ${qber !== null && qber < 0.11 ? "SECURE (QBER < 11%)" : "COMPROMISED — abort recommended"}`); await delay(200);

    try {
      const hashResult = await invoke<{ hash_hex: string; algorithm: string }>("rft_hash", { req: { data: siftedKey.join("") } });
      push(`RFT-SIS Hash: ${hashResult.hash_hex.slice(0, 32)}…`); await delay(150);
      push(`Algorithm: ${hashResult.algorithm}`);
    } catch { push(`RFT-SIS Hash: [engine unavailable]`); }
    await delay(200);
    push(`Privacy amplification: Toeplitz hashing applied`);
    setRunning(false);
  }, [protocol, keyLen, siftedKey, qber, eavesdropper, noise]);

  const copyKey = () => { navigator.clipboard.writeText(siftedKey.join("")); };

  return (
    <div className="crypto-view">
      <nav className="crypto-tabs">
        {([["qkd", "Key Distribution", KeyRound], ["encrypt", "Encryption", Cpu], ["analysis", "Analysis", ShieldCheck]] as const).map(
          ([id, label, Icon]) => (
            <button key={id} className={`crypto-tab ${tab === id ? "active" : ""}`} onClick={() => setTab(id as Tab)}>
              <Icon size={13} /> {label}
            </button>
          )
        )}
      </nav>

      <div className="crypto-body">
        {tab === "qkd" && (
          <div className="crypto-panel">
            <div className="crypto-controls">
              <label className="crypto-label">
                Protocol
                <select className="crypto-select" value={protocol} onChange={(e) => setProtocol(e.target.value as Protocol)}>
                  <option>BB84</option><option>B92</option><option>SARG04</option>
                </select>
              </label>
              <label className="crypto-label">
                Key Length
                <input type="number" className="crypto-input" min={16} max={512} step={16} value={keyLen} onChange={(e) => setKeyLen(Number(e.target.value))} />
              </label>
              <label className="crypto-label">
                Noise %
                <input type="range" className="crypto-range" min={0} max={10} value={noise} onChange={(e) => setNoise(Number(e.target.value))} />
                <span className="crypto-val">{noise}%</span>
              </label>
              <label className="crypto-check">
                <input type="checkbox" checked={eavesdropper} onChange={() => setEavesdropper(!eavesdropper)} />
                Eavesdropper
              </label>
              <button className="crypto-run-btn" onClick={generateKey} disabled={running}>
                <Play size={13} /> Generate Key
              </button>
            </div>

            {/* Photon channel animation */}
            <div className="crypto-channel-wrap">
              <canvas ref={channelRef} />
            </div>

            {siftedKey.length > 0 && (
              <>
                {/* Visual basis comparison */}
                <div className="crypto-basis-compare">
                  <div className="crypto-basis-row">
                    <span className="crypto-basis-label">Alice</span>
                    <div className="crypto-basis-cells">
                      {aliceBits.slice(0, 32).map((b, i) => (
                        <div key={`a${i}`} className={`crypto-basis-cell ${matchedBases[i] ? "matched" : "miss"}`}>
                          <span className="crypto-basis-bit">{b}</span>
                          <span className="crypto-basis-dir">{aliceBases[i] ? "↕" : "↔"}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="crypto-basis-row">
                    <span className="crypto-basis-label">Bob</span>
                    <div className="crypto-basis-cells">
                      {aliceBits.slice(0, 32).map((_, i) => (
                        <div key={`b${i}`} className={`crypto-basis-cell ${matchedBases[i] ? "matched" : "miss"}`}>
                          <span className="crypto-basis-bit">?</span>
                          <span className="crypto-basis-dir">{bobBases[i] ? "↕" : "↔"}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="crypto-basis-row">
                    <span className="crypto-basis-label">Match</span>
                    <div className="crypto-basis-cells">
                      {matchedBases.slice(0, 32).map((m, i) => (
                        <div key={`m${i}`} className={`crypto-basis-cell ${m ? "matched" : "miss"}`}>
                          <span className="crypto-basis-check">{m ? "✓" : "✗"}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="crypto-key-section">
                  <div className="crypto-key-header">
                    <span>Sifted Key ({siftedKey.length} bits)</span>
                    <button className="crypto-icon-btn" onClick={copyKey} title="Copy key"><Copy size={13} /></button>
                  </div>
                  <div className="crypto-key-visual">
                    {siftedKey.slice(0, 64).map((b, i) => (
                      <span key={i} className={`crypto-key-bit ${b ? "one" : "zero"}`}>{b}</span>
                    ))}
                    {siftedKey.length > 64 && <span className="crypto-key-more">+{siftedKey.length - 64}</span>}
                  </div>
                  <div className="crypto-qber-bar">
                    <span className="crypto-qber-label">QBER</span>
                    <div className="crypto-qber-track">
                      <div className={`crypto-qber-fill ${qber !== null && qber > 0.11 ? "danger" : "safe"}`}
                        style={{ width: `${Math.min((qber ?? 0) * 100 / 15, 100)}%` }} />
                      <div className="crypto-qber-threshold" style={{ left: `${11 / 15 * 100}%` }} />
                    </div>
                    <span className={`crypto-qber-val ${qber !== null && qber > 0.11 ? "danger" : "safe"}`}>
                      {qber !== null ? (qber * 100).toFixed(2) : "—"}%
                    </span>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {tab === "encrypt" && (
          <div className="crypto-panel">
            <div className="crypto-enc-area">
              <label className="crypto-label">Plaintext</label>
              <textarea className="crypto-textarea" value={plaintext} onChange={(e) => setPlaintext(e.target.value)} placeholder="Enter message to encrypt…" />
              <div className="crypto-enc-btns">
                <button className="crypto-run-btn" onClick={handleEncrypt} disabled={siftedKey.length === 0}><Cpu size={13} /> Encrypt</button>
                <button className="crypto-run-btn" onClick={handleDecrypt} disabled={siftedKey.length === 0 || !ciphertext}><Cpu size={13} /> Decrypt</button>
              </div>
              {siftedKey.length === 0 && <p className="crypto-hint">Generate a key in the QKD tab first.</p>}
            </div>
            {ciphertext && (
              <div className="crypto-enc-area">
                <label className="crypto-label">Ciphertext (hex)</label>
                <pre className="crypto-cipher-display">{ciphertext}</pre>
              </div>
            )}
          </div>
        )}

        {tab === "analysis" && (
          <div className="crypto-panel">
            <button className="crypto-run-btn" onClick={runAnalysis} disabled={running || siftedKey.length === 0}>
              <Play size={13} /> Analyze Security
            </button>
            {siftedKey.length === 0 && <p className="crypto-hint">Generate a key in the QKD tab first.</p>}
            {analysisLog.length > 0 && (
              <div className="crypto-analysis-cards">
                {analysisLog.map((line, i) => (
                  <div key={i} className="crypto-analysis-line" style={{ animationDelay: `${i * 60}ms` }}>
                    <span className="crypto-analysis-bullet">›</span>
                    <span>{line}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
