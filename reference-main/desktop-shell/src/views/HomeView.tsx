import type { AppId as ViewId } from "../hooks/useWindowManager";
import type { LucideIcon } from "lucide-react";
import {
  MessageCircle,
  Activity,
  ShieldCheck,
  StickyNote,
  FlaskConical,
  Lock,
  Waves,
  Cpu,
} from "lucide-react";
import "./HomeView.css";

const APPS: { id: ViewId | string; icon: LucideIcon; label: string; desc: string; color: string }[] = [
  { id: "chat",      icon: MessageCircle, label: "AI Chat",           desc: "Local inference · Agent mode", color: "#00d4ff" },
  { id: "monitor",   icon: Activity,      label: "System Monitor",    desc: "CPU · Memory · Processes",     color: "#34d399" },
  { id: "validator", icon: ShieldCheck,    label: "RFT Validator",     desc: "Bijection · Entropy · Proofs", color: "#fbbf24" },
  { id: "notes",     icon: StickyNote,     label: "Q-Notes",           desc: "Secure local notes",           color: "#a78bfa" },
  { id: "vault",     icon: Lock,           label: "Q-Vault",           desc: "AES-encrypted storage",        color: "#f87171" },
  { id: "simulator", icon: FlaskConical,   label: "Quantum Simulator", desc: "Symbolic circuit modeling",    color: "#60a5fa" },
  { id: "visualizer",icon: Waves,          label: "RFT Visualizer",    desc: "Wave resonance display",       color: "#c084fc" },
  { id: "crypto",    icon: Cpu,            label: "Quantum Crypto",    desc: "QKD protocol simulation",      color: "#fb923c" },
];

interface HomeViewProps {
  onNavigate: (id: ViewId) => void;
}

export function HomeView({ onNavigate }: HomeViewProps) {
  const handleClick = (id: string) => {
    if (["chat", "monitor", "validator", "notes", "settings"].includes(id)) {
      onNavigate(id as ViewId);
    }
    // vault/simulator/visualizer/crypto → will be wired later
  };

  return (
    <div className="home">
      <header className="home-header">
        <img src="/que-icon.png" alt="QuantoniumOS" className="home-logo" />
        <h1 className="home-title">QuantoniumOS</h1>
        <p className="home-subtitle">Local AI inference · Private by default</p>
      </header>

      <div className="home-grid">
        {APPS.map((app) => (
          <button
            key={app.id}
            className="app-card"
            onClick={() => handleClick(app.id)}
            style={{ "--card-accent": app.color } as React.CSSProperties}
          >
            <div className="app-card-icon">
              <app.icon size={24} />
            </div>
            <div className="app-card-info">
              <span className="app-card-label">{app.label}</span>
              <span className="app-card-desc">{app.desc}</span>
            </div>
          </button>
        ))}
      </div>

      <footer className="home-footer">
        <span>RFTMW Engine · quantoniumos v0.1</span>
      </footer>
    </div>
  );
}
