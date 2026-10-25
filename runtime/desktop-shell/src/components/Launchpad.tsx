import { useState } from "react";
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
  Settings,
  Users,
  Image,
} from "lucide-react";
import type { AppId } from "../hooks/useWindowManager";
import "./Launchpad.css";

interface LaunchpadApp {
  id: AppId;
  icon: LucideIcon;
  label: string;
  category: string;
}

const APPS: LaunchpadApp[] = [
  { id: "validator",  icon: ShieldCheck,   label: "RFT Validation",    category: "ANALYSIS" },
  { id: "visualizer", icon: Waves,         label: "RFT Visualizer",    category: "VISUAL" },
  { id: "chat",       icon: MessageCircle, label: "AI Chat",           category: "AI" },
  { id: "guild",      icon: Users,         label: "QUE Guild",         category: "AGENTS" },
  { id: "simulator",  icon: FlaskConical,  label: "Quantum Simulator", category: "SIMULATION" },
  { id: "crypto",     icon: Cpu,           label: "Quantum Crypto",    category: "SECURITY" },
  { id: "monitor",    icon: Activity,      label: "System Monitor",    category: "SYSTEM" },
  { id: "notes",      icon: StickyNote,    label: "Q-Notes",           category: "RESEARCH" },
  { id: "vault",      icon: Lock,          label: "Q-Vault",           category: "DATA" },
  { id: "imagegen",   icon: Image,         label: "Resonart",          category: "AI" },
  { id: "settings",   icon: Settings,      label: "Settings",          category: "SYSTEM" },
];

const ARCH_RADIUS = 260;
const START_ANGLE = Math.PI * 1.75;  // 315° — upper-left
const SWEEP = Math.PI * 1.5;        // 270° clockwise

interface LaunchpadProps {
  onLaunch: (id: AppId) => void;
  visible: boolean;
}

export function Launchpad({ onLaunch, visible }: LaunchpadProps) {
  const [hovered, setHovered] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const numApps = APPS.length;
  const angleStep = SWEEP / (numApps - 1);

  const handleLaunch = (id: AppId) => {
    setExpanded(false);
    onLaunch(id);
  };

  return (
    <div className={`launchpad ${visible ? "visible" : "hidden"}`}>
      {/* Decorative rings — only show when expanded */}
      <div className={`lp-ring lp-ring-outer ${expanded ? "expanded" : ""}`} />
      <div className={`lp-ring lp-ring-inner ${expanded ? "expanded" : ""}`} />

      {/* Center Q logo — always visible, click to toggle */}
      <button
        className={`lp-center ${expanded ? "expanded" : ""}`}
        onClick={() => setExpanded((v) => !v)}
        title="QuantoniumOS"
      >
        <img src="/que-icon.png" alt="Q" className="lp-q-logo" />
      </button>

      {/* App icons in 270° arch — only show when expanded */}
      {APPS.map((app, i) => {
        const angle = START_ANGLE - i * angleStep;
        const x = Math.cos(angle) * ARCH_RADIUS;
        const y = Math.sin(angle) * ARCH_RADIUS;
        const isHovered = hovered === app.id;

        return (
          <button
            key={app.id}
            className={`lp-app ${isHovered ? "hovered" : ""} ${expanded ? "expanded" : ""}`}
            style={{
              transform: expanded
                ? `translate(${x}px, ${y}px)`
                : `translate(0px, 0px) scale(0)`,
              transitionDelay: expanded ? `${i * 40}ms` : `${(numApps - 1 - i) * 25}ms`,
            }}
            onClick={() => handleLaunch(app.id)}
            onMouseEnter={() => setHovered(app.id)}
            onMouseLeave={() => setHovered(null)}
            title={app.label}
          >
            <div className="lp-app-icon">
              <app.icon size={24} />
            </div>
            <span className="lp-app-label">{app.label}</span>
            <span className="lp-app-cat">{app.category}</span>
          </button>
        );
      })}
    </div>
  );
}
