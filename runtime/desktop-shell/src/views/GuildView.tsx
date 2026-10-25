import { useMemo, useState, useEffect, useRef } from "react";
import {
  Hammer, FlaskConical, ShieldAlert, BookOpen, Sparkles, Waves,
  Eye, Brain, Play, FileText, Circle, AlertTriangle, Map, Gamepad2,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import {
  useGuildAgents, useGuildEventLog, useGuildStatus,
  sendGuildCommand, type GuildAgentStats,
} from "../hooks/useGuild";
import { GuildOffice } from "./GuildOffice";
import { GuildGame } from "./GuildGame";
import "./GuildView.css";

// Engine assignments (3-tier chain):
// Priority: RFTMW 3B (1.2s) → RFTMW 7B (~2s) → Ollama 3B (10s)
// All agents use 3B as primary for fastest response
const AGENT_META: Record<string, { icon: LucideIcon; role: string; color: string; engine: string }> = {
  FORGE:  { icon: Hammer,       role: "Native / ASM builds",    color: "#e67e22", engine: "3B QW" },
  VIAL:   { icon: FlaskConical, role: "Tests & benchmarks",     color: "#27ae60", engine: "3B QW" },
  WARD:   { icon: ShieldAlert,  role: "Security & audits",      color: "#c0392b", engine: "3B QW" },
  LORE:   { icon: BookOpen,     role: "Docs & provenance",      color: "#8e44ad", engine: "3B QW" },
  SPARK:  { icon: Sparkles,     role: "Apps & UI",              color: "#f1c40f", engine: "3B QW" },
  TIDE:   { icon: Waves,        role: "Mobile / frontend",      color: "#3498db", engine: "3B QW" },
  ORACLE: { icon: Eye,          role: "Coordinator & router",   color: "#2980b9", engine: "3B QW" },
  SAGE:   { icon: Brain,        role: "Deep analysis",          color: "#9b59b6", engine: "3B QW" },
};

function AgentCard({ agent }: { agent: GuildAgentStats }) {
  const meta = AGENT_META[agent.name] ?? { icon: Circle, role: "", color: "#888", engine: "?" };
  const Icon = meta.icon;
  const hasData = agent.xp > 0 || agent.level > 1;
  const stale = !hasData && (agent.lastUpdate === 0 || Date.now() - agent.lastUpdate > 60_000);
  const statusClass =
    agent.status === "working" ? "working" :
    agent.status === "cooldown" ? "cooldown" :
    agent.status === "failed" ? "failed" :
    stale ? "offline" : "idle";

  // Engine color based on tier (all 3B now for speed)
  const engineColor = meta.engine?.startsWith("3B") ? "#4a9" :
                      meta.engine?.startsWith("7B") ? "#49a" : "#94a";

  return (
    <div className={`guild-card ${statusClass}`} title={agent.current_quest || undefined}>
      <div className="guild-card-head">
        <div className="guild-card-icon" style={{ color: meta.color }}>
          <Icon size={14} />
        </div>
        <div className="guild-card-id">
          <span className="guild-card-name">{agent.name}</span>
        </div>
        <span className="guild-card-engine" style={{ color: engineColor }}>{meta.engine}</span>
      </div>
      <div className="guild-card-stats">
        <div className="guild-stat">
          <span className="guild-stat-label">LV</span>
          <span className="guild-stat-value">{agent.level}</span>
        </div>
        <div className="guild-stat">
          <span className="guild-stat-label">XP</span>
          <span className="guild-stat-value">{agent.xp}</span>
        </div>
        <div className="guild-stat">
          <span className="guild-stat-label">ST</span>
          <span className="guild-stat-value small">{stale ? "off" : (agent.status ?? "idle")}</span>
        </div>
      </div>
    </div>
  );
}

function formatEvent(ev: Record<string, unknown>): { icon: "i" | "ok" | "err" | "q"; text: string } {
  const t = ev.type as string;
  const name = (ev.name as string) ?? "";
  switch (t) {
    case "ready":
      return { icon: "i", text: `guild online — ${((ev.agents as string[]) ?? []).length} agents ready` };
    case "system":
      return { icon: "i", text: `system: ${ev.msg}` };
    case "stats":
      return { icon: "i", text: `${name} stats → lv${ev.level} xp${ev.xp}` };
    case "quest_start":
      return { icon: "q", text: `${name} ▶ ${ev.quest}` };
    case "quest_done":
      return { icon: "ok", text: `${name} ✓ ${ev.quest} (+${ev.xp} xp)` };
    case "quest_fail":
      return { icon: "err", text: `${name} ✗ ${ev.quest}: ${ev.error}` };
    case "cooldown":
      return { icon: "err", text: `${name} cooldown: ${ev.reason}` };
    case "idle_action":
      return { icon: "i", text: `${name} idle: ${ev.result}` };
    case "log":
      return { icon: "i", text: `${name}: ${ev.msg}` };
    case "chat_stream":
      return { icon: "q", text: `${ev.from} chat ${ev.event}` };
    case "chat_chunk":
      return { icon: "q", text: `${ev.from} ▸ ${(ev.delta as string).trim()}` };
    case "chat_response":
      return { icon: "ok", text: `${ev.from}: ${ev.msg}` };
    case "walk_to":
      return { icon: "i", text: `${name} → (${ev.x},${ev.y})` };
    default:
      return { icon: "i", text: JSON.stringify(ev) };
  }
}

export function GuildView() {
  const agents = useGuildAgents();
  const log = useGuildEventLog();
  const { available } = useGuildStatus();
  const [autoTick, setAutoTick] = useState(true);
  const [viewMode, setViewMode] = useState<"game" | "cards" | "mini">("game");
  const logRef = useRef<HTMLDivElement>(null);
  const [pinned, setPinned] = useState(true);

  // Auto-scroll unless user scrolled up
  useEffect(() => {
    if (!pinned) return;
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [log, pinned]);

  const summary = useMemo(() => {
    const working = agents.filter((a) => a.status === "working").length;
    const totalXp = agents.reduce((n, a) => n + (a.xp || 0), 0);
    return { working, totalXp, count: agents.length };
  }, [agents]);

  const patrol = () => sendGuildCommand({ cmd: "patrol" });
  const report = () => sendGuildCommand({ cmd: "report" });
  const toggleAuto = () => {
    const next = !autoTick;
    setAutoTick(next);
    sendGuildCommand({ cmd: "auto", on: next });
  };

  if (!available) {
    return (
      <div className="guild-view">
        <div className="guild-offline">
          <AlertTriangle size={28} />
          <h3>Guild unavailable</h3>
          <p>
            <code>src/rpg_headless.py</code> isn't running.
            Check the console — Python may be missing or the spawn failed at boot.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="guild-view">
      <header className="guild-header">
        <div className="guild-header-title">
          <h1>QUE Guild</h1>
          <span className="guild-sub">
            {summary.count} agents · {summary.working} working · {summary.totalXp} xp total
          </span>
        </div>
        <div className="guild-header-actions">
          <button
            className={`guild-btn ${viewMode === "game" ? "active" : ""}`}
            onClick={() => setViewMode("game")}
            title="Game view with chatbox"
          >
            <Gamepad2 size={13} /> Game
          </button>
          <button
            className={`guild-btn ${viewMode === "cards" ? "active" : ""}`}
            onClick={() => setViewMode("cards")}
            title="Card grid view"
          >
            <Map size={13} /> Cards
          </button>
          <button
            className={`guild-btn ${viewMode === "mini" ? "active" : ""}`}
            onClick={() => setViewMode("mini")}
            title="Mini office view"
          >
            <Eye size={13} /> Mini
          </button>
          <button className="guild-btn" onClick={patrol} title="Dispatch patrol quests">
            <Play size={13} /> Patrol
          </button>
          <button className="guild-btn" onClick={report} title="Request full status">
            <FileText size={13} /> Report
          </button>
          <button
            className={`guild-btn ${autoTick ? "active" : ""}`}
            onClick={toggleAuto}
            title="Auto-tick loop"
          >
            <Circle size={8} fill={autoTick ? "#22c55e" : "transparent"} /> Auto
          </button>
        </div>
      </header>

      {/* Game View - Full Phaser-style with chatbox */}
      {viewMode === "game" && <GuildGame />}

      {/* Cards View - Agent cards with mini office */}
      {viewMode === "cards" && (
        <>
          <GuildOffice />
          <section className="guild-grid">
            {agents.map((a) => <AgentCard key={a.name} agent={a} />)}
          </section>
        </>
      )}

      {/* Mini View - Just the office */}
      {viewMode === "mini" && <GuildOffice />}

      {/* Event Log - only show in cards/mini mode */}
      {viewMode !== "game" && (
        <section className="guild-log-wrap">
          <div className="guild-log-head">
            <h3>Live feed</h3>
            <label className="guild-pin">
              <input
                type="checkbox"
                checked={pinned}
                onChange={(e) => setPinned(e.target.checked)}
              />
              <span>pin to bottom</span>
            </label>
          </div>
          <div className="guild-log" ref={logRef}>
            {log.length === 0 && (
              <div className="guild-log-empty">waiting for events…</div>
            )}
            {log.slice(-150).map((ev, i) => {
              const f = formatEvent(ev as Record<string, unknown>);
              return (
                <div key={i} className={`guild-log-row ${f.icon}`}>
                  <span className="guild-log-bullet" />
                  <span className="guild-log-text">{f.text}</span>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
