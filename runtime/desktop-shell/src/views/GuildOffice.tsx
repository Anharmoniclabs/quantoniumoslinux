/**
 * GuildOffice — Visual canvas view of the QUE Guild penthouse office.
 * Shows animated mini agents moving around the office space.
 * Engine routing:
 *   - ORACLE + SAGE → 35b qwen35-opus-distil (heavy reasoning)
 *   - Others -> 3b qwen2.5-coder workers
 */
import { useEffect, useRef, useState, useCallback } from "react";
import { useGuildAgents, subscribeGuildEvents, type GuildEvent } from "../hooks/useGuild";
import "./GuildOffice.css";

// Map dimensions from world.py
const MAP_W = 31;
const MAP_H = 17;

// Room definitions from world.py
const ROOMS: Record<string, [number, number, number, number]> = {
  forge_office:   [1,  1,  6, 3],
  vial_office:    [1,  5,  6, 3],
  ward_office:    [1,  9,  6, 3],
  lore_office:    [1,  13, 6, 3],
  spark_office:   [24, 1,  6, 3],
  tide_office:    [24, 5,  6, 3],
  oracle_office:  [24, 9,  6, 3],
  sage_office:    [24, 13, 6, 3],
  conference:     [12, 6,  7, 4],
};

// Agent colors - muted professional palette
const AGENT_COLORS: Record<string, string> = {
  FORGE:  "#e67e22",
  VIAL:   "#27ae60",
  WARD:   "#c0392b",
  LORE:   "#8e44ad",
  SPARK:  "#f1c40f",
  TIDE:   "#3498db",
  ORACLE: "#2980b9",
  SAGE:   "#9b59b6",
};

// Engine assignments
const ENGINE_LABELS: Record<string, { engine: string; color: string }> = {
  FORGE:  { engine: "R1", color: "#4a9" },
  VIAL:   { engine: "R1", color: "#4a9" },
  WARD:   { engine: "R1", color: "#4a9" },
  LORE:   { engine: "R1", color: "#4a9" },
  SPARK:  { engine: "QW", color: "#a94" },
  TIDE:   { engine: "QW", color: "#a94" },
  ORACLE: { engine: "OP", color: "#94a" },
  SAGE:   { engine: "OP", color: "#94a" },
};

const AGENT_HOMES: Record<string, string> = {
  FORGE:  "forge_office",
  VIAL:   "vial_office",
  WARD:   "ward_office",
  LORE:   "lore_office",
  SPARK:  "spark_office",
  TIDE:   "tide_office",
  ORACLE: "oracle_office",
  SAGE:   "sage_office",
};

interface AgentPosition {
  x: number;
  y: number;
  targetX: number;
  targetY: number;
  status: "idle" | "working" | "walking";
}

function getHomePos(agent: string): [number, number] {
  const room = AGENT_HOMES[agent];
  if (room && ROOMS[room]) {
    const [rx, ry, rw, rh] = ROOMS[room];
    return [rx + Math.floor(rw / 2), ry + Math.floor(rh / 2)];
  }
  return [15, 8];
}

export function GuildOffice() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const agents = useGuildAgents();
  const [scale, setScale] = useState(1);
  const [positions, setPositions] = useState<Record<string, AgentPosition>>(() => {
    const init: Record<string, AgentPosition> = {};
    for (const name of Object.keys(AGENT_COLORS)) {
      const [x, y] = getHomePos(name);
      init[name] = { x, y, targetX: x, targetY: y, status: "idle" };
    }
    return init;
  });

  // Auto-fit to container
  useEffect(() => {
    const updateScale = () => {
      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const maxW = rect.width - 16;
      const maxH = 200; // Compact height
      const scaleX = maxW / MAP_W;
      const scaleY = maxH / MAP_H;
      setScale(Math.min(scaleX, scaleY, 14)); // Max 14px per tile
    };
    updateScale();
    window.addEventListener("resize", updateScale);
    return () => window.removeEventListener("resize", updateScale);
  }, []);

  // Subscribe to walk_to events
  useEffect(() => {
    return subscribeGuildEvents((ev: GuildEvent) => {
      if (ev.type === "walk_to") {
        const { name, x, y } = ev as { name: string; x: number; y: number };
        setPositions((prev) => ({
          ...prev,
          [name]: { ...prev[name], targetX: x, targetY: y, status: "walking" },
        }));
      } else if (ev.type === "quest_start") {
        const { name } = ev as { name: string };
        setPositions((prev) => ({
          ...prev,
          [name]: { ...prev[name], status: "working" },
        }));
      } else if (ev.type === "quest_done" || ev.type === "quest_fail") {
        const { name } = ev as { name: string };
        setPositions((prev) => ({
          ...prev,
          [name]: { ...prev[name], status: "idle" },
        }));
      }
    });
  }, []);

  // Animate agents toward their targets
  useEffect(() => {
    const interval = setInterval(() => {
      setPositions((prev) => {
        const next = { ...prev };
        let changed = false;
        for (const name of Object.keys(next)) {
          const p = next[name];
          const dx = p.targetX - p.x;
          const dy = p.targetY - p.y;
          if (Math.abs(dx) > 0.1 || Math.abs(dy) > 0.1) {
            next[name] = { ...p, x: p.x + dx * 0.12, y: p.y + dy * 0.12 };
            changed = true;
          } else if (p.status === "walking") {
            next[name] = { ...p, x: p.targetX, y: p.targetY, status: "idle" };
            changed = true;
          }
        }
        return changed ? next : prev;
      });
    }, 50);
    return () => clearInterval(interval);
  }, []);

  // Render the office
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const W = Math.floor(MAP_W * scale);
    const H = Math.floor(MAP_H * scale);
    
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = "#0c0c10";
    ctx.fillRect(0, 0, W, H);

    // Draw rooms
    for (const [key, [rx, ry, rw, rh]] of Object.entries(ROOMS)) {
      const px = rx * scale;
      const py = ry * scale;
      const pw = rw * scale;
      const ph = rh * scale;

      // Room fill
      const agent = Object.entries(AGENT_HOMES).find(([_, v]) => v === key)?.[0];
      if (key === "conference") {
        ctx.fillStyle = "#14141c";
        ctx.strokeStyle = "#252530";
      } else {
        const color = agent ? AGENT_COLORS[agent] : "#1a1a22";
        ctx.fillStyle = color + "12";
        ctx.strokeStyle = color + "40";
      }
      
      ctx.lineWidth = 1;
      ctx.fillRect(px, py, pw, ph);
      ctx.strokeRect(px + 0.5, py + 0.5, pw - 1, ph - 1);
    }

    // Draw agents
    const agentRadius = Math.max(4, scale * 0.35);
    for (const [name, pos] of Object.entries(positions)) {
      const px = pos.x * scale;
      const py = pos.y * scale;
      const color = AGENT_COLORS[name] || "#fff";
      const agentData = agents.find((a) => a.name === name);
      const isWorking = pos.status === "working" || agentData?.status === "working";
      const isOffline = !agentData || (agentData.lastUpdate === 0 && agentData.xp === 0);
      const engineInfo = ENGINE_LABELS[name];

      // Working glow
      if (isWorking) {
        const pulse = Math.sin(Date.now() / 150) * 0.4 + 0.6;
        const gradient = ctx.createRadialGradient(px, py, 1, px, py, agentRadius * 2);
        gradient.addColorStop(0, color + Math.floor(pulse * 80).toString(16).padStart(2, "0"));
        gradient.addColorStop(1, color + "00");
        ctx.beginPath();
        ctx.arc(px, py, agentRadius * 2, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
      }

      // Agent body
      ctx.beginPath();
      ctx.arc(px, py, agentRadius, 0, Math.PI * 2);
      ctx.fillStyle = isOffline ? "#333" : color;
      ctx.fill();

      // Agent initial
      ctx.fillStyle = "#fff";
      ctx.font = `bold ${Math.max(7, agentRadius * 1.2)}px -apple-system, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(name[0], px, py + 0.5);

      // Engine badge
      if (engineInfo && !isOffline && scale > 8) {
        ctx.fillStyle = engineInfo.color;
        ctx.font = `bold ${Math.max(6, scale * 0.4)}px monospace`;
        ctx.textAlign = "left";
        ctx.fillText(engineInfo.engine, px + agentRadius + 2, py - agentRadius + 4);
      }
    }
  }, [positions, agents, scale]);

  // Redraw on animation frame
  useEffect(() => {
    let animId: number;
    const loop = () => {
      draw();
      animId = requestAnimationFrame(loop);
    };
    loop();
    return () => cancelAnimationFrame(animId);
  }, [draw]);

  return (
    <div className="guild-office" ref={containerRef}>
      <canvas ref={canvasRef} />
    </div>
  );
}
