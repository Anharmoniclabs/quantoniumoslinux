/**
 * GuildGame — Phaser-style isometric game view with chatbox
 * A proper game canvas showing agents moving and executing tasks,
 * with a sidebar chatbox using the 3-tier engine chain.
 * 
 * Engine priority: RFTMW 3B (1.2s) → RFTMW 7B (~2s) → Ollama 3B (10s)
 * Supports both Python (Ollama) and Rust inference backends.
 */
import { useEffect, useRef, useState } from "react";
import Phaser from "phaser";
import {
  useGuildAgents, subscribeGuildEvents, sendGuildCommand,
  type GuildEvent, type GuildAgentStats,
} from "../hooks/useGuild";
import { useRustEngine } from "../hooks/useRustEngine";
import {
  Play, Pause, RefreshCw, Send, MessageSquare, Activity,
  Zap, ChevronUp, ChevronDown, Maximize2, Minimize2, Cpu,
} from "lucide-react";
import "./GuildGame.css";

// ── Map & Room Config ──────────────────────────────────────────────────
const MAP_W = 31;
const MAP_H = 17;
const TILE_SIZE = 24; // Base tile size

const ROOMS: Record<string, { x: number; y: number; w: number; h: number; label: string }> = {
  forge_office:  { x: 1,  y: 1,  w: 6, h: 3, label: "FORGE" },
  vial_office:   { x: 1,  y: 5,  w: 6, h: 3, label: "VIAL" },
  ward_office:   { x: 1,  y: 9,  w: 6, h: 3, label: "WARD" },
  lore_office:   { x: 1,  y: 13, w: 6, h: 3, label: "LORE" },
  spark_office:  { x: 24, y: 1,  w: 6, h: 3, label: "SPARK" },
  tide_office:   { x: 24, y: 5,  w: 6, h: 3, label: "TIDE" },
  oracle_office: { x: 24, y: 9,  w: 6, h: 3, label: "ORACLE" },
  sage_office:   { x: 24, y: 13, w: 6, h: 3, label: "SAGE" },
  conference:    { x: 12, y: 6,  w: 7, h: 4, label: "COUNCIL" },
};

const AGENT_COLORS: Record<string, string> = {
  FORGE:  "#e67e22", VIAL:   "#27ae60", WARD:   "#c0392b", LORE:   "#8e44ad",
  SPARK:  "#f1c40f", TIDE:   "#3498db", ORACLE: "#2980b9", SAGE:   "#9b59b6",
};

const ENGINE_INFO: Record<string, { tier: string; color: string; model: string }> = {
  FORGE:  { tier: "3B",  color: "#4a9", model: "qwen2.5-coder" },
  VIAL:   { tier: "3B",  color: "#4a9", model: "qwen2.5-coder" },
  WARD:   { tier: "3B",  color: "#4a9", model: "qwen2.5-coder" },
  LORE:   { tier: "3B",  color: "#4a9", model: "qwen2.5-coder" },
  SPARK:  { tier: "3B",  color: "#4a9", model: "qwen2.5-coder" },
  TIDE:   { tier: "3B",  color: "#4a9", model: "qwen2.5-coder" },
  ORACLE: { tier: "3B",  color: "#4a9", model: "qwen2.5-coder" },
  SAGE:   { tier: "3B",  color: "#4a9", model: "qwen2.5-coder" },
};

const AGENT_HOMES: Record<string, string> = {
  FORGE: "forge_office", VIAL: "vial_office", WARD: "ward_office", LORE: "lore_office",
  SPARK: "spark_office", TIDE: "tide_office", ORACLE: "oracle_office", SAGE: "sage_office",
};

// ── Types ──────────────────────────────────────────────────────────────
interface AgentPos {
  x: number; y: number;
  targetX: number; targetY: number;
  status: "idle" | "working" | "walking";
  task?: string;
}

interface ChatMsg {
  id: string;
  from: "user" | "system" | string;
  text: string;
  time: number;
}

// ── Helper: Get home position ──────────────────────────────────────────
function getHomePos(agent: string): [number, number] {
  const room = AGENT_HOMES[agent];
  const r = room ? ROOMS[room] : null;
  if (r) return [r.x + Math.floor(r.w / 2), r.y + Math.floor(r.h / 2)];
  return [15, 8];
}

// ── Main Component ─────────────────────────────────────────────────────
export function GuildGame() {
  const phaserHostRef = useRef<HTMLDivElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const phaserGameRef = useRef<Phaser.Game | null>(null);
  const phaserSceneRef = useRef<Phaser.Scene | null>(null);
  const spriteRefs = useRef<Record<string, Phaser.GameObjects.Sprite>>({});
  const labelRefs = useRef<Record<string, Phaser.GameObjects.Text>>({});
  const statusDotRefs = useRef<Record<string, Phaser.GameObjects.Arc>>({});
  const animRefs = useRef<Record<string, Phaser.Tweens.Tween>>({});
  const agents = useGuildAgents();
  const rustEngine = useRustEngine();
  
  // Game state
  const [paused, setPaused] = useState(false);
  const [expanded, setExpanded] = useState(true);
  const [useRust, setUseRust] = useState(false); // Toggle Rust vs Python inference
  
  // Agent positions
  const [positions, setPositions] = useState<Record<string, AgentPos>>(() => {
    const init: Record<string, AgentPos> = {};
    for (const name of Object.keys(AGENT_COLORS)) {
      const [x, y] = getHomePos(name);
      init[name] = { x, y, targetX: x, targetY: y, status: "idle" };
    }
    return init;
  });

  // Chat state
  const [chatMsgs, setChatMsgs] = useState<ChatMsg[]>([
    { id: "init", from: "system", text: "Guild comms online. Use @SAGE or @ORACLE for deep analysis.", time: Date.now() }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  // Auto-init Rust engine on mount
  useEffect(() => {
    if (!rustEngine.status.available_7b && !rustEngine.status.available_35b) {
      rustEngine.initialize().catch(() => {
        // Silent fail - will fall back to Ollama
      });
    }
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMsgs]);

  // Movement interpolation - smooth agent walking
  useEffect(() => {
    const moveInterval = setInterval(() => {
      setPositions(prev => {
        let changed = false;
        const next = { ...prev };
        for (const [name, pos] of Object.entries(prev)) {
          const dx = pos.targetX - pos.x;
          const dy = pos.targetY - pos.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist > 0.1) {
            const speed = 0.15; // tiles per frame
            const step = Math.min(speed, dist);
            next[name] = {
              ...pos,
              x: pos.x + (dx / dist) * step,
              y: pos.y + (dy / dist) * step,
              status: "walking"
            };
            changed = true;
          } else if (pos.status === "walking") {
            // Arrived at destination
            next[name] = {
              ...pos,
              x: pos.targetX,
              y: pos.targetY,
              status: "idle"
            };
            changed = true;
          }
        }
        return changed ? next : prev;
      });
    }, 50); // 20 FPS movement
    return () => clearInterval(moveInterval);
  }, []);

  // Refresh Phaser viewport when panel expands/collapses
  useEffect(() => {
    phaserGameRef.current?.scale.refresh();
  }, [expanded]);

  // Subscribe to guild events
  useEffect(() => {
    return subscribeGuildEvents((ev: GuildEvent) => {
      // Movement events
      if (ev.type === "walk_to") {
        const { name, x, y } = ev as { name: string; x: number; y: number };
        setPositions(prev => {
          const existing = prev[name] ?? { x: 15, y: 8, targetX: 15, targetY: 8, status: "idle" as const };
          return {
            ...prev,
            [name]: { ...existing, targetX: x, targetY: y, status: "walking" }
          };
        });
      } else if (ev.type === "quest_start") {
        const { name, quest } = ev as { name: string; quest: string };
        setPositions(prev => {
          const existing = prev[name] ?? { x: 15, y: 8, targetX: 15, targetY: 8, status: "idle" as const };
          return {
            ...prev,
            [name]: { ...existing, status: "working", task: quest }
          };
        });
        setChatMsgs(prev => [...prev, {
          id: crypto.randomUUID(),
          from: name,
          text: `🎯 Starting: ${quest.slice(0, 80)}...`,
          time: Date.now()
        }]);
      } else if (ev.type === "quest_done") {
        const { name, quest } = ev as { name: string; quest: string };
        setPositions(prev => {
          const existing = prev[name] ?? { x: 15, y: 8, targetX: 15, targetY: 8, status: "idle" as const };
          return {
            ...prev,
            [name]: { ...existing, status: "idle", task: undefined }
          };
        });
        setChatMsgs(prev => [...prev, {
          id: crypto.randomUUID(),
          from: name,
          text: `✅ Completed: ${quest.slice(0, 60)}`,
          time: Date.now()
        }]);
      } else if (ev.type === "quest_fail") {
        const { name, error } = ev as { name: string; error: string };
        setPositions(prev => {
          const existing = prev[name] ?? { x: 15, y: 8, targetX: 15, targetY: 8, status: "idle" as const };
          return {
            ...prev,
            [name]: { ...existing, status: "idle", task: undefined }
          };
        });
        setChatMsgs(prev => [...prev, {
          id: crypto.randomUUID(),
          from: name,
          text: `❌ Failed: ${error?.slice(0, 60) || "unknown error"}`,
          time: Date.now()
        }]);
      }
      
      // Chat responses from SAGE/ORACLE
      if (ev.type === "chat_response") {
        const { from, msg } = ev as { from: string; msg: string };
        setIsTyping(false);
        setChatMsgs(prev => [...prev, {
          id: crypto.randomUUID(),
          from: from || "SYSTEM",
          text: msg || "(empty response)",
          time: Date.now()
        }]);
      } else if (ev.type === "chat_stream" && (ev as { event?: string }).event === "start") {
        setIsTyping(true);
      } else if (ev.type === "chat_stream" && (ev as { event?: string }).event === "end") {
        setIsTyping(false);
      }
      
      // System messages (including reports)
      if (ev.type === "system") {
        const { msg } = ev as { msg: string };
        if (msg && msg.length > 0) {
          setChatMsgs(prev => [...prev, {
            id: crypto.randomUUID(),
            from: "system",
            text: msg,
            time: Date.now()
          }]);
        }
      }
      
      // Stats updates (show in chat for visibility)
      if (ev.type === "stats") {
        const { name, level, xp } = ev as { name: string; level: number; xp: number };
        setChatMsgs(prev => [...prev, {
          id: crypto.randomUUID(),
          from: name,
          text: `📊 Level ${level} | ${xp} XP`,
          time: Date.now()
        }]);
      }
      
      // Log messages
      if (ev.type === "log") {
        const { name, msg } = ev as { name: string; msg: string };
        if (msg && !msg.includes("tick") && !msg.includes("walk")) {
          setChatMsgs(prev => {
            if (prev.length > 100) prev = prev.slice(-50);
            return [...prev, {
              id: crypto.randomUUID(),
              from: name || "SYSTEM",
              text: msg.slice(0, 200),
              time: Date.now()
            }];
          });
        }
      }
    });
  }, []);

  // Initialize Phaser scene once
  useEffect(() => {
    if (!phaserHostRef.current || phaserGameRef.current) return;

    const worldW = MAP_W * TILE_SIZE;
    const worldH = MAP_H * TILE_SIZE;

    const scene = new Phaser.Scene("guild-hq");

    (scene as Phaser.Scene & { create: () => void }).create = () => {
      const g = scene.add.graphics();

      // Agent sprite texture (re-used for all agents)
      if (!scene.textures.exists("guild-agent-dot")) {
        g.clear();
        g.fillStyle(0xffffff, 1);
        g.fillCircle(8, 8, 8);
        g.lineStyle(2, 0x000000, 1);
        g.strokeCircle(8, 8, 8);
        g.generateTexture("guild-agent-dot", 16, 16);
      }

      // Background
      g.clear();
      g.fillStyle(0x080810, 1);
      g.fillRect(0, 0, worldW, worldH);

      // Grid
      g.lineStyle(1, 0x12121a, 0.5);
      for (let x = 0; x <= MAP_W; x++) {
        g.lineBetween(x * TILE_SIZE, 0, x * TILE_SIZE, worldH);
      }
      for (let y = 0; y <= MAP_H; y++) {
        g.lineBetween(0, y * TILE_SIZE, worldW, y * TILE_SIZE);
      }

      // Rooms
      for (const [key, room] of Object.entries(ROOMS)) {
        const px = room.x * TILE_SIZE;
        const py = room.y * TILE_SIZE;
        const pw = room.w * TILE_SIZE;
        const ph = room.h * TILE_SIZE;

        const agentName = Object.entries(AGENT_HOMES).find(([_, v]) => v === key)?.[0];
        const color = agentName ? AGENT_COLORS[agentName] : "#666666";
        const isConference = key === "conference";
        const fill = Phaser.Display.Color.HexStringToColor(isConference ? "#10101a" : `${color}20`).color;
        const border = Phaser.Display.Color.HexStringToColor(isConference ? "#303050" : `${color}70`).color;

        g.fillStyle(fill, 1);
        g.fillRect(px, py, pw, ph);
        g.lineStyle(2, border, 1);
        g.strokeRect(px + 1, py + 1, pw - 2, ph - 2);

        scene.add.text(px + pw / 2, py + 4, room.label, {
          fontFamily: "monospace",
          fontSize: "11px",
          color: isConference ? "#505070" : color,
          fontStyle: "bold",
        }).setOrigin(0.5, 0);
      }

      // Hallways
      g.lineStyle(Math.max(4, TILE_SIZE * 0.4), 0x1a1a25, 0.8);
      g.lineBetween(8 * TILE_SIZE, 2 * TILE_SIZE, 8 * TILE_SIZE, 15 * TILE_SIZE);
      g.lineBetween(23 * TILE_SIZE, 2 * TILE_SIZE, 23 * TILE_SIZE, 15 * TILE_SIZE);
      g.lineBetween(8 * TILE_SIZE, 8 * TILE_SIZE, 12 * TILE_SIZE, 8 * TILE_SIZE);
      g.lineBetween(19 * TILE_SIZE, 8 * TILE_SIZE, 23 * TILE_SIZE, 8 * TILE_SIZE);

      // Agent sprites
      for (const name of Object.keys(AGENT_COLORS)) {
        const [hx, hy] = getHomePos(name);
        const px = hx * TILE_SIZE;
        const py = hy * TILE_SIZE;
        const tint = Phaser.Display.Color.HexStringToColor(AGENT_COLORS[name]).color;

        const sprite = scene.add.sprite(px, py, "guild-agent-dot").setScale(1.5).setTint(tint);
        const label = scene.add.text(px, py - 16, name[0], {
          fontFamily: "-apple-system, sans-serif",
          fontSize: "11px",
          color: "#ffffff",
          fontStyle: "bold",
        }).setOrigin(0.5, 0.5);
        const statusDot = scene.add.circle(px + 9, py - 9, 3, 0x3b82f6, 1);
        const walkAnim = scene.tweens.add({
          targets: sprite,
          scaleX: 1.65,
          scaleY: 1.25,
          yoyo: true,
          repeat: -1,
          duration: 140,
          paused: true,
        });

        spriteRefs.current[name] = sprite;
        labelRefs.current[name] = label;
        statusDotRefs.current[name] = statusDot;
        animRefs.current[name] = walkAnim;
      }
    };

    const game = new Phaser.Game({
      type: Phaser.AUTO,
      parent: phaserHostRef.current,
      backgroundColor: "#080810",
      render: { pixelArt: true, antialias: false },
      scale: {
        mode: Phaser.Scale.FIT,
        autoCenter: Phaser.Scale.CENTER_BOTH,
        width: worldW,
        height: worldH,
      },
      scene: [scene],
    });

    phaserSceneRef.current = scene;
    phaserGameRef.current = game;

    return () => {
      game.destroy(true);
      phaserSceneRef.current = null;
      phaserGameRef.current = null;
      spriteRefs.current = {};
      labelRefs.current = {};
      statusDotRefs.current = {};
      animRefs.current = {};
    };
  }, []);

  // Keep Phaser sprites synced with agent movement/state
  useEffect(() => {
    for (const [name, pos] of Object.entries(positions)) {
      const sprite = spriteRefs.current[name];
      const label = labelRefs.current[name];
      const statusDot = statusDotRefs.current[name];
      const anim = animRefs.current[name];
      if (!sprite || !label || !statusDot || !anim) continue;

      const px = pos.x * TILE_SIZE;
      const py = pos.y * TILE_SIZE;
      sprite.setPosition(px, py);
      label.setPosition(px, py - 16);
      statusDot.setPosition(px + 9, py - 9);

      const agentData = agents.find((a) => a.name === name);
      const isWorking = pos.status === "working" || agentData?.status === "working";
      const isWalking = pos.status === "walking";
      const isOffline = !agentData || (agentData.lastUpdate === 0 && agentData.xp === 0 && agentData.level <= 1);

      if (isOffline) {
        sprite.setTint(0x333333);
      } else {
        sprite.setTint(Phaser.Display.Color.HexStringToColor(AGENT_COLORS[name] || "#ffffff").color);
      }

      const statusColor = isWorking ? 0x22c55e : isWalking ? 0xeab308 : 0x3b82f6;
      statusDot.setFillStyle(statusColor, 1);

      if (isWorking || isWalking) {
        if (anim.paused) anim.resume();
      } else {
        if (!anim.paused) anim.pause();
        sprite.setScale(1.5);
      }
    }
  }, [positions, agents]);

  // ── Chat Handlers ────────────────────────────────────────────────────
  const handleChatSend = async () => {
    if (!chatInput.trim()) return;
    const text = chatInput.trim();
    setChatInput("");
    
    // Add user message
    setChatMsgs(prev => [...prev, {
      id: crypto.randomUUID(),
      from: "user",
      text,
      time: Date.now()
    }]);
    
    // Check for @SAGE or @ORACLE mentions for 35B routing
    const sageMatch = text.match(/^@SAGE\s+(.+)/i);
    const oracleMatch = text.match(/^@ORACLE\s+(.+)/i);
    const use35b = !!(sageMatch || oracleMatch);
    const prompt = sageMatch?.[1] || oracleMatch?.[1] || text;
    
    setIsTyping(true);
    
    // Use Rust engine if enabled and available
    if (useRust && (rustEngine.status.available_7b || (use35b && rustEngine.status.available_35b))) {
      try {
        const response = await rustEngine.inference(prompt, {
          use35b,
          maxTokens: 256,
          temperature: 0.7,
        });
        
        setIsTyping(false);
        setChatMsgs(prev => [...prev, {
          id: crypto.randomUUID(),
          from: use35b ? (sageMatch ? "SAGE" : "ORACLE") : "FORGE",
          text: `[Rust] ${response || "(empty response)"}`,
          time: Date.now()
        }]);
        return;
      } catch (err) {
        // Fall back to Python/Ollama
        console.warn("[GuildGame] Rust inference failed, falling back:", err);
      }
    }
    
    // Fall back to Python/Ollama via guild
    const hasAtMention = text.match(/^@\w+/);
    const routed = hasAtMention ? text : `@FORGE ${text}`;
    
    try {
      await sendGuildCommand({ cmd: "chat", text: routed });
    } catch (err) {
      setIsTyping(false);
      setChatMsgs(prev => [...prev, {
        id: crypto.randomUUID(),
        from: "system",
        text: `Error: ${err instanceof Error ? err.message : "Unknown error"}`,
        time: Date.now()
      }]);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleChatSend();
    }
  };

  // ── Control Actions ──────────────────────────────────────────────────
  const handlePatrol = () => sendGuildCommand({ cmd: "patrol" });
  const handleReport = () => sendGuildCommand({ cmd: "report" });
  const handleReset = () => {
    setPositions(prev => {
      const next: Record<string, AgentPos> = {};
      for (const name of Object.keys(prev)) {
        const [x, y] = getHomePos(name);
        next[name] = { x, y, targetX: x, targetY: y, status: "idle" };
      }
      return next;
    });
  };

  // ── Render ───────────────────────────────────────────────────────────
  const rustAvailable = rustEngine.status.available_7b || rustEngine.status.available_35b;
  
  return (
    <div className={`guild-game ${expanded ? "expanded" : "compact"}`}>
      {/* Game Canvas Panel */}
      <div className="game-panel">
        <div className="game-header">
          <h3><Activity size={14} /> QUE Guild HQ</h3>
          <div className="game-controls">
            <button onClick={() => setPaused(!paused)} title={paused ? "Resume" : "Pause"}>
              {paused ? <Play size={14} /> : <Pause size={14} />}
            </button>
            <button onClick={handlePatrol} title="Send agents on patrol">
              <Zap size={14} />
            </button>
            <button onClick={handleReport} title="Request status report">
              <MessageSquare size={14} />
            </button>
            <button onClick={handleReset} title="Reset positions">
              <RefreshCw size={14} />
            </button>
            <button 
              onClick={() => setUseRust(!useRust)} 
              title={useRust ? "Using Rust engine" : "Using Ollama"}
              className={useRust ? "active" : ""}
              disabled={!rustAvailable}
            >
              <Cpu size={14} />
            </button>
            <button onClick={() => setExpanded(!expanded)} title={expanded ? "Compact" : "Expand"}>
              {expanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
            </button>
          </div>
        </div>
        <div className="game-canvas-wrap">
          <div ref={phaserHostRef} className="phaser-host" />
        </div>
        <div className="game-legend">
          <span className="legend-item"><span className="dot" style={{ background: "#4a9" }} /> 3B QW</span>
          <span className="status-dot working" /> Working
          <span className="status-dot walking" /> Moving
          <span className="status-dot idle" /> Idle
          {rustAvailable && (
            <span className="legend-item rust-status">
              <Cpu size={10} /> {useRust ? "Rust" : "Ollama"}
            </span>
          )}
        </div>
      </div>

      {/* Chatbox Panel */}
      <div className="chat-panel">
        <div className="chat-header">
          <h3><MessageSquare size={14} /> Agent Comms</h3>
          <span className="chat-hint">@SAGE @ORACLE for routing</span>
        </div>
        
        <div className="chat-messages">
          {chatMsgs.slice(-50).map(msg => (
            <div key={msg.id} className={`chat-msg ${msg.from === "user" ? "user" : msg.from === "system" ? "system" : "agent"}`}>
              {msg.from !== "user" && msg.from !== "system" && (
                <span className="msg-from" style={{ color: AGENT_COLORS[msg.from] || "#888" }}>
                  {msg.from}
                </span>
              )}
              <span className="msg-text">{msg.text}</span>
            </div>
          ))}
          {isTyping && (
            <div className="chat-msg typing">
              <span className="typing-indicator">
                <span /><span /><span />
              </span>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="chat-input-area">
          <input
            type="text"
            placeholder="Talk to agents... (@SAGE for analysis)"
            value={chatInput}
            onChange={e => setChatInput(e.target.value)}
            onKeyDown={handleKeyDown}
            className="chat-input"
          />
          <button onClick={handleChatSend} className="chat-send" disabled={!chatInput.trim()}>
            <Send size={16} />
          </button>
        </div>

        <div className="chat-quick">
          <button onClick={() => setChatInput("@SAGE analyze current tasks")}>@SAGE analyze</button>
          <button onClick={() => setChatInput("@ORACLE route priority")}>@ORACLE route</button>
          <button onClick={() => {
            // Generate local report with all agent stats
            const working = agents.filter(a => a.status === "working");
            const idle = agents.filter(a => a.status !== "working");
            const totalXp = agents.reduce((sum, a) => sum + (a.xp || 0), 0);
            const report = `📋 GUILD REPORT\n` +
              `━━━━━━━━━━━━━━━━━━━━\n` +
              `🟢 Working: ${working.map(a => a.name).join(", ") || "none"}\n` +
              `⚪ Idle: ${idle.map(a => a.name).join(", ")}\n` +
              `━━━━━━━━━━━━━━━━━━━━\n` +
              agents.map(a => `${a.name}: Lv${a.level} | ${a.xp}xp${a.current_quest ? ` | 🎯 ${a.current_quest.slice(0, 30)}...` : ""}`).join("\n") +
              `\n━━━━━━━━━━━━━━━━━━━━\n` +
              `Total XP: ${totalXp}`;
            setChatMsgs(prev => [...prev, {
              id: crypto.randomUUID(),
              from: "system",
              text: report,
              time: Date.now()
            }]);
            // Also request server report
            sendGuildCommand({ cmd: "report" });
          }}>📋 Full Report</button>
        </div>
      </div>
    </div>
  );
}
