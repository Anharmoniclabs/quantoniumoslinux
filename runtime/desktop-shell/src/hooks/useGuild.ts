// useGuild — singleton subscription to the Rust guild event bus.
//
// Rust spawns src/rpg_headless.py once at launch. This module maintains a
// single broadcast subscription over a Tauri Channel, and fans events out
// to every React consumer. Any view (GuildView, ChatView @-mentions,
// desktop toasts) can subscribe without causing duplicate IPC traffic.

import { useEffect, useState, useSyncExternalStore, useCallback } from "react";
import { invoke, Channel } from "@tauri-apps/api/core";

export interface GuildAgentStats {
  name: string;
  level: number;
  xp: number;
  hp?: number;
  mp?: number;
  current_quest?: string | null;
  status?: string;
  lastUpdate: number;
}

export type GuildEvent =
  | { type: "ready"; agents: string[]; homes?: Record<string, [number, number]> }
  | { type: "stats"; name: string; level: number; xp: number; hp?: number; mp?: number; [k: string]: unknown }
  | { type: "quest_start"; name: string; quest: string }
  | { type: "quest_done"; name: string; quest: string; xp: number; level: number }
  | { type: "quest_fail"; name: string; quest: string; error: string }
  | { type: "cooldown"; name: string; reason: string }
  | { type: "idle_action"; name: string; result: string }
  | { type: "walk_to"; name: string; x: number; y: number }
  | { type: "log"; name: string; msg: string; category?: string }
  | { type: "system"; msg: string }
  | { type: "chat_stream"; from: string; event: "start" | "end" }
  | { type: "chat_chunk"; from: string; delta: string }
  | { type: "chat_response"; from: string; msg: string }
  | { type: string; [k: string]: unknown };

const KNOWN_AGENTS = ["FORGE", "VIAL", "WARD", "LORE", "SPARK", "TIDE", "ORACLE", "SAGE"];

type Listener = (ev: GuildEvent) => void;

// ── Module-singleton state ────────────────────────────────────────────
let subscribed = false;
let started = false;
let available: boolean | null = null;
const listeners = new Set<Listener>();
const agents = new Map<string, GuildAgentStats>();
const eventLog: GuildEvent[] = [];
const MAX_LOG = 300;

// Cached snapshots for useSyncExternalStore — MUST be stable references
let agentsSnapshotCache: GuildAgentStats[] = [];
let eventLogSnapshotCache: GuildEvent[] = [];

function rebuildSnapshots() {
  agentsSnapshotCache = KNOWN_AGENTS.map(
    (name) =>
      agents.get(name) ?? {
        name,
        level: 1,
        xp: 0,
        status: "offline",
        lastUpdate: 0,
      }
  );
  eventLogSnapshotCache = eventLog.slice();
}

// Initialize cache immediately
rebuildSnapshots();

// React-facing store subscriptions (useSyncExternalStore)
const storeListeners = new Set<() => void>();
function notifyStore() {
  rebuildSnapshots();
  storeListeners.forEach((l) => l());
}

function handleEvent(ev: GuildEvent) {
  // 1. Update derived agent state
  const name = (ev as { name?: string }).name;
  if (name && KNOWN_AGENTS.includes(name)) {
    const existing = agents.get(name) ?? {
      name,
      level: 1,
      xp: 0,
      lastUpdate: Date.now(),
    };
    const next: GuildAgentStats = { ...existing, lastUpdate: Date.now() };
    if (ev.type === "stats") {
      if (typeof (ev as { level?: number }).level === "number") next.level = (ev as { level: number }).level;
      if (typeof (ev as { xp?: number }).xp === "number") next.xp = (ev as { xp: number }).xp;
      if (typeof (ev as { hp?: number }).hp === "number") next.hp = (ev as { hp: number }).hp;
      if (typeof (ev as { mp?: number }).mp === "number") next.mp = (ev as { mp: number }).mp;
      next.status = "idle";
    } else if (ev.type === "quest_start") {
      next.current_quest = (ev as { quest: string }).quest;
      next.status = "working";
    } else if (ev.type === "quest_done") {
      next.current_quest = null;
      next.status = "idle";
      if (typeof (ev as { level?: number }).level === "number") next.level = (ev as { level: number }).level;
      if (typeof (ev as { xp?: number }).xp === "number") next.xp = (ev as { xp: number }).xp;
    } else if (ev.type === "quest_fail") {
      next.current_quest = null;
      next.status = "failed";
    } else if (ev.type === "cooldown") {
      next.status = "cooldown";
    }
    agents.set(name, next);
  }

  // 2. Append to rolling event log
  eventLog.push(ev);
  if (eventLog.length > MAX_LOG) eventLog.splice(0, eventLog.length - MAX_LOG);

  // 3. Fan out to live listeners + re-render store consumers
  listeners.forEach((l) => {
    try { l(ev); } catch { /* ignore */ }
  });
  notifyStore();
}

async function startSubscription(): Promise<void> {
  if (subscribed) return;
  subscribed = true;
  try {
    const ok = await invoke<boolean>("guild_status");
    available = ok;
    if (!ok) return;
    const channel = new Channel<{ line: string }>();
    channel.onmessage = (m) => {
      try {
        const ev = JSON.parse(m.line) as GuildEvent;
        handleEvent(ev);
      } catch {
        // drop malformed
      }
    };
    await invoke("guild_events", { onEvent: channel });
    started = true;
  } catch (err) {
    console.warn("[guild] subscription failed:", err);
    subscribed = false;
  }
}

// Eager boot — first time anyone imports this module. Defer to a
// microtask so an invoke failure can't crash React's initial render.
if (typeof queueMicrotask === "function") {
  queueMicrotask(() => {
    startSubscription().catch((err) => {
      console.warn("[guild] boot failed:", err);
    });
  });
} else {
  setTimeout(() => {
    startSubscription().catch((err) => {
      console.warn("[guild] boot failed:", err);
    });
  }, 0);
}

// ── Imperative API ────────────────────────────────────────────────────
export function isGuildAvailable(): boolean {
  return available === true;
}

export async function sendGuildCommand(cmd: object): Promise<void> {
  const line = JSON.stringify(cmd);
  await invoke("guild_command", { cmd: line });
}

export function subscribeGuildEvents(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getAgentsSnapshot(): GuildAgentStats[] {
  return agentsSnapshotCache;
}

export function getEventLogSnapshot(): GuildEvent[] {
  return eventLogSnapshotCache;
}

// ── React hooks ───────────────────────────────────────────────────────
function subscribeStore(onChange: () => void): () => void {
  storeListeners.add(onChange);
  return () => {
    storeListeners.delete(onChange);
  };
}

export function useGuildAgents(): GuildAgentStats[] {
  return useSyncExternalStore(subscribeStore, getAgentsSnapshot, getAgentsSnapshot);
}

export function useGuildEventLog(): GuildEvent[] {
  return useSyncExternalStore(subscribeStore, getEventLogSnapshot, getEventLogSnapshot);
}

export function useGuildStatus(): { available: boolean; started: boolean } {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const iv = setInterval(() => setTick((t) => t + 1), 1500);
    return () => clearInterval(iv);
  }, []);
  return { available: available === true, started };
  // eslint-disable-next-line react-hooks/exhaustive-deps
}

export function useGuildEvent(filter: (ev: GuildEvent) => boolean, handler: Listener) {
  const stable = useCallback(handler, [handler]);
  useEffect(() => subscribeGuildEvents((ev) => {
    if (filter(ev)) stable(ev);
  }), [stable, filter]);
}

export { KNOWN_AGENTS };
