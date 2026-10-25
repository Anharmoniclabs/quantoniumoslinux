// Desktop-wide toast surface for guild events.
// Fires on quest_done / quest_fail / cooldown / ready — the signals a
// user might miss when GuildView isn't focused. Auto-dismisses.

import { useEffect, useState, useCallback } from "react";
import { subscribeGuildEvents, type GuildEvent } from "../hooks/useGuild";
import "../views/GuildView.css";

interface Toast {
  id: string;
  kind: "ok" | "err" | "q" | "i";
  agent: string;
  text: string;
}

const DISMISS_MS = 5_000;
const MAX_TOASTS = 4;

function buildToast(ev: GuildEvent): Toast | null {
  switch ((ev as { type?: string }).type) {
    case "ready":
      return {
        id: crypto.randomUUID(),
        kind: "q",
        agent: "guild",
        text: `online — ${((ev as { agents?: string[] }).agents ?? []).length} agents ready`,
      };
    case "quest_done": {
      const e = ev as { name: string; quest: string; xp: number };
      return {
        id: crypto.randomUUID(),
        kind: "ok",
        agent: e.name,
        text: `✓ ${e.quest}  (+${e.xp} xp)`,
      };
    }
    case "quest_fail": {
      const e = ev as { name: string; quest: string; error: string };
      return {
        id: crypto.randomUUID(),
        kind: "err",
        agent: e.name,
        text: `✗ ${e.quest}: ${e.error}`,
      };
    }
    case "cooldown": {
      const e = ev as { name: string; reason: string };
      return {
        id: crypto.randomUUID(),
        kind: "err",
        agent: e.name,
        text: `cooldown — ${e.reason}`,
      };
    }
    default:
      return null;
  }
}

export function GuildToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  useEffect(() => {
    return subscribeGuildEvents((ev) => {
      const toast = buildToast(ev);
      if (!toast) return;
      setToasts((prev) => {
        const next = [...prev, toast];
        return next.length > MAX_TOASTS ? next.slice(next.length - MAX_TOASTS) : next;
      });
      setTimeout(() => dismiss(toast.id), DISMISS_MS);
    });
  }, [dismiss]);

  if (toasts.length === 0) return null;

  return (
    <div className="guild-toast-container">
      {toasts.map((t) => (
        <div
          key={t.id}
          className={`guild-toast ${t.kind}`}
          onClick={() => dismiss(t.id)}
          role="status"
        >
          <div className="guild-toast-head">
            <span>{t.agent}</span>
            <span>guild</span>
          </div>
          <div className="guild-toast-body">{t.text}</div>
        </div>
      ))}
    </div>
  );
}
