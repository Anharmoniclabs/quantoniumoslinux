import { useState, useCallback } from "react";

export type AppId = "chat" | "monitor" | "validator" | "notes" | "settings" | "vault" | "simulator" | "visualizer" | "crypto";

export interface WindowState {
  appId: AppId;
  title: string;
  x: number;
  y: number;
  w: number;
  h: number;
  minimized: boolean;
  zIndex: number;
}

const APP_DEFAULTS: Record<AppId, { title: string; w: number; h: number }> = {
  chat:       { title: "Cael — AI Chat",        w: 680, h: 520 },
  monitor:    { title: "System Monitor",        w: 640, h: 500 },
  validator:  { title: "RFT Validator",         w: 580, h: 440 },
  notes:      { title: "Q-Notes",               w: 720, h: 500 },
  settings:   { title: "Settings",              w: 480, h: 360 },
  vault:      { title: "Q-Vault",               w: 660, h: 480 },
  simulator:  { title: "Quantum Simulator",     w: 720, h: 520 },
  visualizer: { title: "RFT Visualizer",        w: 740, h: 480 },
  crypto:     { title: "Quantum Crypto",        w: 700, h: 520 },
};

let nextZ = 100;

function cascade(count: number) {
  const base = 60;
  const step = 30;
  return { x: base + (count % 8) * step, y: base + (count % 8) * step };
}

export function useWindowManager() {
  const [windows, setWindows] = useState<WindowState[]>([]);

  const openApp = useCallback((appId: AppId) => {
    setWindows((prev) => {
      // If already open, just focus it
      const existing = prev.find((w) => w.appId === appId);
      if (existing) {
        nextZ++;
        return prev.map((w) =>
          w.appId === appId
            ? { ...w, minimized: false, zIndex: nextZ }
            : w
        );
      }
      // Open new
      const defaults = APP_DEFAULTS[appId];
      const pos = cascade(prev.length);
      nextZ++;
      return [
        ...prev,
        {
          appId,
          title: defaults.title,
          x: pos.x,
          y: pos.y,
          w: defaults.w,
          h: defaults.h,
          minimized: false,
          zIndex: nextZ,
        },
      ];
    });
  }, []);

  const closeApp = useCallback((appId: AppId) => {
    setWindows((prev) => prev.filter((w) => w.appId !== appId));
  }, []);

  const focusApp = useCallback((appId: AppId) => {
    nextZ++;
    setWindows((prev) =>
      prev.map((w) => (w.appId === appId ? { ...w, zIndex: nextZ } : w))
    );
  }, []);

  const minimizeApp = useCallback((appId: AppId) => {
    setWindows((prev) =>
      prev.map((w) => (w.appId === appId ? { ...w, minimized: true } : w))
    );
  }, []);

  return { windows, openApp, closeApp, focusApp, minimizeApp };
}
