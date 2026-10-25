import { useState, useEffect, useCallback } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { AppWindow } from "./components/AppWindow";
import { Launchpad } from "./components/Launchpad";
import { BootScreen } from "./components/BootScreen";
import { DesktopClock } from "./components/DesktopClock";
import { ChatView } from "./views/ChatView";
import { MonitorView } from "./views/MonitorView";
import { ValidatorView } from "./views/ValidatorView";
import { NotesView } from "./views/NotesView";
import { SettingsView } from "./views/SettingsView";
import { VaultView } from "./views/VaultView";
import { SimulatorView } from "./views/SimulatorView";
import { VisualizerView } from "./views/VisualizerView";
import { CryptoView } from "./views/CryptoView";
import { GuildView } from "./views/GuildView";
import { ImageGenView } from "./views/ImageGenView";
import { GuildToasts } from "./components/GuildToasts";
import { useWindowManager } from "./hooks/useWindowManager";
import type { AppId } from "./hooks/useWindowManager";
import "./App.css";

const VIEW_MAP: Record<AppId, React.ComponentType> = {
  chat: ChatView,
  monitor: MonitorView,
  validator: ValidatorView,
  notes: NotesView,
  settings: SettingsView,
  vault: VaultView,
  simulator: SimulatorView,
  visualizer: VisualizerView,
  crypto: CryptoView,
  guild: GuildView,
  imagegen: ImageGenView,
};

export function App() {
  const [booted, setBooted] = useState(false);
  const { windows, openApp, closeApp, focusApp, minimizeApp } = useWindowManager();
  const hasVisibleWindows = windows.some((w) => !w.minimized);

  const handleLaunch = (id: AppId) => {
    openApp(id);
  };

  const handleBooted = useCallback(() => {
    setBooted(true);
  }, []);

  // ESC key toggles fullscreen ↔ windowed
  useEffect(() => {
    const handleKey = async (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        try {
          const win = getCurrentWindow();
          const isFs = await win.isFullscreen();
          if (isFs) {
            await win.setFullscreen(false);
            await win.setDecorations(true);
          } else {
            await win.setDecorations(false);
            await win.setFullscreen(true);
          }
        } catch {
          // Window API not available
        }
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, []);

  // Show boot screen until engine is ready
  if (!booted) {
    return <BootScreen onReady={handleBooted} />;
  }

  return (
    <div className="shell">
      <main className="shell-desktop">
        {/* Clock / date — top right */}
        <DesktopClock />

        {/* Center launchpad — visible when no windows are on top */}
        <Launchpad onLaunch={handleLaunch} visible={!hasVisibleWindows} />

        {/* Floating windows */}
        {windows.map((win) =>
          win.minimized ? null : (
            <AppWindow
              key={win.appId}
              id={win.appId}
              title={win.title}
              initialX={win.x}
              initialY={win.y}
              initialW={win.w}
              initialH={win.h}
              zIndex={win.zIndex}
              onFocus={() => focusApp(win.appId)}
              onClose={() => closeApp(win.appId)}
              onMinimize={() => minimizeApp(win.appId)}
            >
              {(() => {
                const View = VIEW_MAP[win.appId];
                return View ? <View /> : null;
              })()}
            </AppWindow>
          )
        )}
      </main>

      {/* Desktop toasts — fired by guild events across all apps */}
      <GuildToasts />

      {/* Taskbar for minimized windows */}
      {windows.some((w) => w.minimized) && (
        <div className="taskbar">
          {windows
            .filter((w) => w.minimized)
            .map((w) => (
              <button
                key={w.appId}
                className="taskbar-btn"
                onClick={() => {
                  focusApp(w.appId);
                  openApp(w.appId);
                }}
              >
                {w.title}
              </button>
            ))}
        </div>
      )}
    </div>
  );
}
