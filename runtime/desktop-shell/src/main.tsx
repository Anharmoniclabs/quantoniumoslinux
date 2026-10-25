import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./theme.css";

// Pre-React breadcrumb: if React never mounts, at least we see why.
const rootEl = document.getElementById("root")!;
rootEl.innerHTML =
  '<div style="position:fixed;inset:0;display:flex;align-items:center;justify-content:center;color:#909090;font:12px system-ui;letter-spacing:1px">booting…</div>';

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { err: Error | null }
> {
  state = { err: null as Error | null };
  static getDerivedStateFromError(err: Error) {
    return { err };
  }
  componentDidCatch(err: Error, info: React.ErrorInfo) {
    console.error("[boot] render crash:", err, info);
  }
  render() {
    if (this.state.err) {
      return (
        <pre
          style={{
            position: "fixed",
            inset: 0,
            margin: 0,
            padding: 24,
            color: "#c04040",
            background: "#0a0a0a",
            font: "12px ui-monospace, monospace",
            whiteSpace: "pre-wrap",
            overflow: "auto",
            zIndex: 999999,
          }}
        >
          {"render crash:\n\n" + (this.state.err.stack || this.state.err.message)}
        </pre>
      );
    }
    return this.props.children;
  }
}

window.addEventListener("error", (e) => {
  console.error("[boot] window error:", e.error || e.message);
});
window.addEventListener("unhandledrejection", (e) => {
  console.error("[boot] unhandled rejection:", e.reason);
});

try {
  ReactDOM.createRoot(rootEl).render(
    <React.StrictMode>
      <ErrorBoundary>
        <App />
      </ErrorBoundary>
    </React.StrictMode>
  );
} catch (err) {
  rootEl.innerHTML =
    '<pre style="position:fixed;inset:0;margin:0;padding:24px;color:#c04040;background:#0a0a0a;font:12px ui-monospace,monospace;white-space:pre-wrap;overflow:auto">mount failed:\n\n' +
    String((err as Error)?.stack || err) +
    "</pre>";
}
