import { useState, useEffect } from "react";
import { Mic, MicOff, Shield } from "lucide-react";
import "./SettingsView.css";

type PermState = "unknown" | "granted" | "denied" | "requesting";

export function SettingsView() {
  const [micPerm, setMicPerm] = useState<PermState>("unknown");
  const [micError, setMicError] = useState("");

  // Check current permission state on mount
  useEffect(() => {
    // Don't trust cached denials — always allow retry
    const stored = localStorage.getItem("qos-mic-permission");
    if (stored === "granted") {
      setMicPerm("granted");
    }
    // Note: skip Permissions API query — WebView2 may report stale denials
  }, []);

  const toggleMicPermission = async () => {
    if (micPerm === "granted") {
      // Can't programmatically revoke — just clear our state
      setMicPerm("unknown");
      localStorage.removeItem("qos-mic-permission");
      return;
    }

    setMicPerm("requesting");
    setMicError("");

    // Check if mediaDevices API exists
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setMicError("mediaDevices API not available (insecure context?)");
      setMicPerm("denied");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((t) => t.stop());
      setMicPerm("granted");
      setMicError("");
      localStorage.setItem("qos-mic-permission", "granted");
    } catch (e: any) {
      const msg = `${e.name}: ${e.message}`;
      console.error("[Settings] Microphone permission denied:", msg, e);
      setMicError(msg);
      setMicPerm("denied");
      localStorage.setItem("qos-mic-permission", "denied");
    }
  };

  return (
    <div className="settings-view">
      <header className="settings-header">
        <h1>Settings</h1>
      </header>

      <section className="settings-section">
        <h2><Shield size={12} style={{ marginRight: 6, verticalAlign: -1 }} />Permissions</h2>
        <div className="settings-row">
          <div className="settings-perm-info">
            <span className="settings-label">
              {micPerm === "granted" ? <Mic size={14} /> : <MicOff size={14} />}
              <span style={{ marginLeft: 8 }}>Microphone</span>
            </span>
            <span className="settings-hint">
              {micPerm === "granted"
                ? "Voice input enabled for Cael chat"
                : micPerm === "denied"
                ? micError || "Denied — reset in system settings"
                : "Required for voice chat with Cael"}
            </span>
          </div>
          <button
            className={`perm-toggle ${micPerm === "granted" ? "active" : ""} ${micPerm === "requesting" ? "requesting" : ""}`}
            onClick={toggleMicPermission}
            disabled={micPerm === "requesting"}
          >
            <span className="perm-toggle-knob" />
          </button>
        </div>
      </section>

      <section className="settings-section">
        <h2>Engine</h2>
        <div className="settings-row">
          <span className="settings-label">Backend URL</span>
          <input className="settings-input" defaultValue="http://127.0.0.1:8000" readOnly />
        </div>
      </section>

      <section className="settings-section">
        <h2>About</h2>
        <div className="settings-about">
          <p><strong>QuantoniumOS Desktop Shell</strong></p>
          <p>Built with Tauri + React</p>
          <p className="dim">v0.1.0-alpha</p>
        </div>
      </section>
    </div>
  );
}
