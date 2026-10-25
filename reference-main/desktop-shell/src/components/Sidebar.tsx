import type { AppId as ViewId } from "../hooks/useWindowManager";
import type { LucideIcon } from "lucide-react";
import {
  MessageCircle,
  Activity,
  ShieldCheck,
  StickyNote,
  Settings,
} from "lucide-react";
import "./Sidebar.css";

const NAV_ITEMS: { id: ViewId; icon: LucideIcon; label: string }[] = [
  { id: "chat",      icon: MessageCircle, label: "Chat" },
  { id: "monitor",   icon: Activity,    label: "Monitor" },
  { id: "validator", icon: ShieldCheck,  label: "Validator" },
  { id: "notes",     icon: StickyNote,  label: "Notes" },
];

interface SidebarProps {
  active: ViewId | null;
  openWindows: ViewId[];
  onNavigate: (id: ViewId) => void;
}

export function Sidebar({ openWindows, onNavigate }: SidebarProps) {
  return (
    <nav className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <img src="/que-icon.png" alt="Q" width={28} height={28} />
      </div>

      {/* Nav items */}
      <div className="sidebar-nav">
        {NAV_ITEMS.map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            className={`sidebar-btn ${openWindows.includes(id) ? "open" : ""}`}
            onClick={() => onNavigate(id)}
            title={label}
          >
            <Icon size={20} />
            <span className="sidebar-label">{label}</span>
          </button>
        ))}
      </div>

      {/* Bottom: settings */}
      <div className="sidebar-bottom">
        <button
          className={`sidebar-btn ${openWindows.includes("settings") ? "open" : ""}`}
          onClick={() => onNavigate("settings")}
          title="Settings"
        >
          <Settings size={20} />
          <span className="sidebar-label">Settings</span>
        </button>
      </div>
    </nav>
  );
}
