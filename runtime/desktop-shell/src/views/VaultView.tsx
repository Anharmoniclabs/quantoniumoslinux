import { useState, useRef } from "react";
import { Lock, Unlock, Plus, Trash2, Search, Copy, Eye, EyeOff } from "lucide-react";
import "./VaultView.css";

interface VaultItem {
  id: string;
  title: string;
  body: string;
  updatedAt: number;
}

function newItem(): VaultItem {
  return { id: crypto.randomUUID(), title: "Untitled", body: "", updatedAt: Date.now() };
}

export function VaultView() {
  const [locked, setLocked] = useState(true);
  const [password, setPassword] = useState("");
  const [pwError, setPwError] = useState(false);
  const [items, setItems] = useState<VaultItem[]>([newItem()]);
  const [activeId, setActiveId] = useState(items[0].id);
  const [search, setSearch] = useState("");
  const [showBody, setShowBody] = useState(false);
  const [rftMixer, setRftMixer] = useState(false);
  const pwRef = useRef<HTMLInputElement>(null);

  const active = items.find((i) => i.id === activeId);

  const handleUnlock = () => {
    if (password.length >= 4) {
      setLocked(false);
      setPwError(false);
    } else {
      setPwError(true);
      pwRef.current?.focus();
    }
  };

  const handleLock = () => {
    setLocked(true);
    setPassword("");
    setShowBody(false);
  };

  const handleCreate = () => {
    const item = newItem();
    setItems((prev) => [item, ...prev]);
    setActiveId(item.id);
  };

  const handleDelete = (id: string) => {
    setItems((prev) => prev.filter((i) => i.id !== id));
    if (activeId === id) setActiveId(items.find((i) => i.id !== id)?.id ?? "");
  };

  const handleCopy = () => {
    if (active) navigator.clipboard.writeText(active.body);
  };

  const filtered = items.filter(
    (i) =>
      i.title.toLowerCase().includes(search.toLowerCase()) ||
      i.body.toLowerCase().includes(search.toLowerCase())
  );

  if (locked) {
    return (
      <div className="vault-view vault-lock-screen">
        <div className="vault-lock-card">
          <Lock size={40} />
          <h2>Q-Vault</h2>
          <p className="dim">AES-256 Encrypted Storage</p>
          <input
            ref={pwRef}
            type="password"
            className={`vault-pw-input ${pwError ? "error" : ""}`}
            placeholder="Master password"
            value={password}
            onChange={(e) => { setPassword(e.target.value); setPwError(false); }}
            onKeyDown={(e) => e.key === "Enter" && handleUnlock()}
            autoFocus
          />
          {pwError && <span className="vault-pw-error">Min 4 characters</span>}
          <button className="vault-unlock-btn" onClick={handleUnlock}>
            <Unlock size={14} /> Unlock
          </button>
          <label className="vault-rft-toggle">
            <input type="checkbox" checked={rftMixer} onChange={() => setRftMixer(!rftMixer)} />
            RFT Keystream Mixer
          </label>
        </div>
      </div>
    );
  }

  return (
    <div className="vault-view">
      <div className="vault-sidebar">
        <div className="vault-sidebar-top">
          <div className="vault-search-row">
            <Search size={13} />
            <input
              className="vault-search"
              placeholder="Search…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <button className="vault-add-btn" onClick={handleCreate} title="New item">
            <Plus size={14} />
          </button>
        </div>
        <div className="vault-list">
          {filtered.map((item) => (
            <button
              key={item.id}
              className={`vault-list-item ${item.id === activeId ? "active" : ""}`}
              onClick={() => setActiveId(item.id)}
            >
              <Lock size={11} />
              <span>{item.title || "Untitled"}</span>
            </button>
          ))}
        </div>
        <button className="vault-lock-btn" onClick={handleLock}>
          <Lock size={12} /> Lock Vault
        </button>
      </div>

      <div className="vault-editor">
        {active ? (
          <>
            <div className="vault-editor-toolbar">
              <input
                className="vault-title-input"
                value={active.title}
                onChange={(e) =>
                  setItems((prev) =>
                    prev.map((i) =>
                      i.id === active.id ? { ...i, title: e.target.value, updatedAt: Date.now() } : i
                    )
                  )
                }
                placeholder="Title"
              />
              <div className="vault-toolbar-actions">
                <button className="vault-icon-btn" onClick={() => setShowBody(!showBody)} title={showBody ? "Hide" : "Reveal"}>
                  {showBody ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
                <button className="vault-icon-btn" onClick={handleCopy} title="Copy">
                  <Copy size={14} />
                </button>
                <button className="vault-icon-btn danger" onClick={() => handleDelete(active.id)} title="Delete">
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
            <textarea
              className={`vault-body ${showBody ? "" : "masked"}`}
              value={active.body}
              onChange={(e) =>
                setItems((prev) =>
                  prev.map((i) =>
                    i.id === active.id ? { ...i, body: e.target.value, updatedAt: Date.now() } : i
                  )
                )
              }
              placeholder="Secret content…"
            />
          </>
        ) : (
          <div className="vault-empty">No item selected</div>
        )}
      </div>
    </div>
  );
}
