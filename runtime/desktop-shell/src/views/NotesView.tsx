import { useState } from "react";
import { Plus, Search, FileText, Trash2 } from "lucide-react";
import "./NotesView.css";

interface Note {
  id: string;
  title: string;
  body: string;
  updatedAt: number;
}

function newNote(): Note {
  return {
    id: crypto.randomUUID(),
    title: "Untitled",
    body: "",
    updatedAt: Date.now(),
  };
}

export function NotesView() {
  const [notes, setNotes] = useState<Note[]>([newNote()]);
  const [activeId, setActiveId] = useState<string>(notes[0].id);
  const [search, setSearch] = useState("");

  const active = notes.find((n) => n.id === activeId);

  const handleCreate = () => {
    const n = newNote();
    setNotes((prev) => [n, ...prev]);
    setActiveId(n.id);
  };

  const handleDelete = (id: string) => {
    setNotes((prev) => prev.filter((n) => n.id !== id));
    if (activeId === id) {
      setActiveId(notes.find((n) => n.id !== id)?.id ?? "");
    }
  };

  const handleEdit = (field: "title" | "body", value: string) => {
    setNotes((prev) =>
      prev.map((n) =>
        n.id === activeId ? { ...n, [field]: value, updatedAt: Date.now() } : n
      )
    );
  };

  const filtered = notes.filter(
    (n) =>
      n.title.toLowerCase().includes(search.toLowerCase()) ||
      n.body.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="notes-view">
      {/* Sidebar */}
      <aside className="notes-sidebar">
        <div className="notes-sb-header">
          <span className="notes-sb-title">Notes</span>
          <button className="icon-btn" onClick={handleCreate} title="New note">
            <Plus size={15} />
          </button>
        </div>

        <div className="notes-search-wrap">
          <Search size={13} className="notes-search-icon" />
          <input
            className="notes-search"
            placeholder="Search…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <div className="notes-list">
          {filtered.map((n) => (
            <div
              key={n.id}
              className={`notes-item ${n.id === activeId ? "active" : ""}`}
              onClick={() => setActiveId(n.id)}
            >
              <FileText size={13} className="notes-item-icon" />
              <div className="notes-item-info">
                <span className="notes-item-title">{n.title || "Untitled"}</span>
                <span className="notes-item-preview">{n.body.slice(0, 50) || "Empty note"}</span>
              </div>
              <button
                className="notes-item-del"
                onClick={(e) => { e.stopPropagation(); handleDelete(n.id); }}
                title="Delete"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* Editor */}
      <div className="notes-editor">
        {active ? (
          <>
            <input
              className="notes-title-input"
              value={active.title}
              onChange={(e) => handleEdit("title", e.target.value)}
              placeholder="Title"
            />
            <textarea
              className="notes-body-input"
              value={active.body}
              onChange={(e) => handleEdit("body", e.target.value)}
              placeholder="Start typing…"
            />
          </>
        ) : (
          <div className="notes-empty">Select or create a note</div>
        )}
      </div>
    </div>
  );
}
