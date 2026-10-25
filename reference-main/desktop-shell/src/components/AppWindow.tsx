import { useState, useRef, useCallback, useEffect } from "react";
import { X, Minus, Maximize2, Minimize2 } from "lucide-react";
import "./AppWindow.css";

interface AppWindowProps {
  id: string;
  title: string;
  children: React.ReactNode;
  initialX?: number;
  initialY?: number;
  initialW?: number;
  initialH?: number;
  zIndex: number;
  onFocus: () => void;
  onClose: () => void;
  onMinimize: () => void;
}

export function AppWindow({
  id,
  title,
  children,
  initialX = 80,
  initialY = 40,
  initialW = 700,
  initialH = 500,
  zIndex,
  onFocus,
  onClose,
  onMinimize,
}: AppWindowProps) {
  const [pos, setPos] = useState({ x: initialX, y: initialY });
  const [size, setSize] = useState({ w: initialW, h: initialH });
  const [maximized, setMaximized] = useState(false);
  const [preMax, setPreMax] = useState({ x: initialX, y: initialY, w: initialW, h: initialH });
  const dragging = useRef(false);
  const resizing = useRef(false);
  const offset = useRef({ x: 0, y: 0 });

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      if ((e.target as HTMLElement).closest(".window-controls")) return;
      onFocus();
      dragging.current = true;
      offset.current = { x: e.clientX - pos.x, y: e.clientY - pos.y };
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    },
    [pos, onFocus]
  );

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (dragging.current) {
        setPos({ x: e.clientX - offset.current.x, y: Math.max(0, e.clientY - offset.current.y) });
        if (maximized) setMaximized(false);
      }
      if (resizing.current) {
        const newW = Math.max(360, e.clientX - pos.x);
        const newH = Math.max(240, e.clientY - pos.y);
        setSize({ w: newW, h: newH });
      }
    },
    [pos, maximized]
  );

  const handlePointerUp = useCallback(() => {
    dragging.current = false;
    resizing.current = false;
  }, []);

  const handleResizeDown = useCallback(
    (e: React.PointerEvent) => {
      e.stopPropagation();
      onFocus();
      resizing.current = true;
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    },
    [onFocus]
  );

  const toggleMaximize = () => {
    if (maximized) {
      setPos({ x: preMax.x, y: preMax.y });
      setSize({ w: preMax.w, h: preMax.h });
      setMaximized(false);
    } else {
      setPreMax({ ...pos, ...size });
      setPos({ x: 0, y: 0 });
      setSize({ w: window.innerWidth, h: window.innerHeight });
      setMaximized(true);
    }
  };

  const handleTitleDblClick = () => toggleMaximize();

  return (
    <div
      className={`app-window ${maximized ? "maximized" : ""}`}
      style={{
        left: pos.x,
        top: pos.y,
        width: size.w,
        height: size.h,
        zIndex,
      }}
      onPointerDown={() => onFocus()}
    >
      {/* Title bar */}
      <div
        className="window-titlebar"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onDoubleClick={handleTitleDblClick}
      >
        <span className="window-title">{title}</span>
        <div className="window-controls">
          <button className="win-btn" onClick={onMinimize} title="Minimize">
            <Minus size={12} />
          </button>
          <button className="win-btn" onClick={toggleMaximize} title="Maximize">
            {maximized ? <Minimize2 size={12} /> : <Maximize2 size={12} />}
          </button>
          <button className="win-btn close" onClick={onClose} title="Close">
            <X size={12} />
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="window-body">{children}</div>

      {/* Resize handle */}
      {!maximized && (
        <div
          className="window-resize"
          onPointerDown={handleResizeDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
        />
      )}
    </div>
  );
}
