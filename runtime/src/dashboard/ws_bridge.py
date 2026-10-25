# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
WebSocket bridge — connects the Phaser browser dashboard to rpg_headless.py.

Architecture:
  [Browser/Phaser] <--WS--> [ws_bridge.py :9000] <--stdin/stdout--> [rpg_headless.py]

- HTTP serves static files from web/ on port 9000
- WebSocket upgrade at /ws streams JSON events bidirectionally
- Pure stdlib — no pip install needed (socket, threading, subprocess)
"""
from __future__ import annotations

import hashlib
import base64
import http.server
import json
import os
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

_SELF_DIR = Path(__file__).resolve().parent
_WEB_DIR = _SELF_DIR / "web"
_REPO_ROOT = _SELF_DIR.parents[1]
_HEADLESS = _REPO_ROOT / "src" / "rpg_headless.py"

PORT = 9000
WS_MAGIC = "258EAFA5-E914-47DA-95CA-5AB5DC799F07"


# ── WebSocket Framing ────────────────────────────────────────────

def _ws_accept_key(key: str) -> str:
    h = hashlib.sha1((key + WS_MAGIC).encode()).digest()
    return base64.b64encode(h).decode()


def _ws_encode_frame(payload: bytes, opcode: int = 0x1) -> bytes:
    """Encode a WebSocket frame (server→client, no mask)."""
    length = len(payload)
    header = bytes([0x80 | opcode])
    if length < 126:
        header += bytes([length])
    elif length < 65536:
        header += bytes([126]) + struct.pack("!H", length)
    else:
        header += bytes([127]) + struct.pack("!Q", length)
    return header + payload


def _ws_decode_frame(data: bytes) -> tuple[int, bytes, int]:
    """Decode a WebSocket frame. Returns (opcode, payload, total_consumed)."""
    if len(data) < 2:
        return -1, b"", 0
    opcode = data[0] & 0x0F
    masked = bool(data[1] & 0x80)
    length = data[1] & 0x7F
    offset = 2

    if length == 126:
        if len(data) < 4:
            return -1, b"", 0
        length = struct.unpack("!H", data[2:4])[0]
        offset = 4
    elif length == 127:
        if len(data) < 10:
            return -1, b"", 0
        length = struct.unpack("!Q", data[2:10])[0]
        offset = 10

    if masked:
        if len(data) < offset + 4:
            return -1, b"", 0
        mask = data[offset:offset + 4]
        offset += 4
    else:
        mask = None

    if len(data) < offset + length:
        return -1, b"", 0

    payload = data[offset:offset + length]
    if mask:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

    return opcode, payload, offset + length


# ── WebSocket Client Handler ────────────────────────────────────

class WSClient:
    def __init__(self, conn: socket.socket, addr):
        self.conn = conn
        self.addr = addr
        self.alive = True
        self._send_lock = threading.Lock()

    def send_json(self, obj: dict):
        if not self.alive:
            return
        try:
            data = json.dumps(obj, separators=(",", ":"), default=str).encode("utf-8")
            frame = _ws_encode_frame(data)
            with self._send_lock:
                self.conn.sendall(frame)
        except Exception:
            self.alive = False

    def close(self):
        self.alive = False
        try:
            self.conn.close()
        except Exception:
            pass


# ── Bridge Server ────────────────────────────────────────────────

class Bridge:
    def __init__(self, port: int = PORT, pack: str | None = None):
        self.port = port
        self._pack = pack
        self.clients: list[WSClient] = []
        self._clients_lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._cmd_queue: queue.Queue = queue.Queue()
        self._running = True
        # Cache events so late-joining browsers get the full state
        self._cached_ready: dict | None = None
        self._cached_stats: dict[str, dict] = {}  # agent name -> latest stats
        self._cached_walk: dict[str, dict] = {}   # agent name -> latest walk_to
        self._cache_lock = threading.Lock()

    def _trace_event(self, obj: dict):
        msg_type = obj.get("type", "")
        name = obj.get("name", "")
        if msg_type == "ready":
            agents = obj.get("agents", [])
            print(f"[event] ready agents={len(agents)} {', '.join(agents)}")
        elif msg_type == "system":
            print(f"[event] system {obj.get('msg', '')[:180]}")
        elif msg_type == "walk_to" and name:
            print(f"[event] walk_to {name} -> ({obj.get('x')}, {obj.get('y')})")
        elif msg_type == "quest_start" and name:
            print(f"[event] quest_start {name}: {obj.get('quest', '')[:160]}")
        elif msg_type == "quest_done" and name:
            print(f"[event] quest_done {name}: +{obj.get('xp', 0)}xp {obj.get('quest', '')[:140]}")
        elif msg_type == "quest_fail" and name:
            print(f"[event] quest_fail {name}: {obj.get('error', '')[:160]}")
        elif msg_type == "idle_action" and name:
            print(f"[event] idle_action {name}: {obj.get('result', '')[:160]}")
        elif msg_type == "chat_response":
            print(f"[event] chat_response {obj.get('from', 'SYSTEM')}: {obj.get('msg', '')[:180]}")
        elif msg_type == "log" and name:
            print(f"[event] log {name}: {obj.get('msg', '')[:180]}")

    def start(self):
        """Start headless process, HTTP server, and accept WS connections."""
        # Start rpg_headless.py
        self._start_headless()

        # Start stdout reader
        threading.Thread(target=self._read_stdout, daemon=True).start()

        # Start HTTP + WS server
        self._serve()

    def _start_headless(self):
        python = sys.executable
        cmd = [python, str(_HEADLESS)]
        if self._pack:
            cmd.extend(["--pack", self._pack])
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(_REPO_ROOT),
            text=True,
            bufsize=1,
        )
        # Stderr reader (log to console)
        def _err_reader():
            for line in self._process.stderr:
                print(f"[headless stderr] {line.rstrip()}", file=sys.stderr)
        threading.Thread(target=_err_reader, daemon=True).start()

    def _read_stdout(self):
        """Read JSON lines from headless stdout and broadcast to all WS clients."""
        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._trace_event(obj)
            self._broadcast(obj)

    def _broadcast(self, obj: dict):
        # Cache important events for late-joining clients
        msg_type = obj.get('type', '')
        with self._cache_lock:
            if msg_type == 'ready':
                self._cached_ready = obj
            elif msg_type == 'stats' and 'name' in obj:
                self._cached_stats[obj['name']] = obj
            elif msg_type == 'walk_to' and 'name' in obj:
                self._cached_walk[obj['name']] = obj

        with self._clients_lock:
            dead = []
            for client in self.clients:
                try:
                    client.send_json(obj)
                except Exception:
                    dead.append(client)
            for c in dead:
                c.alive = False
                self.clients.remove(c)

    def _send_to_headless(self, obj: dict):
        if self._process and self._process.stdin:
            try:
                cmd = obj.get("cmd", "")
                if cmd:
                    print(f"[bridge] browser cmd {cmd} {json.dumps(obj, separators=(',', ':'), default=str)[:180]}")
                line = json.dumps(obj, separators=(",", ":")) + "\n"
                self._process.stdin.write(line)
                self._process.stdin.flush()
            except Exception as e:
                print(f"[bridge] stdin write error: {e}", file=sys.stderr)

    def _serve(self):
        """Blocking HTTP+WS server on self.port."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind(("0.0.0.0", self.port))
        server_sock.listen(5)
        print(f"[bridge] Dashboard at  http://localhost:{self.port}")
        print(f"[bridge] WebSocket at  ws://localhost:{self.port}/ws")

        while self._running:
            try:
                conn, addr = server_sock.accept()
            except Exception:
                break
            threading.Thread(target=self._handle_connection, args=(conn, addr),
                             daemon=True).start()

    def _handle_connection(self, conn: socket.socket, addr):
        """Handle an incoming TCP connection — HTTP or WebSocket upgrade."""
        try:
            data = conn.recv(4096)
            if not data:
                conn.close()
                return

            request = data.decode("utf-8", errors="replace")
            first_line = request.split("\r\n")[0]
            method, path, _ = first_line.split(" ", 2)

            # Parse headers
            headers = {}
            for line in request.split("\r\n")[1:]:
                if ": " in line:
                    k, v = line.split(": ", 1)
                    headers[k.lower()] = v

            # WebSocket upgrade?
            if "upgrade" in headers.get("connection", "").lower() and path == "/ws":
                self._ws_upgrade(conn, addr, headers)
                return

            # Static file serving
            self._serve_file(conn, path)

        except Exception as e:
            print(f"[bridge] connection error: {e}", file=sys.stderr)
            try:
                conn.close()
            except Exception:
                pass

    def _ws_upgrade(self, conn: socket.socket, addr, headers: dict):
        """Complete WebSocket handshake and enter message loop."""
        ws_key = headers.get("sec-websocket-key", "")
        accept = _ws_accept_key(ws_key)

        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        )
        conn.sendall(response.encode())
        print(f"[bridge] websocket client connected from {addr[0]}:{addr[1]}")

        client = WSClient(conn, addr)
        with self._clients_lock:
            self.clients.append(client)

        # Replay cached state so late-joining browsers see current world
        with self._cache_lock:
            if self._cached_ready:
                client.send_json(self._cached_ready)
            for stats in self._cached_stats.values():
                client.send_json(stats)
            # Replay last known agent positions so reconnects don't snap-home
            for walk in self._cached_walk.values():
                client.send_json(walk)

        # Read messages from browser
        buf = b""
        while client.alive and self._running:
            try:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk

                while len(buf) >= 2:
                    opcode, payload, consumed = _ws_decode_frame(buf)
                    if consumed == 0:
                        break
                    buf = buf[consumed:]

                    if opcode == 0x8:  # close
                        client.alive = False
                        break
                    elif opcode == 0x9:  # ping
                        pong = _ws_encode_frame(payload, opcode=0xA)
                        conn.sendall(pong)
                    elif opcode == 0x1:  # text
                        try:
                            cmd = json.loads(payload.decode("utf-8"))
                            self._send_to_headless(cmd)
                        except json.JSONDecodeError:
                            pass

            except Exception:
                break

        client.close()
        with self._clients_lock:
            if client in self.clients:
                self.clients.remove(client)

    def _serve_file(self, conn: socket.socket, path: str):
        """Serve a static file from web/."""
        if path == "/" or path == "":
            path = "/index.html"

        # Security: no traversal
        clean = path.lstrip("/").replace("\\", "/")
        if ".." in clean:
            self._send_http(conn, 403, "text/plain", b"Forbidden")
            return

        file_path = _WEB_DIR / clean
        if not file_path.exists() or not file_path.is_file():
            self._send_http(conn, 404, "text/plain", b"Not Found")
            return

        # MIME types
        ext = file_path.suffix.lower()
        mime = {
            ".html": "text/html",
            ".js": "application/javascript",
            ".css": "text/css",
            ".json": "application/json",
            ".png": "image/png",
            ".ico": "image/x-icon",
            ".svg": "image/svg+xml",
        }.get(ext, "application/octet-stream")

        try:
            content = file_path.read_bytes()
            self._send_http(conn, 200, mime, content)
        except Exception:
            self._send_http(conn, 500, "text/plain", b"Internal Error")

    @staticmethod
    def _send_http(conn: socket.socket, status: int, content_type: str, body: bytes):
        reason = {200: "OK", 403: "Forbidden", 404: "Not Found", 500: "Error"}.get(status, "OK")
        header = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n"
            "Pragma: no-cache\r\n"
            "Expires: 0\r\n"
            "Connection: close\r\n"
            "\r\n"
        )
        try:
            conn.sendall(header.encode() + body)
        finally:
            conn.close()

    def shutdown(self):
        self._running = False
        if self._process:
            self._send_to_headless({"cmd": "quit"})
            self._process.wait(timeout=5)
        with self._clients_lock:
            for c in self.clients:
                c.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="QUE Guild Dashboard Bridge")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port (default {PORT})")
    parser.add_argument("--pack", default=None, help=".rftmwpk pack for native C++ engine")
    args = parser.parse_args()

    bridge = Bridge(port=args.port, pack=args.pack)
    try:
        bridge.start()
    except KeyboardInterrupt:
        print("\n[bridge] Shutting down...")
        bridge.shutdown()


if __name__ == "__main__":
    main()
