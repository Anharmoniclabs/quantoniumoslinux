# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Chatbox custom widgets — TypingDots, ChatBubble, EmptyState.
"""

from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import (
    QColor, QPainter, QPen, QFont, QPainterPath, QBrush, QPixmap,
)
from PyQt5.QtWidgets import QWidget, QSizePolicy, QLabel, QVBoxLayout, QHBoxLayout

from PyQt5.QtCore import QSize

from src.apps.chatbox_theme import parse_rgba


# ── Typing indicator (3-dot pulse animation) ───────────────────────

class TypingDots(QWidget):
    """Animated 3-dot indicator matching mobile TypingIndicator.tsx."""

    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self._color = color
        self._opacities = [0.3, 0.3, 0.3]
        self._step = 0
        self.setFixedSize(60, 30)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(200)

    def set_color(self, c: QColor):
        self._color = c
        self.update()

    def _tick(self):
        self._step = (self._step + 1) % 6
        for i in range(3):
            phase = (self._step - i) % 6
            self._opacities[i] = 1.0 if phase < 2 else 0.3
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx = self.width() // 2
        cy = self.height() // 2
        r = 3.5
        gap = 12
        for i in range(3):
            c = QColor(self._color)
            c.setAlphaF(self._opacities[i])
            p.setBrush(c)
            p.setPen(Qt.NoPen)
            x = cx + (i - 1) * gap
            p.drawEllipse(QRectF(x - r, cy - r, r * 2, r * 2))
        p.end()

    def stop(self):
        self._timer.stop()


# ── Thinking / progress panel ──────────────────────────────────────

class ThinkingPanel(QWidget):
    """ChatGPT/Claude-style live status panel shown while the model works.

    Displays the current phase ("Encoding prompt", "Prefilling 42/128",
    "Generating · 17 tok · 0.4 tok/s"), ticks an elapsed-time counter,
    and can collapse to a compact footer once generation completes.
    """

    def __init__(self, colors: dict, parent=None):
        super().__init__(parent)
        self._colors = colors
        self._stage = "thinking"
        self._info: dict = {}
        self._stats: dict = {}
        self._t0 = None  # set on first show
        self._final = False
        self._final_text = ""
        self._reasoning_text = ""  # accumulated R1 reasoning tokens

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 4, 18, 4)
        lay.setSpacing(10)

        dot_color = parse_rgba(colors["glow"])
        self._dots = TypingDots(dot_color)
        lay.addWidget(self._dots)

        self._label = QLabel("Thinking…")
        self._label.setObjectName("ThinkingLabel")
        self._label.setStyleSheet(
            f"color: {colors['muted']}; font-size: 11px; "
            f"font-family: 'Inter', 'Segoe UI', sans-serif; font-style: italic;"
        )
        lay.addWidget(self._label, 1)

        # 10-Hz tick to keep the elapsed counter moving
        import time
        self._time_mod = time
        self._t0 = time.perf_counter()
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._refresh)
        self._tick_timer.start(100)

    def set_colors(self, colors: dict):
        self._colors = colors
        self._label.setStyleSheet(
            f"color: {colors['muted']}; font-size: 11px; "
            f"font-family: 'Inter', 'Segoe UI', sans-serif; font-style: italic;"
        )
        self._dots.set_color(parse_rgba(colors["glow"]))

    def set_stage(self, stage: str, info: dict):
        self._stage = stage
        self._info = dict(info or {})
        self._refresh()

    def set_stats(self, stats: dict):
        self._stats = dict(stats or {})
        self._refresh()

    def append_reasoning(self, token: str):
        """Append an R1 reasoning token — shown live in the status bar."""
        self._reasoning_text += token
        self._stage = "reasoning"
        self._refresh()

    def finalize(self):
        """Freeze into a compact metadata footer."""
        self._final = True
        self._tick_timer.stop()
        self._dots.stop()
        self._dots.hide()

        n_prompt = self._stats.get("n_prompt", 0)
        n_output = self._stats.get("n_output", 0)
        tps = self._stats.get("tokens_per_sec", 0.0)
        elapsed = self._stats.get("elapsed_s", 0.0)
        parts = []
        if n_output:
            parts.append(f"{n_output} tok")
        if tps:
            parts.append(f"{tps:.1f} tok/s")
        if n_prompt:
            parts.append(f"prompt {n_prompt}")
        if elapsed:
            parts.append(f"{elapsed:.1f}s")
        self._final_text = " · ".join(parts) if parts else ""
        self._label.setText(self._final_text)
        # Collapse to a tight single line
        self.layout().setContentsMargins(18, 0, 18, 0)
        self.setFixedHeight(20)

    def _refresh(self):
        if self._final:
            return
        elapsed = self._time_mod.perf_counter() - (self._t0 or self._time_mod.perf_counter())
        stage = self._stage
        info = self._info
        stats = self._stats

        if stage == "thinking":
            text = f"Thinking… {elapsed:0.1f}s"
        elif stage == "encoding":
            text = f"Encoding prompt… {elapsed:0.1f}s"
        elif stage == "encoded":
            n = info.get("n_tokens", 0)
            text = f"Prompt encoded · {n} tokens · {elapsed:0.1f}s"
        elif stage in ("prefill_progress", "prefill"):
            val = info.get("value", info.get("done", 0))
            total = info.get("total", 0)
            cached = info.get("cached", 0)
            if total:
                text = f"Prefilling · {val}/{total} · {elapsed:0.1f}s"
            elif val:
                text = f"Prefilling · {val} tokens · {elapsed:0.1f}s"
            else:
                text = f"Prefilling… {elapsed:0.1f}s"
            if cached > 0:
                text = text.replace(f" · {elapsed:0.1f}s", f" · {cached} cached · {elapsed:0.1f}s")
        elif stage in ("decode_start", "decode"):
            text = f"Generating… {elapsed:0.1f}s"
        elif stage == "decoding":
            n_out = stats.get("n_output", 0)
            tps = stats.get("tokens_per_sec", 0.0)
            last = str(stats.get("last_token", "") or "")
            last = last.replace("\n", "\\n").replace("\r", "\\r")
            if len(last) > 18:
                last = last[:15] + "..."
            token_part = f" · token `{last}`" if last else ""
            if tps > 0:
                text = f"Generating · {n_out} tok · {tps:.1f} tok/s{token_part} · {elapsed:0.1f}s"
            else:
                text = f"Generating · {n_out} tok{token_part} · {elapsed:0.1f}s"
        elif stage == "tool_call":
            detail = info.get("detail", "")
            text = f"🔧 Calling tool · {detail} · {elapsed:0.1f}s"
        elif stage == "tool_result":
            detail = info.get("detail", "")
            text = f"✓ Tool result · {detail[:60]} · {elapsed:0.1f}s"
        elif stage == "reasoning":
            # Show last ~80 chars of R1's reasoning in the status bar
            snippet = self._reasoning_text.replace("\n", " ").replace("\r", "")
            if len(snippet) > 80:
                snippet = "…" + snippet[-77:]
            n_tok = len(self._reasoning_text.split())
            text = f"💭 Reasoning · {n_tok} tok · {snippet} · {elapsed:0.1f}s"
        else:
            text = f"{stage}… {elapsed:0.1f}s"
        self._label.setText(text)


# ── Chat bubble ────────────────────────────────────────────────────

class ChatBubble(QWidget):
    """Single message bubble matching mobile ChatBubble.tsx.

    Features:
    - Asymmetric border-radius (sharp corner on sender side)
    - Semi-transparent glass background
    - 1px colored border
    - Timestamp + optional confidence badge
    """

    def __init__(self, text: str, is_user: bool, colors: dict, timestamp: str = "",
                 confidence: float = 0, parent=None):
        super().__init__(parent)
        self._text = text
        self._is_user = is_user
        self._colors = colors
        self._timestamp = timestamp
        self._confidence = confidence
        self._cached_lines = None  # invalidated on text change
        self._cached_max_w = 0
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.setMinimumHeight(44)

    def set_text(self, text: str):
        self._text = text
        self._cached_lines = None
        self.updateGeometry()
        self.update()

    def append_text(self, token: str):
        self._text += token
        self._cached_lines = None
        self.updateGeometry()
        self.update()

    def sizeHint(self):
        fm = self.fontMetrics()
        max_w = int(self.parent().width() * 0.78) if self.parent() else 500
        max_w = max(max_w, 200)
        pad_h, pad_v = 16, 11
        lines = self._get_wrapped_lines(fm, max_w - 2 * pad_h)
        line_h = int(fm.height() * 1.47)
        text_h = max(len(lines), 1) * line_h
        ts_h = 18 if self._timestamp else 0
        total_h = pad_v + 9 + text_h + ts_h + 4
        return QSize(max_w + 28, max(36, total_h))

    def _get_wrapped_lines(self, fm, max_w):
        """Return cached wrapped lines, recomputing only when text or width changed."""
        if self._cached_lines is not None and self._cached_max_w == max_w:
            return self._cached_lines
        self._cached_lines = self._wrap_lines(fm, max_w)
        self._cached_max_w = max_w
        return self._cached_lines

    def _wrap_lines(self, fm, max_w):
        lines = []
        for paragraph in self._text.split("\n"):
            if not paragraph:
                lines.append("")
                continue
            words = paragraph.split(" ")
            cur = ""
            for w in words:
                test = (cur + " " + w).strip() if cur else w
                if fm.horizontalAdvance(test) > max_w and cur:
                    lines.append(cur)
                    cur = w
                else:
                    cur = test
            if cur:
                lines.append(cur)
        return lines or [""]

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        fm = self.fontMetrics()
        rect = self.rect().adjusted(14, 5, -14, -5)
        max_bw = int(rect.width() * 0.78)
        pad_h, pad_top, pad_bot = 16, 11, 9

        lines = self._get_wrapped_lines(fm, max_bw - 2 * pad_h)
        line_h = int(fm.height() * 1.47)
        text_h = len(lines) * line_h
        max_line_w = max(fm.horizontalAdvance(l) for l in lines) if lines else 100
        bw = min(max_bw, max_line_w + 2 * pad_h + 8)
        ts_h = 18 if self._timestamp else 0
        bh = pad_top + pad_bot + text_h + ts_h

        if self._is_user:
            bx = rect.right() - bw
        else:
            bx = rect.left()
        by = rect.top()

        c = self._colors
        if self._is_user:
            bg = parse_rgba(c["userBubble"])
            border = parse_rgba(c["userBubbleBorder"])
            fg = parse_rgba(c["userBubbleText"])
        else:
            bg = parse_rgba(c["aiBubble"])
            border = parse_rgba(c["aiBubbleBorder"])
            fg = parse_rgba(c["aiBubbleText"])

        # Draw bubble with asymmetric corners
        path = QPainterPath()
        R = 20.0
        sharp = 4.0
        br = QRectF(bx, by, bw, bh)

        if self._is_user:
            # Sharp top-right corner
            path.moveTo(br.left() + R, br.top())
            path.lineTo(br.right() - sharp, br.top())
            path.lineTo(br.right(), br.top() + sharp)
            path.lineTo(br.right(), br.bottom() - R)
            path.arcTo(br.right() - 2 * R, br.bottom() - 2 * R, 2 * R, 2 * R, 0, -90)
            path.lineTo(br.left() + R, br.bottom())
            path.arcTo(br.left(), br.bottom() - 2 * R, 2 * R, 2 * R, 270, -90)
            path.lineTo(br.left(), br.top() + R)
            path.arcTo(br.left(), br.top(), 2 * R, 2 * R, 180, -90)
        else:
            # Sharp top-left corner
            path.moveTo(br.left() + sharp, br.top())
            path.lineTo(br.right() - R, br.top())
            path.arcTo(br.right() - 2 * R, br.top(), 2 * R, 2 * R, 90, -90)
            path.lineTo(br.right(), br.bottom() - R)
            path.arcTo(br.right() - 2 * R, br.bottom() - 2 * R, 2 * R, 2 * R, 0, -90)
            path.lineTo(br.left() + R, br.bottom())
            path.arcTo(br.left(), br.bottom() - 2 * R, 2 * R, 2 * R, 270, -90)
            path.lineTo(br.left(), br.top() + sharp)
            path.lineTo(br.left() + sharp, br.top())

        p.fillPath(path, bg)
        p.setPen(QPen(border, 1))
        p.drawPath(path)

        # Draw text
        p.setPen(fg)
        tx = bx + pad_h
        ty = by + pad_top + fm.ascent()
        for i, line in enumerate(lines):
            p.drawText(int(tx), int(ty + i * line_h), line)

        # Timestamp + confidence
        if self._timestamp:
            muted = parse_rgba(self._colors["muted"])
            p.setPen(muted)
            small = QFont(p.font())
            small.setPointSize(8)
            p.setFont(small)
            ts_text = self._timestamp
            if self._confidence > 0:
                ts_text += f" \u00b7 {int(self._confidence * 100)}%"
            ts_w = p.fontMetrics().horizontalAdvance(ts_text)
            p.drawText(int(bx + bw - ts_w - pad_h), int(by + bh - 6), ts_text)
        p.end()


# ── Empty state overlay ───────────────────────────────────────────

class EmptyState(QWidget):
    """Centered empty state with QUE logo and tagline."""

    def __init__(self, colors: dict, parent=None):
        super().__init__(parent)
        self._colors = colors
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Load the logo once
        import os
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "ui", "que_icon.png")
        self._logo = QPixmap(logo_path) if os.path.exists(logo_path) else None

    def set_colors(self, c: dict):
        self._colors = c
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        cx = self.width() // 2
        cy = self.height() // 2 - 50  # shift up slightly to balance with text below

        # Logo image (or fallback rounded-rect)
        logo_size = 120
        if self._logo and not self._logo.isNull():
            scaled = self._logo.scaled(logo_size, logo_size,
                                       Qt.KeepAspectRatio,
                                       Qt.SmoothTransformation)
            p.drawPixmap(cx - scaled.width() // 2,
                         cy - scaled.height() // 2, scaled)
        else:
            logo_r = 60
            logo_bg = parse_rgba(self._colors["card"])
            p.setBrush(logo_bg)
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(cx - logo_r, cy - logo_r, logo_r * 2, logo_r * 2, 28, 28)
            glow = parse_rgba(self._colors["glow"])
            p.setPen(glow)
            logo_font = QFont("Inter", 28, QFont.Bold)
            p.setFont(logo_font)
            p.drawText(QRectF(cx - logo_r, cy - logo_r, logo_r * 2, logo_r * 2),
                       Qt.AlignCenter, "QUE")

        # Title
        y = cy + logo_size // 2 + 24
        fg = parse_rgba(self._colors["foreground"])
        p.setPen(fg)
        title_font = QFont("Inter", 20, QFont.Bold)
        title_font.setLetterSpacing(QFont.AbsoluteSpacing, -0.3)
        p.setFont(title_font)
        p.drawText(QRectF(0, y, self.width(), 36), Qt.AlignCenter, "How can I help you?")

        # Subtitle
        y += 40
        muted = parse_rgba(self._colors["muted"])
        p.setPen(muted)
        sub_font = QFont("Inter", 11)
        p.setFont(sub_font)
        p.drawText(QRectF(0, y, self.width(), 24), Qt.AlignCenter,
                   "Local AI inference \u00b7 Private by default")
        p.end()
