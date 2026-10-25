# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Chatbox color palettes and parsing — mirrors mobile useColors.ts.
"""

import re
from PyQt5.QtGui import QColor

# ── Color Palette (matches mobile useColors.ts) ───────────────────

DARK = {
    "background":       "#121212",
    "foreground":       "#e0e0e0",
    "primary":          "#b0b0b0",
    "primaryFg":        "#121212",
    "muted":            "#606060",
    "accent":           "#909090",
    "glow":             "#ffffff",
    "glowMuted":        "#1a1a1a",
    "destructive":      "#c04040",
    "safetyGreen":      "#60a060",
    "safetyRed":        "#c04040",
    "border":           "#252525",
    "card":             "#1a1a1a",
    "glassBg":          "#161616",
    "glassBorder":      "#2e2e2e",
    "glassShadow":      "#050505",
    "userBubble":       "rgba(38,38,38,0.70)",
    "userBubbleBorder": "rgba(255,255,255,0.10)",
    "userBubbleText":   "#f0f0f0",
    "aiBubble":         "rgba(26,26,26,0.70)",
    "aiBubbleBorder":   "rgba(120,120,120,0.12)",
    "aiBubbleText":     "#e0e0e0",
    "gradStart":        "#121212",
    "gradMid":          "#151515",
    "gradEnd":          "#1a1a1a",
    "inputBg":          "#161616",
    "inputBorderFocus": "#808080",
    "sendActive":       "#ffffff",
    "sendDisabled":     "#2e2e2e",
    "headerBg":         "#0f0f0f",
    "headerBorder":     "#252525",
}

LIGHT = {
    "background":       "#f0f0f0",
    "foreground":       "#1a1a1a",
    "primary":          "#404040",
    "primaryFg":        "#ffffff",
    "muted":            "#888888",
    "accent":           "#606060",
    "glow":             "#000000",
    "glowMuted":        "#e8e8e8",
    "destructive":      "#c04040",
    "safetyGreen":      "#60a060",
    "safetyRed":        "#c04040",
    "border":           "#d0d0d0",
    "card":             "#f8f8f8",
    "glassBg":          "#f4f4f4",
    "glassBorder":      "#e0e0e0",
    "glassShadow":      "#c8c8c8",
    "userBubble":       "rgba(255,255,255,0.60)",
    "userBubbleBorder": "rgba(0,0,0,0.10)",
    "userBubbleText":   "#1a1a1a",
    "aiBubble":         "rgba(255,255,255,0.50)",
    "aiBubbleBorder":   "rgba(0,0,0,0.08)",
    "aiBubbleText":     "#1a1a1a",
    "gradStart":        "#e0e0e0",
    "gradMid":          "#e8e8e8",
    "gradEnd":          "#f4f4f4",
    "inputBg":          "#f4f4f4",
    "inputBorderFocus": "#404040",
    "sendActive":       "#1a1a1a",
    "sendDisabled":     "#c8c8c8",
    "headerBg":         "#e8e8e8",
    "headerBorder":     "#d0d0d0",
}

# ── Safe mode patterns (matches mobile ChatContext) ────────────────

BLOCKED_PATTERNS = ["rm -rf", "format c:", "shutdown /s", "powershell -c"]


def parse_rgba(s: str) -> QColor:
    """Parse 'rgba(r,g,b,a)' or '#rrggbb' to QColor."""
    s = s.strip()
    if s.startswith("#"):
        return QColor(s)
    m = re.match(r"rgba?\((\d+),\s*(\d+),\s*(\d+),?\s*([\d.]+)?\)", s)
    if m:
        r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        a = float(m.group(4)) if m.group(4) else 1.0
        c = QColor(r, g, b)
        c.setAlphaF(a)
        return c
    return QColor(s)
