#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
# Copyright (C) 2025 Luis M. Minier / quantoniumos
# See LICENSE-SPX-PROPRIETARY.md for terms.
"""
System Monitor — quantoniumos Performance Dashboard (Layer 5, Claim 4)
======================================================================

PyQt5 application providing real-time CPU/memory/disk monitoring alongside
a Vertex Resonance Graph that visualises φ-modulated signal amplitudes as a
live animation.

Layer:   5 — Application Suite (quantonium_os_src/apps/)
License: LicenseRef-SPX-Proprietary (research/education only)
Patent:  USPTO 19/169,399, Claim 4 — Hybrid Mode Integration

Components
----------
VertexGraphWidget
    Fallback renderer when matplotlib is absent.  Draws the resonance
    amplitude curve using Qt's QPainter directly.

MainWindow
    Main dashboard window.  Polls system metrics via psutil every 2 s and
    updates CPU/memory/disk progress bars plus a per-process table filtered
    to python/quantonium processes.  The Vertex Resonance panel shows a
    time-varying golden-ratio (φ) modulated sinusoid as a streaming signal
    proxy, implemented with matplotlib when available or VertexGraphWidget
    otherwise.
"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QProgressBar, QGroupBox, QTableWidget, QTableWidgetItem)
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QFont, QPainter, QColor, QPen, QBrush, QPainterPath
import psutil
import time
import math

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Qt5Agg")
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False


class VertexGraphWidget(QWidget):
    """Simple fallback vertex resonance renderer when matplotlib is absent."""

    def __init__(self, parent=None, vertex_count=32):
        super().__init__(parent)
        self.vertex_count = vertex_count
        self.amplitudes = [0.0] * vertex_count
        self.setMinimumHeight(220)

    def set_amplitudes(self, amplitudes):
        if not amplitudes:
            return
        self.vertex_count = len(amplitudes)
        self.amplitudes = list(amplitudes)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QColor("#0f1216"))

        if not self.amplitudes:
            painter.end()
            return

        max_amp = max(self.amplitudes) or 1.0
        width = rect.width()
        height = rect.height()
        spacing = width / max(1, len(self.amplitudes) - 1)
        points = []

        for idx, amp in enumerate(self.amplitudes):
            x = idx * spacing
            y = height - ((amp / max_amp) * (height * 0.7) + height * 0.1)
            points.append(QPointF(x, y))

        if not points:
            painter.end()
            return

        path = QPainterPath()
        path.moveTo(points[0])
        for point in points[1:]:
            path.lineTo(point)

        fill_path = QPainterPath(path)
        fill_path.lineTo(points[-1].x(), height)
        fill_path.lineTo(points[0].x(), height)
        fill_path.closeSubpath()

        painter.setPen(QPen(QColor("#00aaff"), 2))
        painter.drawPath(path)
        painter.fillPath(fill_path, QBrush(QColor("#00aaff22")))
        painter.end()


class MainWindow(QMainWindow):
    """Main system-monitor window.

    Lays out CPU, memory, and disk progress bars, an RFT-process table, and
    the Vertex Resonance Graph panel.  A QTimer fires every 2 000 ms to
    refresh all metrics via :meth:`update_stats`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Monitor - quantoniumos")
        self.setGeometry(100, 100, 900, 700)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        title = QLabel("📊 System Performance Monitor")
        title.setFont(QFont("Sans Serif", 16, QFont.Bold))
        title.setStyleSheet("color: #00aaff; padding: 10px;")
        layout.addWidget(title)
        
        # CPU
        cpu_group = QGroupBox("CPU Usage")
        cpu_layout = QVBoxLayout()
        self.cpu_bar = QProgressBar()
        self.cpu_label = QLabel("0%")
        cpu_layout.addWidget(self.cpu_bar)
        cpu_layout.addWidget(self.cpu_label)
        cpu_group.setLayout(cpu_layout)
        layout.addWidget(cpu_group)
        
        # Memory
        mem_group = QGroupBox("Memory Usage")
        mem_layout = QVBoxLayout()
        self.mem_bar = QProgressBar()
        self.mem_label = QLabel("0 MB / 0 MB")
        mem_layout.addWidget(self.mem_bar)
        mem_layout.addWidget(self.mem_label)
        mem_group.setLayout(mem_layout)
        layout.addWidget(mem_group)
        
        # Disk
        disk_group = QGroupBox("Disk Usage")
        disk_layout = QVBoxLayout()
        self.disk_bar = QProgressBar()
        self.disk_label = QLabel("0 GB / 0 GB")
        disk_layout.addWidget(self.disk_bar)
        disk_layout.addWidget(self.disk_label)
        disk_group.setLayout(disk_layout)
        layout.addWidget(disk_group)
        
        # Process table
        proc_group = QGroupBox("RFT Processes")
        proc_layout = QVBoxLayout()
        self.proc_table = QTableWidget()
        self.proc_table.setColumnCount(3)
        self.proc_table.setHorizontalHeaderLabels(["Process", "CPU %", "Memory (MB)"])
        proc_layout.addWidget(self.proc_table)
        proc_group.setLayout(proc_layout)
        layout.addWidget(proc_group)
        
        self.vertex_fig = None
        self.vertex_canvas = None
        self.vertex_ax = None
        self.vertex_graph_widget = None

        if HAS_MATPLOTLIB:
            self.vertex_fig = Figure(figsize=(5, 2.5), dpi=100)
            self.vertex_canvas = FigureCanvas(self.vertex_fig)
            self.vertex_ax = self.vertex_fig.add_subplot(1, 1, 1)
            vertex_group = QGroupBox("Vertex Resonance Graph")
            vertex_layout = QVBoxLayout()
            vertex_layout.addWidget(self.vertex_canvas)
            vertex_group.setLayout(vertex_layout)
            layout.addWidget(vertex_group)
        else:
            self.vertex_graph_widget = VertexGraphWidget()
            vertex_group = QGroupBox("Vertex Resonance Graph")
            vertex_layout = QVBoxLayout()
            vertex_layout.addWidget(self.vertex_graph_widget)
            vertex_group.setLayout(vertex_layout)
            layout.addWidget(vertex_group)
        
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.update_stats)
        layout.addWidget(self.refresh_btn)
        
        self.statusBar().showMessage("Monitoring system performance")
        self.set_dark_theme()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(2000)
        self.update_stats()
    
    def set_dark_theme(self):
        """Apply the quantoniumos dark stylesheet to all child widgets."""
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #1a1a1a; color: #ffffff; }
            QGroupBox { border: 2px solid #00aaff; border-radius: 5px; margin-top: 10px; padding-top: 10px; color: #00aaff; font-weight: bold; }
            QPushButton { background-color: #00aaff; color: #fff; border: none; padding: 8px 16px; border-radius: 4px; }
            QPushButton:hover { background-color: #00ffaa; }
            QProgressBar { background-color: #2a2a2a; border: 1px solid #00aaff; text-align: center; }
            QProgressBar::chunk { background-color: #00aaff; }
            QTableWidget { background-color: #2a2a2a; color: #fff; border: 1px solid #00aaff; }
            QHeaderView::section { background-color: #00aaff; color: #fff; padding: 5px; border: none; }
        """)
    
    def update_stats(self):
        """Refresh all system metrics and update the UI widgets.

        Reads CPU percent, virtual memory, and root-disk usage via psutil,
        updates the three progress bars and labels, then populates the process
        table with up to five python/quantonium processes sorted by CPU usage.
        Always calls :meth:`_update_vertex_graph` when matplotlib is available.
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_bar.setValue(int(cpu_percent))
            self.cpu_label.setText(f"{cpu_percent:.1f}% - {psutil.cpu_count()} cores")
            
            mem = psutil.virtual_memory()
            self.mem_bar.setValue(int(mem.percent))
            self.mem_label.setText(f"{mem.used / 1024**3:.1f} GB / {mem.total / 1024**3:.1f} GB ({mem.percent:.1f}%)")
            
            disk = psutil.disk_usage('/')
            self.disk_bar.setValue(int(disk.percent))
            self.disk_label.setText(f"{disk.used / 1024**3:.1f} GB / {disk.total / 1024**3:.1f} GB ({disk.percent:.1f}%)")
            
            processes = []
            for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_info']):
                try:
                    info = proc.info
                    if 'python' in info['name'].lower() or 'quantonium' in info['name'].lower():
                        processes.append((info['name'], info['cpu_percent'], info['memory_info'].rss / 1024**2))
                except:
                    pass
            
            processes.sort(key=lambda x: x[1], reverse=True)
            self.proc_table.setRowCount(min(len(processes), 5))
            
            for i, (name, cpu, mem) in enumerate(processes[:5]):
                self.proc_table.setItem(i, 0, QTableWidgetItem(name))
                self.proc_table.setItem(i, 1, QTableWidgetItem(f"{cpu:.1f}"))
                self.proc_table.setItem(i, 2, QTableWidgetItem(f"{mem:.1f}"))
        
        except Exception as e:
            self.statusBar().showMessage(f"Error updating stats: {e}")
        finally:
            if HAS_MATPLOTLIB:
                self._update_vertex_graph()

    def _update_vertex_graph(self):
        """Redraw the Vertex Resonance Graph for the current time step.

        Generates a 32-vertex amplitude array using a φ-modulated sinusoid
        ``|sin(ω·(1+0.2·sin(t))) · cos(ω·φ + t)|`` and plots it with a cyan
        fill.  Called from :meth:`update_stats` each refresh cycle.
        """
        try:
            vertex_count = 32
            t = time.time()
            phi = 1.618033988749895
            frequencies = np.linspace(0, 2 * np.pi, vertex_count, endpoint=False)
            amplitudes = np.abs(np.sin(frequencies * (1 + 0.2 * np.sin(t))) * np.cos(frequencies * phi + t))

            self.vertex_ax.clear()
            self.vertex_ax.plot(amplitudes, color="#00aaff", linewidth=2)
            self.vertex_ax.fill_between(range(vertex_count), 0, amplitudes, color="#00aaff", alpha=0.35)
            self.vertex_ax.set_title("Vertex Resonance (RFT streaming)")
            self.vertex_ax.set_ylabel("Amplitude")
            self.vertex_ax.set_xlabel("Vertex Index")
            self.vertex_ax.set_xlim(0, vertex_count - 1)
            self.vertex_ax.set_ylim(0, amplitudes.max() * 1.1 if amplitudes.max() > 0 else 1.0)
            self.vertex_ax.set_xticks([])
            self.vertex_ax.set_facecolor("#0f1216")
            self.vertex_fig.tight_layout()
            self.vertex_canvas.draw_idle()
        except Exception as exc:
            self.statusBar().showMessage(f"Vertex graph error: {exc}")
