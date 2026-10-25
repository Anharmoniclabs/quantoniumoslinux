# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Voice chat helpers for the QSHLL chatbox.

VoiceRecorder  — push-to-talk mic → faster-whisper STT (falls back to openai-whisper)
VoiceSpeaker   — pyttsx3 TTS in a background thread with voice/rate/volume control
"""

import logging
import threading
import tempfile
import os

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

# ── Availability flags ────────────────────────────────────────────

_HAS_WHISPER = False
_HAS_SOUNDDEVICE = False
_HAS_PYTTSX3 = False
_WHISPER_BACKEND = None  # "faster" or "openai"

try:
    from faster_whisper import WhisperModel as _FasterWhisper
    _HAS_WHISPER = True
    _WHISPER_BACKEND = "faster"
except Exception as _exc:
    _FasterWhisper = None
    logger.info("faster-whisper unavailable: %s", _exc)
    try:
        import whisper as _whisper
        _HAS_WHISPER = True
        _WHISPER_BACKEND = "openai"
    except Exception as _exc2:
        _whisper = None
        logger.info("openai-whisper unavailable: %s", _exc2)

try:
    import sounddevice as _sd
    _HAS_SOUNDDEVICE = True
except Exception as _exc:
    _sd = None
    logger.info("sounddevice unavailable: %s", _exc)

try:
    import pyttsx3 as _pyttsx3
    _HAS_PYTTSX3 = True
except Exception as _exc:
    _pyttsx3 = None
    logger.info("pyttsx3 unavailable: %s", _exc)


def voice_available() -> dict:
    """Return availability of voice subsystems."""
    return {"stt": _HAS_WHISPER and _HAS_SOUNDDEVICE, "tts": _HAS_PYTTSX3}


def list_voices() -> list:
    """Enumerate installed TTS voices. Returns [{id, name, gender, languages}, ...]."""
    if not _HAS_PYTTSX3:
        return []
    try:
        engine = _pyttsx3.init()
        out = []
        for v in engine.getProperty("voices"):
            langs = getattr(v, "languages", []) or []
            langs = [str(l) for l in langs]
            out.append({
                "id": v.id,
                "name": v.name or "",
                "gender": (getattr(v, "gender", None) or "").lower() or None,
                "age": getattr(v, "age", None),
                "languages": langs,
            })
        try:
            engine.stop()
        except Exception:
            pass
        return out
    except Exception as exc:
        logger.warning("list_voices failed: %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════
#  VOICE RECORDER  (mic → Whisper STT)
# ══════════════════════════════════════════════════════════════════

class VoiceRecorder(QObject):
    """Push-to-talk recorder that transcribes via local Whisper."""

    transcribed = pyqtSignal(str)   # emitted with recognised text
    error = pyqtSignal(str)
    recording_changed = pyqtSignal(bool)  # True = started, False = stopped

    SAMPLE_RATE = 16_000  # Whisper expects 16 kHz mono

    def __init__(self, model_name: str = "base", parent=None):
        super().__init__(parent)
        self._model_name = model_name
        self._model = None  # lazy-loaded
        self._chunks: list = []
        self._stream = None
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self):
        """Begin recording from the default microphone."""
        if not _HAS_SOUNDDEVICE:
            self.error.emit("sounddevice not installed — run: pip install sounddevice")
            return
        if not _HAS_WHISPER:
            self.error.emit("No STT backend — run: pip install faster-whisper")
            return
        if self._recording:
            return

        self._chunks = []
        self._recording = True
        self.recording_changed.emit(True)

        def _audio_cb(indata, frames, time_info, status):
            if status:
                logger.debug("sounddevice status: %s", status)
            self._chunks.append(indata.copy())

        try:
            self._stream = _sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=_audio_cb,
            )
            self._stream.start()
        except Exception as exc:
            self._recording = False
            self.recording_changed.emit(False)
            self.error.emit(f"Mic error: {exc}")

    def stop(self):
        """Stop recording and transcribe in a background thread."""
        if not self._recording:
            return
        self._recording = False
        self.recording_changed.emit(False)

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if not self._chunks:
            return

        audio = np.concatenate(self._chunks, axis=0).flatten()
        self._chunks = []

        # Run Whisper in a thread to avoid blocking the UI
        threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    def _transcribe(self, audio: np.ndarray):
        try:
            if _WHISPER_BACKEND == "faster":
                if self._model is None:
                    logger.info("Loading faster-whisper '%s' (int8)…", self._model_name)
                    self._model = _FasterWhisper(
                        self._model_name, device="cpu", compute_type="int8"
                    )
                segments, _info = self._model.transcribe(
                    audio.astype(np.float32),
                    language="en",
                    beam_size=1,
                    vad_filter=True,
                )
                text = " ".join(seg.text for seg in segments).strip()
            else:
                if self._model is None:
                    logger.info("Loading openai-whisper '%s' model…", self._model_name)
                    self._model = _whisper.load_model(self._model_name)
                result = self._model.transcribe(audio, language="en", fp16=False)
                text = result.get("text", "").strip()
            if text:
                self.transcribed.emit(text)
        except Exception as exc:
            self.error.emit(f"Transcription failed: {exc}")


# ══════════════════════════════════════════════════════════════════
#  VOICE SPEAKER  (pyttsx3 TTS)
# ══════════════════════════════════════════════════════════════════

class VoiceSpeaker(QObject):
    """Background TTS via pyttsx3 (SAPI5 on Windows, espeak on Linux)."""

    speaking = pyqtSignal(bool)  # True = started, False = done
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, voice_id: str = None, rate: int = 180, volume: float = 1.0, parent=None):
        super().__init__(parent)
        self._busy = False
        self._voice_id = voice_id
        self._rate = int(rate)
        self._volume = float(volume)

    def set_voice(self, voice_id: str):
        self._voice_id = voice_id

    def set_rate(self, rate: int):
        self._rate = int(rate)

    def set_volume(self, volume: float):
        self._volume = max(0.0, min(1.0, float(volume)))

    def speak(self, text: str):
        """Speak *text* in a background thread. Ignores if already speaking."""
        if not _HAS_PYTTSX3:
            self.error.emit("pyttsx3 not installed — run: pip install pyttsx3")
            return
        if self._busy or not text.strip():
            return
        self._busy = True
        self.speaking.emit(True)
        threading.Thread(
            target=self._run_tts,
            args=(text, self._voice_id, self._rate, self._volume),
            daemon=True,
        ).start()

    def _run_tts(self, text: str, voice_id: str, rate: int, volume: float):
        try:
            engine = _pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)
            if voice_id:
                try:
                    engine.setProperty("voice", voice_id)
                except Exception as exc:
                    logger.warning("Could not set voice %s: %s", voice_id, exc)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            logger.warning("TTS error: %s", exc)
            self.error.emit(f"TTS error: {exc}")
        finally:
            self._busy = False
            self.speaking.emit(False)
            self.finished.emit()
