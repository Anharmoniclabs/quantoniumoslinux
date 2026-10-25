import { useState, useRef, useCallback, useEffect } from "react";

interface UseVoiceReturn {
  isListening: boolean;
  isSpeaking: boolean;
  autoSpeak: boolean;
  transcript: string;
  error: string;
  startListening: () => void;
  stopListening: () => void;
  toggleAutoSpeak: () => void;
  speak: (text: string) => void;
  stopSpeaking: () => void;
  supported: boolean;
}

const SpeechRecognitionCtor =
  (window as any).SpeechRecognition ||
  (window as any).webkitSpeechRecognition;

export function useVoice(): UseVoiceReturn {
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState("");
  const recognitionRef = useRef<any>(null);
  const supported = !!SpeechRecognitionCtor;

  useEffect(() => {
    if (!supported) return;
    const recognition = new SpeechRecognitionCtor();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event: any) => {
      let final = "";
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          final += t;
        } else {
          interim += t;
        }
      }
      setTranscript(final || interim);
    };

    recognition.onend = () => setIsListening(false);

    recognition.onerror = (event: any) => {
      console.error("[Voice] SpeechRecognition error:", event.error, event.message);
      setIsListening(false);
      const code = event.error as string;
      if (code === "not-allowed") {
        setError("Microphone access denied — check system permissions");
      } else if (code === "network") {
        setError("Speech service unavailable — network blocked by CSP");
      } else if (code === "no-speech") {
        setError("No speech detected — try again");
      } else if (code === "service-not-allowed") {
        setError("Speech recognition not available in this browser");
      } else {
        setError(`Voice error: ${code || "unknown"}`);
      }
    };

    recognitionRef.current = recognition;
    return () => {
      recognition.abort();
    };
  }, [supported]);

  const startListening = useCallback(async () => {
    if (!recognitionRef.current || isListening) return;
    setTranscript("");
    setError("");

    // Explicitly request microphone permission via getUserMedia first.
    // WebView2 blocks SpeechRecognition silently without this.
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      // Stop the tracks immediately — we just needed the permission grant
      stream.getTracks().forEach((t) => t.stop());
    } catch (e: any) {
      console.error("[Voice] getUserMedia denied:", e);
      setError("Microphone access denied — check system permissions");
      return;
    }

    try {
      recognitionRef.current.start();
      setIsListening(true);
    } catch (e: any) {
      console.error("[Voice] start() threw:", e);
      setError(e.message || "Failed to start voice input");
      setIsListening(false);
    }
  }, [isListening]);

  const stopListening = useCallback(() => {
    if (!recognitionRef.current) return;
    recognitionRef.current.stop();
  }, []);

  const speak = useCallback((text: string) => {
    const clean = text
      .replace(/```[\s\S]*?```/g, " code block ")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/^\s*\[IMAGEGEN\][^\r\n]*$/gim, "")
      .replace(/[*_~#>]/g, "")
      .replace(/<think>[\s\S]*?<\/think>/gi, "")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
    if (!clean) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(clean);
    utterance.rate = 1.05;
    utterance.pitch = 0.95;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  }, []);

  const stopSpeaking = useCallback(() => {
    window.speechSynthesis.cancel();
    setIsSpeaking(false);
  }, []);

  const toggleAutoSpeak = useCallback(() => {
    setAutoSpeak((prev) => !prev);
  }, []);

  return {
    isListening,
    isSpeaking,
    autoSpeak,
    transcript,
    error,
    startListening,
    stopListening,
    toggleAutoSpeak,
    speak,
    stopSpeaking,
    supported,
  };
}
