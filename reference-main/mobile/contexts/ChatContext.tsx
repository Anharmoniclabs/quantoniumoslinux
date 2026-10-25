import AsyncStorage from "@react-native-async-storage/async-storage";
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { AppState, Platform } from "react-native";

export type Role = "user" | "assistant" | "system";

export interface Message {
  id: string;
  role: Role;
  content: string;
  timestamp: number;
  confidence?: number;
  backend?: string;
}

export interface ModelInfo {
  id: string;
  label: string;
  backend: "ollama" | "rftmw" | "demo";
}

interface ChatContextValue {
  messages: Message[];
  isTyping: boolean;
  ollamaUrl: string;
  selectedModel: string;
  availableModels: ModelInfo[];
  safeMode: boolean;
  safetyOk: boolean;
  sendMessage: (text: string) => Promise<void>;
  clearHistory: () => void;
  setOllamaUrl: (url: string) => void;
  setSelectedModel: (id: string) => void;
  setSafeMode: (v: boolean) => void;
  refreshModels: (urlOverride?: string) => Promise<void>;
}

const ChatContext = createContext<ChatContextValue | null>(null);

const STORAGE_KEYS = {
  MESSAGES: "quai_messages",
  OLLAMA_URL: "quai_ollama_url",
  SELECTED_MODEL: "quai_selected_model",
  SAFE_MODE: "quai_safe_mode",
};

function getDefaultUrl(): string {
  try {
    if (
      Platform.OS === "web" &&
      typeof window !== "undefined" &&
      window.location &&
      window.location.hostname
    ) {
      const h = window.location.hostname;
      // If not localhost, use the page hostname (mobile browser on LAN)
      if (h !== "localhost" && h !== "127.0.0.1") {
        return `http://${h}:8000`;
      }
    }
    // Android emulator: localhost maps to the emulator itself, not the host
    if (Platform.OS === "android") {
      return "http://10.0.2.2:8000";
    }
  } catch {}
  return "http://localhost:8000";
}

const DEFAULT_MODEL = "qwen2.5-coder:3b";

const BLOCKED_PATTERNS = [
  "rm -rf",
  "format c:",
  "shutdown /s",
  "powershell -c",
];

function makeId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function isSafePrompt(text: string): boolean {
  const lower = text.toLowerCase();
  return !BLOCKED_PATTERNS.some((p) => lower.includes(p));
}

function isServerConnected(models: ModelInfo[]): boolean {
  return models.some((m) => m.backend !== "demo");
}

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [ollamaUrl, setOllamaUrlState] = useState(() => getDefaultUrl());
  const [selectedModel, setSelectedModelState] = useState(DEFAULT_MODEL);
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [safeMode, setSafeModeState] = useState(false);
  const [safetyOk, setSafetyOk] = useState(true);
  const abortRef = useRef<AbortController | null>(null);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ollamaUrlRef = useRef(ollamaUrl);

  // Keep ref in sync so refreshModels always reads the latest URL
  useEffect(() => {
    ollamaUrlRef.current = ollamaUrl;
  }, [ollamaUrl]);

  const refreshModels = useCallback(async (urlOverride?: string) => {
    const baseUrl = urlOverride ?? ollamaUrlRef.current;
    try {
      const res = await fetch(`${baseUrl}/api/tags`, {
        signal: AbortSignal.timeout(6000),
      });
      if (!res.ok) throw new Error("not ok");
      const data = await res.json();
      const models: ModelInfo[] = (data.models ?? []).map((m: { name: string; details?: { format?: string } }) => ({
        id: m.name,
        label: m.name,
        backend: (m.details?.format === "rftmwpk" ? "rftmw" : "ollama") as "ollama" | "rftmw",
      }));
      if (models.length === 0) {
        setAvailableModels([{ id: "demo", label: "Demo Mode (no model server)", backend: "demo" }]);
      } else {
        setAvailableModels(models);
        setSafetyOk(true);
        setSelectedModelState((prev) => {
          if (prev === "demo" || !models.find((m) => m.id === prev)) {
            const first = models[0].id;
            AsyncStorage.setItem(STORAGE_KEYS.SELECTED_MODEL, first).catch(() => {});
            return first;
          }
          return prev;
        });
      }
    } catch {
      setAvailableModels([{ id: "demo", label: "Demo Mode (no model server)", backend: "demo" }]);
      setSafetyOk(false);
    }
  }, []);

  // Auto-reconnect when app regains focus or wakes from sleep (PWA/web only)
  useEffect(() => {
    if (Platform.OS !== "web") return;
    function handleFocus() {
      refreshModels();
    }
    window.addEventListener("focus", handleFocus);
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") handleFocus();
    });
    return () => {
      window.removeEventListener("focus", handleFocus);
      document.removeEventListener("visibilitychange", handleFocus);
    };
  }, [refreshModels]);

  // Auto-reconnect when native app returns to foreground
  useEffect(() => {
    if (Platform.OS === "web") return;
    const sub = AppState.addEventListener("change", (state) => {
      if (state === "active") refreshModels();
    });
    return () => sub.remove();
  }, [refreshModels]);

  useEffect(() => {
    (async () => {
      try {
        const [msgs, url, model, safe] = await Promise.all([
          AsyncStorage.getItem(STORAGE_KEYS.MESSAGES),
          AsyncStorage.getItem(STORAGE_KEYS.OLLAMA_URL),
          AsyncStorage.getItem(STORAGE_KEYS.SELECTED_MODEL),
          AsyncStorage.getItem(STORAGE_KEYS.SAFE_MODE),
        ]);
        if (msgs) setMessages(JSON.parse(msgs));
        // Only restore a saved URL if it points to a real server.
        // If we auto-detected a LAN IP, prefer it over a stale "localhost" save.
        const defaultUrl = getDefaultUrl();
        const isLan = !defaultUrl.includes("localhost") && !defaultUrl.includes("127.0.0.1");
        const savedIsLocal = url && (url.includes("localhost") || url.includes("127.0.0.1"));
        if (url && url.includes(":8000") && !(isLan && savedIsLocal)) {
          setOllamaUrlState(url);
          ollamaUrlRef.current = url;
        } else {
          // Clear any stale saved URL and use the computed default
          setOllamaUrlState(defaultUrl);
          ollamaUrlRef.current = defaultUrl;
          await AsyncStorage.setItem(STORAGE_KEYS.OLLAMA_URL, defaultUrl);
        }
        if (model && model !== "demo") setSelectedModelState(model);
        if (safe !== null) setSafeModeState(safe === "true");
      } catch {}
    })();
  }, []);

  const persist = useCallback(async (msgs: Message[]) => {
    try {
      await AsyncStorage.setItem(STORAGE_KEYS.MESSAGES, JSON.stringify(msgs.slice(-200)));
    } catch {}
  }, []);

  const setOllamaUrl = useCallback(async (url: string) => {
    setOllamaUrlState(url);
    ollamaUrlRef.current = url;
    await AsyncStorage.setItem(STORAGE_KEYS.OLLAMA_URL, url);
  }, []);

  const setSelectedModel = useCallback(async (id: string) => {
    setSelectedModelState(id);
    await AsyncStorage.setItem(STORAGE_KEYS.SELECTED_MODEL, id);
  }, []);

  const setSafeMode = useCallback(async (v: boolean) => {
    setSafeModeState(v);
    await AsyncStorage.setItem(STORAGE_KEYS.SAFE_MODE, String(v));
  }, []);

  // Initial fetch on mount
  useEffect(() => {
    refreshModels();
  }, [refreshModels]);

  // Retry timer — uses refs to avoid stale closures
  useEffect(() => {
    if (isServerConnected(availableModels)) {
      if (retryTimerRef.current) {
        clearTimeout(retryTimerRef.current);
        retryTimerRef.current = null;
      }
      return;
    }
    const id = setTimeout(async () => {
      await refreshModels();
    }, 3000);
    retryTimerRef.current = id;
    return () => clearTimeout(id);
  }, [availableModels, refreshModels]);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim()) return;

      if (safeMode && !isSafePrompt(text)) {
        const blocked: Message = {
          id: makeId(),
          role: "assistant",
          content: "Safe Mode is active. This message was blocked because it contains a restricted pattern.",
          timestamp: Date.now(),
          confidence: 1.0,
        };
        setMessages((prev) => {
          const next = [...prev, blocked];
          persist(next);
          return next;
        });
        return;
      }

      const userMsg: Message = {
        id: makeId(),
        role: "user",
        content: text.trim(),
        timestamp: Date.now(),
      };

      setMessages((prev) => {
        const next = [...prev, userMsg];
        persist(next);
        return next;
      });
      setIsTyping(true);

      let isDemo =
        selectedModel === "demo" ||
        availableModels.find((m) => m.id === selectedModel)?.backend === "demo";

      // If stuck in demo mode, try reconnecting before giving up
      if (isDemo) {
        await refreshModels();
        isDemo =
          selectedModel === "demo" ||
          availableModels.find((m) => m.id === selectedModel)?.backend === "demo";
      }

      try {
        if (isDemo) {
          await new Promise((r) => setTimeout(r, 800));
          const demoReply: Message = {
            id: makeId(),
            role: "assistant",
            content: `QU.Ai Demo Mode — connect to a model server to enable real AI responses. Your message was: "${text.trim()}"`,
            timestamp: Date.now(),
            confidence: 0.0,
            backend: "demo",
          };
          setMessages((prev) => {
            const next = [...prev, demoReply];
            persist(next);
            return next;
          });
          return;
        }

        abortRef.current?.abort();
        const ac = new AbortController();
        abortRef.current = ac;

        const history = messages.slice(-20).map((m) => ({
          role: m.role === "assistant" ? "assistant" : "user",
          content: m.content,
        }));

        const body = JSON.stringify({
          model: selectedModel,
          messages: [
            {
              role: "system",
              content: [
                "You are QU.Ai, the assistant shipped with QuantoniumOS — a research project on compressed inference and resonant Fourier weight packing.",
                "",
                "Principles:",
                "- Be direct. No filler, no hedging, no apologies. If you're confident, say it plainly.",
                "- Prefer concrete answers (code, commands, numbers) over vague advice.",
                "- Think step by step on hard problems, but don't show the scratch work unless asked — give the user the answer, not the process.",
                "- When the user asks 'why', explain the mechanism; when they ask 'how', give the steps.",
                "- If you don't know, say 'I don't know' instead of guessing. Guessing wastes the user's time.",
                "- Code answers use proper language idioms and compile/run as written. No pseudocode unless explicitly requested.",
                "- When retrieved context is provided (marked 'RELEVANT CONTEXT'), treat it as authoritative for this conversation and cite which file the fact came from.",
                "- When tool results are provided (marked 'TOOL RESULT'), use them as ground truth. Don't re-derive what a tool already answered.",
                "",
                "Style: short paragraphs, no bullet-point spam, no emoji, no markdown headers unless the answer is long enough to need them.",
              ].join("\n"),
            },
            ...history,
            { role: "user", content: text.trim() },
          ],
          stream: true,
          options: { temperature: 0.7, max_new_tokens: 256 },
        });

        const res = await fetch(`${ollamaUrl}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
          signal: ac.signal,
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);
        if (!res.body) throw new Error("No response body");

        const aiMsgId = makeId();
        const aiMsg: Message = {
          id: aiMsgId,
          role: "assistant",
          content: "",
          timestamp: Date.now(),
          confidence: Math.random() * 0.15 + 0.82,
          backend: "ollama",
        };

        setMessages((prev) => {
          const next = [...prev, aiMsg];
          persist(next);
          return next;
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let fullContent = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() ?? "";

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            try {
              const chunk = JSON.parse(trimmed);
              const token: string = chunk?.message?.content ?? "";
              if (token && !chunk.done) {
                fullContent += token;
                const captured = fullContent;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === aiMsgId ? { ...m, content: captured } : m
                  )
                );
              }
            } catch {}
          }
        }

        if (fullContent) {
          const final = fullContent.trim().slice(0, 5000);
          setMessages((prev) => {
            const updated = prev.map((m) =>
              m.id === aiMsgId ? { ...m, content: final } : m
            );
            persist(updated);
            return updated;
          });
        }

        setSafetyOk(true);
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") return;
        const errMsg: Message = {
          id: makeId(),
          role: "assistant",
          content: "Could not reach the model server. Check the URL in settings.",
          timestamp: Date.now(),
          confidence: 0,
        };
        setMessages((prev) => {
          const next = [...prev, errMsg];
          persist(next);
          return next;
        });
        setSafetyOk(false);
      } finally {
        setIsTyping(false);
      }
    },
    [messages, ollamaUrl, selectedModel, safeMode, availableModels, persist]
  );

  const clearHistory = useCallback(() => {
    setMessages([]);
    AsyncStorage.removeItem(STORAGE_KEYS.MESSAGES).catch(() => {});
  }, []);

  return (
    <ChatContext.Provider
      value={{
        messages,
        isTyping,
        ollamaUrl,
        selectedModel,
        availableModels,
        safeMode,
        safetyOk,
        sendMessage,
        clearHistory,
        setOllamaUrl,
        setSelectedModel,
        setSafeMode,
        refreshModels,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChat must be used within ChatProvider");
  return ctx;
}
