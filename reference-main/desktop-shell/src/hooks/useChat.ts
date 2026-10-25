import { useState, useCallback, useRef, useEffect } from "react";
import { invoke, Channel } from "@tauri-apps/api/core";

const DEFAULT_MODEL = "deepseek-r1:7b";

const SYSTEM_PROMPT =
  "You are Cael, the local AI assistant inside QuantoniumOS. " +
  "You are helpful, concise, and friendly. You run fully on-device via Ollama — " +
  "no cloud, no telemetry. Answer directly and keep responses focused.";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
}

export interface EngineStatus {
  connected: boolean;
  model: string | null;
  backend: string | null;
}

interface OllamaStatus {
  running: boolean;
  model: string;
  models: string[];
  backend: string;
}

interface ChatToken {
  token: string;
  done: boolean;
}

/**
 * Chat hook — all inference goes through Rust middleware → Ollama.
 * No direct HTTP from frontend. Everything routes through the OS layer.
 */
export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [status, setStatus] = useState<EngineStatus>({
    connected: false,
    model: null,
    backend: null,
  });
  const abortRef = useRef(false);

  // Health check — poll Rust → Ollama status
  useEffect(() => {
    const check = async () => {
      try {
        const s = await invoke<OllamaStatus>("ollama_status");
        setStatus({
          connected: s.running,
          model: s.model || null,
          backend: s.backend || "ollama",
        });
      } catch {
        setStatus({ connected: false, model: null, backend: null });
      }
    };
    check();
    const iv = setInterval(check, 10000);
    return () => clearInterval(iv);
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsStreaming(true);
      abortRef.current = false;

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "",
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMsg]);

      try {
        // Build history with system prompt
        const history = [
          { role: "system", content: SYSTEM_PROMPT },
          ...[...messages, userMsg].slice(-20).map((m) => ({
            role: m.role,
            content: m.content,
          })),
        ];

        // Stream tokens via Tauri Channel (Rust middleware → Ollama → back)
        const channel = new Channel<ChatToken>();
        channel.onmessage = (event: ChatToken) => {
          if (abortRef.current) return;
          if (event.token) {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsg.id
                  ? { ...m, content: m.content + event.token }
                  : m
              )
            );
          }
        };

        await invoke("chat_stream", {
          req: {
            messages: history,
            model: status.model || DEFAULT_MODEL,
          },
          onEvent: channel,
        });
      } catch (err: unknown) {
        if (abortRef.current) return;
        const errMsg = err instanceof Error ? err.message : String(err);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id
              ? { ...m, content: m.content || `Connection error: ${errMsg}` }
              : m
          )
        );
      } finally {
        setIsStreaming(false);
      }
    },
    [messages, status.model]
  );

  const stopStreaming = useCallback(() => {
    abortRef.current = true;
  }, []);

  const clearHistory = useCallback(() => {
    setMessages([]);
  }, []);

  return { messages, isStreaming, status, sendMessage, stopStreaming, clearHistory };
}
