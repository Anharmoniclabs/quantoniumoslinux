import { useState, useRef, useEffect } from "react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { useChat } from "../hooks/useChat";
import { useVoice } from "../hooks/useVoice";
import { Send, Square, Trash2, Mic, MicOff, Volume2, VolumeX } from "lucide-react";
import "./ChatView.css";

export function ChatView() {
  const { messages, isStreaming, status, activeModel, sendMessage, stopStreaming, clearHistory } = useChat();
  const voice = useVoice();
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const lastMsgCountRef = useRef(0);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-speak: when streaming finishes and a new assistant message is complete
  useEffect(() => {
    if (isStreaming || !voice.autoSpeak) return;
    if (messages.length > lastMsgCountRef.current) {
      const last = messages[messages.length - 1];
      if (last?.role === "assistant" && last.content) {
        voice.speak(last.content);
      }
    }
    lastMsgCountRef.current = messages.length;
  }, [isStreaming, messages, voice.autoSpeak]);

  // When voice recognition finishes, auto-send the transcript
  useEffect(() => {
    if (!voice.isListening && voice.transcript) {
      setInput("");
      sendMessage(voice.transcript);
    }
  }, [voice.isListening, voice.transcript]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || isStreaming) return;
    setInput("");
    sendMessage(text);
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleMicClick = () => {
    if (voice.isListening) {
      voice.stopListening();
    } else {
      voice.startListening();
    }
  };

  return (
    <div className="chat-view">
      {/* Header */}
      <header className="chat-header">
        <div className="chat-header-left">
          <span className="chat-header-title">Eldrae</span>
          <span className={`conn-dot ${status.connected ? "connected" : ""}`} />
        </div>
        <div className="chat-header-right">
          {(activeModel || status.model) && <span className="chat-badge">{activeModel || status.model}</span>}
          <button
            className={`icon-btn ${voice.autoSpeak ? "icon-active" : ""}`}
            onClick={voice.autoSpeak ? () => { voice.toggleAutoSpeak(); voice.stopSpeaking(); } : voice.toggleAutoSpeak}
            title={voice.autoSpeak ? "Mute Eldrae" : "Speak responses"}
          >
            {voice.autoSpeak ? <Volume2 size={15} /> : <VolumeX size={15} />}
          </button>
          <button className="icon-btn" onClick={clearHistory} title="Clear chat">
            <Trash2 size={15} />
          </button>
        </div>
      </header>

      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <img src="/que-icon.png" alt="" className="chat-empty-logo" />
            <h2>How can I help you?</h2>
            <p>Local AI inference · Private by default</p>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`chat-bubble ${msg.role === "user" ? "user" : "ai"}`}
          >
            <div className="bubble-content">
              {msg.content || (isStreaming ? "…" : "")}
            </div>
            {msg.imagePath && (
              <img
                src={convertFileSrc(msg.imagePath)}
                alt={msg.content || "Generated image"}
                className="bubble-image"
              />
            )}
            {msg.role === "assistant" && msg.content && !isStreaming && (
              <button
                className="bubble-speak-btn"
                onClick={() => voice.isSpeaking ? voice.stopSpeaking() : voice.speak(msg.content)}
                title="Read aloud"
              >
                {voice.isSpeaking ? <VolumeX size={12} /> : <Volume2 size={12} />}
              </button>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="chat-input-area">
        {voice.isListening && (
          <div className="voice-indicator">
            <span className="voice-pulse" />
            <span className="voice-label">
              {voice.transcript || "Listening..."}
            </span>
          </div>
        )}
        {voice.error && !voice.isListening && (
          <div className="voice-error">{voice.error} — enable in Settings → Permissions</div>
        )}
        <div className="chat-input-pill">
          <textarea
            ref={inputRef}
            className="chat-input"
            value={voice.isListening ? voice.transcript : input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message Eldrae..."
            rows={1}
            readOnly={voice.isListening}
          />
          {voice.supported && !isStreaming && (
            <button
              className={`mic-btn ${voice.isListening ? "listening" : ""}`}
              onClick={handleMicClick}
              title={voice.isListening ? "Stop listening" : "Voice input"}
            >
              {voice.isListening ? <MicOff size={16} /> : <Mic size={16} />}
            </button>
          )}
          {isStreaming ? (
            <button className="send-btn stop" onClick={stopStreaming} title="Stop">
              <Square size={16} />
            </button>
          ) : (
            <button
              className={`send-btn ${input.trim() ? "active" : ""}`}
              onClick={handleSend}
              disabled={!input.trim() || voice.isListening}
              title="Send"
            >
              <Send size={16} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
