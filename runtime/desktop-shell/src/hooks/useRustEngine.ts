/**
 * useRustEngine — Hook for calling Rust inference engine via Tauri commands.
 * 
 * Provides direct access to the RFTMW Rust inference backend for:
 * - Eldrae's Qwen 3B seed brain
 * - 35B models (qwen35-opus-distil)
 */
import { useState, useEffect, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";

export interface RustEngineStatus {
  available_7b: boolean;
  available_35b: boolean;
  backend: string;
}

export interface RustInferResult {
  tokens: number[];
  engine: string;
}

// Simple BPE-style tokenizer for ChatML format
// This is a placeholder - real tokenization should match the model's vocabulary
const SPECIAL_TOKENS: Record<string, number> = {
  "<|im_start|>": 151644,
  "<|im_end|>": 151645,
  "<|endoftext|>": 151643,
};

function simpleTokenize(text: string): number[] {
  // Very basic tokenization - in production, use the model's actual tokenizer
  const tokens: number[] = [];
  let remaining = text;
  
  // Handle special tokens
  for (const [token, id] of Object.entries(SPECIAL_TOKENS)) {
    while (remaining.includes(token)) {
      const idx = remaining.indexOf(token);
      if (idx > 0) {
        // Tokenize text before special token (very simple byte-level)
        const before = remaining.slice(0, idx);
        for (const char of before) {
          tokens.push(char.charCodeAt(0) + 256); // Simple offset
        }
      }
      tokens.push(id);
      remaining = remaining.slice(idx + token.length);
    }
  }
  
  // Remaining text
  for (const char of remaining) {
    tokens.push(char.charCodeAt(0) + 256);
  }
  
  return tokens;
}

function simpleDetokenize(tokens: number[]): string {
  let result = "";
  for (const token of tokens) {
    // Handle special tokens
    const special = Object.entries(SPECIAL_TOKENS).find(([_, id]) => id === token);
    if (special) {
      result += special[0];
    } else if (token >= 256 && token < 512) {
      // Simple byte-level decode
      result += String.fromCharCode(token - 256);
    } else {
      // Unknown token - skip
    }
  }
  return result;
}

export function useRustEngine() {
  const [status, setStatus] = useState<RustEngineStatus>({
    available_7b: false,
    available_35b: false,
    backend: "unknown",
  });
  const [initializing, setInitializing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Check status on mount
  useEffect(() => {
    checkStatus();
  }, []);

  const checkStatus = useCallback(async () => {
    try {
      const s = await invoke<RustEngineStatus>("rust_engine_status");
      setStatus(s);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  const initialize = useCallback(async () => {
    setInitializing(true);
    setError(null);
    try {
      const current = await invoke<RustEngineStatus>("rust_engine_status");
      if (current.available_7b || current.available_35b) {
        setStatus(current);
        return "Rust engine already ready";
      }

      const result = await invoke<string>("rust_engine_init");
      await checkStatus();
      return result;
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      throw e;
    } finally {
      setInitializing(false);
    }
  }, [checkStatus]);

  const inference = useCallback(async (
    prompt: string,
    options?: {
      maxTokens?: number;
      temperature?: number;
      use35b?: boolean;
    }
  ): Promise<string> => {
    // Format as ChatML
    const chatml = `<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
${prompt}<|im_end|>
<|im_start|>assistant
`;
    
    const tokens = simpleTokenize(chatml);
    
    const result = await invoke<RustInferResult>("rust_inference", {
      req: {
        tokens,
        max_tokens: options?.maxTokens ?? 128,
        temperature: options?.temperature ?? 0.7,
        use_35b: options?.use35b ?? false,
      },
    });
    
    return simpleDetokenize(result.tokens);
  }, []);

  return {
    status,
    initializing,
    error,
    initialize,
    inference,
    checkStatus,
  };
}

export { simpleTokenize, simpleDetokenize };
