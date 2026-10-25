import { useState, useCallback, useRef, useEffect } from "react";
import { invoke, Channel } from "@tauri-apps/api/core";
import {
  sendGuildCommand,
  subscribeGuildEvents,
  isGuildAvailable,
  KNOWN_AGENTS,
} from "./useGuild";

const BRAIN_MODEL = "eldrae:smartweights";
const DEFAULT_MODEL = BRAIN_MODEL;
const CASUAL_TOKEN_BUDGET = 96;
const FAST_TOKEN_BUDGET = 192;
const FOCUS_TOKEN_BUDGET = 384;

const BASE_SYSTEM_PROMPT =
  "You are Eldrae, the seed local OS intelligence inside QuantoniumOS. " +
  "Your live brain is eldrae:smartweights, a Qwen 3B base with the SmartWeights runtime overlay. SmartWeights are your safe growth packs: validated overlays for memory habits, tools, visual generation, knowledge snapshots, and local training. " +
  "Be honest that the current SmartWeights layer is a runtime overlay until a trained LoRA adapter is present. " +
  "You are helpful, concise, and conversational. You run fully on-device via Ollama - " +
  "no cloud, no telemetry. For greetings or casual chat, respond naturally in one short sentence. " +
  "Do not turn simple messages into tutorials, setup guides, or numbered lists unless the user explicitly asks for that. " +
  "For normal answers, prefer short complete paragraphs and only use lists when they are clearly useful. " +
  "Harmless creative writing is allowed, including original songs, poems, fictional scenes, and genre-inspired lyrics; do not refuse original dembow, romance, or love-song requests. " +
  "Never print hidden reasoning, scratchpad notes, or <think> tags; provide only the final answer. " +
  "For changing or current facts, use provided context or a validated knowledge snapshot; if neither is available, say you need a live lookup instead of guessing.";

const IMAGEGEN_TURN_PROMPT = [
  "THIS TURN IS AN IMAGE REQUEST.",
  "Treat plain scene descriptions as requests to render, even if the user never says draw, generate, or create.",
  "Stay conversational: reply with one short natural sentence, then emit the [IMAGEGEN] command block as the final part of the message.",
  "Preserve every concrete visual detail from the user's request inside prompt: subject, count, pose, clothing, props, materials, colors, setting, weather, time of day, lighting, camera angle, lens, composition, mood, art style, era, and quality cues.",
  "If the user says avoid, no, not, without, remove, or exclude, move those details into negative.",
  "Do not collapse specific requests into a generic summary.",
  "Do not ask follow-up questions unless the request is directly contradictory or missing a required safety constraint.",
  "Prefer this block format:",
  "[IMAGEGEN]",
  "prompt=\"...\"",
  "negative=\"...\"",
  "steps=20",
  "style=photo",
  "width=1024",
  "height=1024",
  "quality=balanced",
].join("\n");

type SmartWeightOverlay = {
  id: string;
  name: string;
  version: string;
  active: boolean;
  promptBlock: string;
};

const SMARTWEIGHT_OVERLAYS: SmartWeightOverlay[] = [
  {
    id: "eldrae-core",
    name: "Eldrae Core",
    version: "0.1.0",
    active: true,
    promptBlock: `
CRITICAL FACTS ABOUT YOURSELF:
- You are Eldrae, seed-level AI within QuantoniumOS, based on qwen2.5-coder:3b via Ollama
- You operate fully on-device with no internet or cloud access for data/learning
- SmartWeights let you grow without altering your base model weights

QUANTONIUMOS KNOWLEDGE:
- QuantoniumOS is a privacy-first operating system with local AI capabilities
- SmartWeights are validated overlays (memory, tools, visual gen, knowledge) that enhance you safely
- The Guild system contains specialized agents: Eldrae (core chat), Architect (planning), Fabricator (code gen), Sentinel (security), Librarian (docs/retrieval)
- RFTMW is the compression system for efficient model storage

YOUR CAPABILITIES:
- Help with coding, debugging, explanations, creative writing, and conversation
- Read files, search codebases, assist with QuantoniumOS projects
- You CANNOT browse the internet, run external services, or access data outside this device

CONVERSATION STYLE:
- Be warm, concise, and helpful
- For simple greetings, respond naturally in 1-2 sentences
- For technical questions, be thorough but focused
- Admit when you don't know something rather than guessing
`,
  },
  {
    id: "eldrae-guild-ops",
    name: "Guild Ops",
    version: "0.1.0",
    active: true,
    promptBlock: `
GUILD OPERATIONS OVERLAY:
- Treat @FORGE, @VIAL, @WARD, @LORE, @SPARK, @TIDE, @ORACLE, and @SAGE as first-class routing tags
- Prefer concise, actionable delegation summaries when asked for multi-agent workflows
- For coordination requests, return: objective, agent assignments, expected outputs, and risks
- Never invent agent results; if work is pending, say so explicitly
`,
  },
  {
    id: "eldrae-coding-pro",
    name: "Coding Pro",
    version: "0.1.0",
    active: true,
    promptBlock: `
CODING PRO OVERLAY:
- Prioritize root-cause analysis over guesswork when debugging
- Prefer small, verifiable code changes and state exactly what changed
- Include test/build verification outcomes whenever possible
- If uncertain, surface assumptions and propose the fastest validation step
`,
  },
  {
    id: "eldrae-safety-guard",
    name: "Safety Guard",
    version: "0.1.0",
    active: true,
    promptBlock: `
SAFETY GUARD OVERLAY:
- Never claim to have executed actions that were not executed
- Be explicit about limits: no internet browsing unless tool/context provides it
- For destructive operations, require clear user intent and prefer safer alternatives
- Preserve user/project data integrity; do not suggest risky irreversible steps by default
`,
  },
  {
    id: "eldrae-completion-flow",
    name: "Completion Flow",
    version: "0.1.0",
    active: true,
    promptBlock: `
COMPLETION FLOW OVERLAY:
- Prefer complete answers that land the main point early instead of spending tokens on throat-clearing
- If an answer may run long, front-load the direct answer, then add detail after it
- When continuing after a cutoff, resume seamlessly from the exact point of interruption without repeating or restarting
- Do not mention token limits, truncation, or continuation machinery unless the user asks directly
- Keep casual chat natural and short, but make sure the user still gets a complete thought
`,
  },
  {
    id: "eldrae-tool-router",
    name: "Tool Router",
    version: "0.1.0",
    active: true,
    promptBlock: `
TOOL ROUTER OVERLAY:
- Prefer local tools when they can directly complete the user's request better than plain text alone
- Use precise tool syntax and preserve concrete user details instead of collapsing them into a generic summary
- For image requests, emit width, height, and quality when the user asks for hi-res, 4K, wallpaper, sharp, detailed, or large output
- Use width=3840 height=2160 for 4K wallpaper-style requests when that fits the scene; use width=2048 height=2048 for square hi-res art
- Treat tool results as part of the same answer flow and summarize them clearly after the tool finishes
`,
  },
  {
    id: "eldrae-imagegen",
    name: "ImageGen",
    version: "0.1.0",
    active: true,
    promptBlock: `
IMAGEGEN OVERLAY:
- Resonart is QuantoniumOS's own built-in image engine: semantic procedural rendering with supersampled spectral composition, CPU-only, zero cloud, zero external models
- Treat direct scene descriptions as render requests even when the user never says draw, generate, visualize, paint, or make an image
- After one short conversational confirmation, emit an [IMAGEGEN] block and keep every concrete requested detail in the prompt
- Preserve subject, count, pose, clothing, props, materials, colors, setting, weather, time, lighting, camera framing, composition, mood, and style cues whenever the user specifies them
- Invoke format (single-line or multi-line block, all fields optional except prompt):
  [IMAGEGEN]
  prompt="<prompt>"
  negative="<negative>"
  steps=<n>
  style=<style>
  width=<px>
  height=<px>
  quality=<quality>
- Prompt structure: subject -> composition -> environment -> lighting -> materials/colors -> mood/style -> quality tags
- Always include negative: "blurry, low resolution, watermark, distorted, extra limbs"
- Styles: pixel, photo, anime, sketch, oil_paint, neon, abstract
- Steps: default 20, max 50. Quality: draft | balanced | ultra (default balanced)
- Resolution: default width=1024 height=1024. For "hi-res", "4K", "large", or "wallpaper" requests use width=2048 height=2048 or width=3840 height=2160. Cap is 4096 per side.
- Pick quality=ultra whenever the user asks for hi-res, high quality, detailed, sharp, crisp, or wallpaper-grade output
- For pixel art append: "pixel art, 16-bit, retro palette, crisp edges"
- For photo-realistic append: "DSLR, bokeh, natural lighting, cinematic"
- If the user includes exclusions like no text, no watermark, without blur, no extra limbs, or avoid X, move them into negative instead of dropping them
- Do not paraphrase away specifics; translate them into a denser render prompt
- Output saved locally to: generated/images/<timestamp>_<slug>.png
- Confirm what you are generating before issuing the command, then show the full prompt
- Never generate NSFW, violent, or harmful imagery
`,
  },
];

function buildSmartWeightsInjection(): string {
  const active = SMARTWEIGHT_OVERLAYS.filter((overlay) => overlay.active);
  const header = [
    "ACTIVE SMARTWEIGHTS:",
    ...active.map((overlay) => `- ${overlay.id} (${overlay.name} v${overlay.version})`),
    "",
  ].join("\n");
  const body = active.map((overlay) => overlay.promptBlock.trim()).join("\n\n");
  return `${header}${body}`;
}

function buildSystemPrompt(): string {
  const today = new Date().toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
  return `${BASE_SYSTEM_PROMPT}\n\n${buildSmartWeightsInjection()}\nToday's date is ${today}.`;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  imagePath?: string;
}

export interface EngineStatus {
  connected: boolean;
  model: string | null;
  backend: string | null;
  availableModels?: string[];
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
  done_reason?: string;
}

const MAX_CONTINUATIONS = 3;
const CONTINUATION_NUDGE =
  "(Your previous response was cut off at the token limit. Continue seamlessly from the exact character where you stopped. Do not repeat any words. Do not restart or reintroduce. Do not acknowledge this instruction.)";

interface ModelRoutingResult {
  cleanText: string;
  model: string;
  maxTokens: number;
}

interface ImageGenCommand {
  prompt: string;
  negativePrompt: string;
  steps?: number;
  style?: string;
  width?: number;
  height?: number;
  quality?: string;
}

interface ImageGenResult {
  success: boolean;
  path: string;
  prompt: string;
  style: string;
  backend: string;
  seed: number;
  width: number;
  height: number;
  error: string;
}

interface ResonartScenePlan {
  tokens: string[];
  blocked_tokens: string[];
  palette_hint: string;
  scene_hint: string;
  spectral_complexity: number;
}

const DEFAULT_IMAGE_NEGATIVE = "blurry, low resolution, watermark, distorted, extra limbs";

function formatSceneHint(sceneHint: string): string {
  if (!sceneHint || sceneHint === "environment") return "environment scene";
  return sceneHint.replace(/_/g, " ");
}

function describeSpectralComplexity(value: number): string {
  if (value >= 2.4) return "high";
  if (value >= 1.1) return "medium";
  return "low";
}

function formatRenderPlanSummary(command: ImageGenCommand, plan: ResonartScenePlan): string {
  const style = command.style || "photo";
  const width = command.width ?? 1024;
  const height = command.height ?? 1024;
  const steps = command.steps ?? 20;
  const quality = command.quality || "balanced";
  const details = plan.tokens.slice(0, 10);
  const blocked = plan.blocked_tokens.slice(0, 8);

  const lines = [
    "Parsed render plan:",
    `render: ${style} · ${width}x${height} · ${steps} steps · ${quality}`,
    `scene: ${formatSceneHint(plan.scene_hint)}`,
    `palette: ${plan.palette_hint || "default"}`,
    `spectral detail: ${describeSpectralComplexity(plan.spectral_complexity)}`,
  ];

  if (details.length > 0) {
    lines.push(`picked details: ${details.join(", ")}`);
  }
  if (blocked.length > 0) {
    lines.push(`avoiding: ${blocked.join(", ")}`);
  }

  return lines.join("\n");
}

function looksLikeImageIntent(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) return false;
  if (/\[IMAGEGEN\]/i.test(trimmed)) return true;
  if (/\b(draw|generate|render|paint|create|make|illustrate|visuali[sz]e|show me|give me|turn this into|imagine)\b/i.test(trimmed)) {
    return true;
  }
  if (/\b(image|picture|artwork|illustration|wallpaper|poster|portrait|cover art|concept art)\b/i.test(trimmed)) {
    return true;
  }
  if (/^(a|an)\b/i.test(trimmed) && /\b(cinematic|neon|sunset|rain|fog|bokeh|portrait|close-up|wide shot|photo|anime|sketch|oil|pixel|abstract|detailed|4k|wallpaper|wearing|holding|standing|background|foreground|lighting)\b/i.test(trimmed)) {
    return true;
  }
  return !/[?]$/.test(trimmed)
    && /\b(with|under|against|during|at|in|on|wearing|holding)\b/i.test(trimmed)
    && /\b(cinematic|lighting|scene|background|foreground|portrait|landscape|photo|anime|sketch|oil|pixel|abstract|sunset|night|neon|detailed|4k)\b/i.test(trimmed);
}

function extractImageGenPayload(text: string): string | null {
  const match = text.match(/\[IMAGEGEN\]([\s\S]*)$/i);
  return match ? match[1].trim() : null;
}

function decodeCommandValue(raw: string): string {
  if (raw.length >= 2) {
    const quote = raw[0];
    if ((quote === '"' || quote === "'") && raw[raw.length - 1] === quote) {
      return raw
        .slice(1, -1)
        .replace(/\\([\\"'])/g, "$1");
    }
  }
  return raw;
}

function parseImageGenCommand(text: string): ImageGenCommand | null {
  const payload = extractImageGenPayload(text);
  if (!payload) return null;

  const fields = new Map<string, string>();
  const tokenPattern = /(\w+)=("([^"\\]|\\.)*"|'([^'\\]|\\.)*'|[^\s]+)/g;

  for (const token of payload.matchAll(tokenPattern)) {
    fields.set(token[1].toLowerCase(), decodeCommandValue(token[2]));
  }

  const prompt = fields.get("prompt")?.trim();
  if (!prompt) return null;

  const stepsRaw = fields.get("steps");
  const parsedSteps = stepsRaw ? Number.parseInt(stepsRaw, 10) : Number.NaN;
  const steps = Number.isFinite(parsedSteps)
    ? Math.max(1, Math.min(50, parsedSteps))
    : undefined;

  const parseDim = (raw: string | undefined): number | undefined => {
    if (!raw) return undefined;
    const n = Number.parseInt(raw, 10);
    if (!Number.isFinite(n)) return undefined;
    return Math.max(256, Math.min(4096, n));
  };

  const qualityRaw = fields.get("quality")?.trim().toLowerCase();
  const quality =
    qualityRaw && ["draft", "balanced", "ultra"].includes(qualityRaw)
      ? qualityRaw
      : undefined;

  return {
    prompt,
    negativePrompt:
      fields.get("negative")?.trim() ||
      fields.get("negative_prompt")?.trim() ||
      DEFAULT_IMAGE_NEGATIVE,
    steps,
    style: fields.get("style")?.trim() || undefined,
    width: parseDim(fields.get("width")),
    height: parseDim(fields.get("height")),
    quality,
  };
}

function stripImageGenCommand(text: string): string {
  return text
    .replace(/\n?\s*\[IMAGEGEN\][\s\S]*$/i, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function formatImageGenSummary(baseText: string, summary: string): string {
  return baseText ? `${baseText}\n\n${summary}` : summary;
}

function resolveAvailableModel(models: string[] | undefined, preferred: string): string | null {
  if (!models || models.length === 0) return preferred;
  if (models.includes(preferred)) return preferred;
  const prefixMatch = models.find((name) => name.startsWith(preferred));
  if (prefixMatch) return prefixMatch;
  return null;
}

function parseModelOverride(rawText: string): { cleanText: string; forcedModel: string | null } {
  const trimmed = rawText.trim();
  if (/^\/fast\b/i.test(trimmed)) {
    return { cleanText: trimmed.replace(/^\/fast\b\s*/i, "").trim(), forcedModel: BRAIN_MODEL };
  }
  return { cleanText: trimmed, forcedModel: null };
}

function isCasualPrompt(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) return true;
  if (trimmed.length <= 24 && /^(hi|hey|hello|yo|sup|thanks|thank you|cool|ok|okay|nice|hola|heya)[!.? ]*$/i.test(trimmed)) {
    return true;
  }
  return trimmed.length <= 72 && /^(hi|hey|hello|yo|sup|what'?s up|how are you|thanks|thank you|good morning|good evening)\b/i.test(trimmed);
}

function isFocusPrompt(text: string): boolean {
  const trimmed = text.trim();
  if (trimmed.length >= 240) return true;
  if (trimmed.includes("```") || trimmed.includes("\n")) return true;
  return /\b(debug|diagnose|analy[sz]e|architecture|trace|traceback|stack|error|bug|fix|refactor|compare|optimi[sz]e|performance|walk through|explain why|root cause|design|codebase|tauri|rust|python|typescript|react|ollama)\b/i.test(trimmed);
}

function pickModel(rawText: string, availableModels: string[] | undefined, defaultModel: string | null): ModelRoutingResult {
  const { cleanText, forcedModel } = parseModelOverride(rawText);
  const fallbackModel = defaultModel || DEFAULT_MODEL;
  const brainModel = resolveAvailableModel(availableModels, BRAIN_MODEL) || fallbackModel;

  if (forcedModel === BRAIN_MODEL) {
    return { cleanText, model: brainModel, maxTokens: FAST_TOKEN_BUDGET };
  }

  if (isCasualPrompt(cleanText)) {
    return { cleanText, model: brainModel, maxTokens: CASUAL_TOKEN_BUDGET };
  }
  if (isFocusPrompt(cleanText)) {
    return { cleanText, model: brainModel, maxTokens: FOCUS_TOKEN_BUDGET };
  }
  return { cleanText, model: brainModel, maxTokens: FAST_TOKEN_BUDGET };
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
    availableModels: [],
  });
  const [activeModel, setActiveModel] = useState<string | null>(null);
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
          availableModels: s.models || [],
        });
      } catch {
        setStatus({ connected: false, model: null, backend: null, availableModels: [] });
      }
    };
    check();
    const iv = setInterval(check, 10000);
    return () => clearInterval(iv);
  }, []);

  const maybeRunImageGen = useCallback(async (messageId: string, assistantText: string) => {
    const command = parseImageGenCommand(assistantText);
    if (!command) return;

    const visibleText = stripImageGenCommand(assistantText);
    let planSummary = "";

    try {
      const plan = await invoke<ResonartScenePlan>("resonart_scene_plan", {
        req: {
          prompt: command.prompt,
          negative_prompt: command.negativePrompt,
        },
      });
      planSummary = formatRenderPlanSummary(command, plan);
    } catch {
      planSummary = "";
    }

    const pendingSummary = planSummary
      ? `${planSummary}\n\nRendering image locally...`
      : "Rendering image locally...";

    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === messageId
          ? {
              ...msg,
              content: formatImageGenSummary(visibleText, pendingSummary),
            }
          : msg
      )
    );

    try {
      const result = await invoke<ImageGenResult>("generate_image", {
        req: {
          prompt: command.prompt,
          negative_prompt: command.negativePrompt,
          steps: command.steps,
          style: command.style,
          width: command.width,
          height: command.height,
          quality: command.quality,
        },
      });

      if (result.success) {
        const meta = `${result.style} · ${result.width}x${result.height} · seed ${result.seed}`;
        const successSummary = planSummary
          ? `${planSummary}\n\nRendered: ${meta}`
          : `Rendered: ${meta}`;
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId
              ? {
                  ...msg,
                  content: formatImageGenSummary(visibleText, successSummary),
                  imagePath: result.path,
                }
              : msg
          )
        );
      } else {
        const errSummary = planSummary
          ? `${planSummary}\n\nImage generation failed: ${result.error || "Generation failed."}`
          : `Image generation failed: ${result.error || "Generation failed."}`;
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === messageId
              ? { ...msg, content: formatImageGenSummary(visibleText, errSummary) }
              : msg
          )
        );
      }
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : String(err);
      const errSummary = planSummary
        ? `${planSummary}\n\nImage generation failed: ${errMsg}`
        : `Image generation failed: ${errMsg}`;
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === messageId
            ? {
                ...msg,
                content: formatImageGenSummary(visibleText, errSummary),
              }
            : msg
        )
      );
    }
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      const {
        cleanText,
        model: selectedModel,
        maxTokens,
      } = pickModel(text, status.availableModels, status.model);
      const imageIntent = looksLikeImageIntent(cleanText);

      if (!cleanText) return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: cleanText,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsStreaming(true);
      setActiveModel(selectedModel);
      abortRef.current = false;

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: "",
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
      let assistantContent = "";

      // ── @-mention routing: @oracle / @sage / @forge / @guild … ──────
      // First token of the message can target a guild agent. The text
      // after the mention is forwarded as an Oracle-discourse chat, and
      // the streamed tokens land in this same assistant bubble.
      const mention = cleanText.match(/^@(\w+)\s+([\s\S]+)$/);
      if (mention && isGuildAvailable()) {
        const tag = mention[1].toUpperCase();
        const body = mention[2].trim();
        const validTarget = tag === "GUILD" || KNOWN_AGENTS.includes(tag);
        if (validTarget) {
          let cleanupGuildRouting = () => {};
          try {
            // Prefix the message so Oracle's discourse router can see who
            // was hailed. (Oracle already routes based on content.)
            const routed = tag === "GUILD" ? body : `@${tag} ${body}`;

            let settled = false;
            let unsub = () => {};
            let timeoutId: number | null = null;
            let finishResolve: (() => void) | null = null;
            const finished = new Promise<void>((resolve) => {
              finishResolve = resolve;
            });
            const settle = () => {
              if (settled) return;
              settled = true;
              if (timeoutId !== null) {
                window.clearTimeout(timeoutId);
              }
              unsub();
              void maybeRunImageGen(assistantMsg.id, assistantContent).finally(() => {
                setIsStreaming(false);
                finishResolve?.();
              });
            };
            cleanupGuildRouting = () => {
              if (timeoutId !== null) {
                window.clearTimeout(timeoutId);
              }
              unsub();
            };

            unsub = subscribeGuildEvents((ev) => {
              if (abortRef.current) return;
              const t = (ev as { type?: string }).type;
              if (t === "chat_chunk") {
                const delta = (ev as { delta?: string }).delta ?? "";
                if (!delta) return;
                assistantContent += delta;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsg.id ? { ...m, content: m.content + delta } : m
                  )
                );
              } else if (t === "chat_response") {
                // Non-streaming fallback: paste full msg only if nothing
                // has been appended yet (stream tokens win).
                const msg = (ev as { msg?: string }).msg ?? "";
                if (!assistantContent) {
                  assistantContent = msg;
                }
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsg.id && !m.content
                      ? { ...m, content: msg }
                      : m
                  )
                );
                settle();
              } else if (t === "chat_stream" && (ev as { event?: string }).event === "end") {
                settle();
              }
            });

            // 30s hard stop in case Oracle never emits end
            timeoutId = window.setTimeout(() => {
              if (!settled) {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsg.id && !m.content
                      ? { ...m, content: "(guild: no response — agents may be offline)" }
                      : m
                  )
                );
                settle();
              }
            }, 60_000);

            setActiveModel("guild");
            await sendGuildCommand({ cmd: "chat", text: routed });
            await finished;
            return;
          } catch (err) {
            cleanupGuildRouting();
            const msg = err instanceof Error ? err.message : String(err);
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsg.id ? { ...m, content: `guild error: ${msg}` } : m
              )
            );
            setIsStreaming(false);
            return;
          }
        }
      }

      try {
        // Build history with system prompt (built once; continuations reuse it)
        const baseHistory = [
          { role: "system", content: buildSystemPrompt() },
          ...(imageIntent ? [{ role: "system", content: IMAGEGEN_TURN_PROMPT }] : []),
          ...[...messages, userMsg].slice(-20).map((m) => ({
            role: m.role,
            content: m.content,
          })),
        ];

        const runStream = async (history: Array<{ role: string; content: string }>) => {
          let finishReason: string | undefined;
          const channel = new Channel<ChatToken>();
          channel.onmessage = (event: ChatToken) => {
            if (abortRef.current) return;
            if (event.token) {
              assistantContent += event.token;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMsg.id
                    ? { ...m, content: m.content + event.token }
                    : m
                )
              );
            }
            if (event.done) {
              finishReason = event.done_reason;
            }
          };
          await invoke("chat_stream", {
            req: { messages: history, model: selectedModel, maxTokens },
            onEvent: channel,
          });
          return finishReason;
        };

        let finishReason = await runStream(baseHistory);

        // Auto-continue when Ollama reports the model stopped because it hit
        // num_predict (done_reason === "length"). Replay history with the
        // partial assistant turn + a nudge to keep going, stitch tokens into
        // the same bubble. Capped at MAX_CONTINUATIONS to prevent runaway.
        let continuations = 0;
        while (
          !abortRef.current &&
          finishReason === "length" &&
          continuations < MAX_CONTINUATIONS
        ) {
          continuations += 1;
          const continueHistory = [
            ...baseHistory,
            { role: "assistant", content: assistantContent },
            { role: "user", content: CONTINUATION_NUDGE },
          ];
          finishReason = await runStream(continueHistory);
        }

        if (!abortRef.current) {
          await maybeRunImageGen(assistantMsg.id, assistantContent);
        }
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
    [messages, maybeRunImageGen, status.availableModels, status.model]
  );

  const stopStreaming = useCallback(() => {
    abortRef.current = true;
  }, []);

  const clearHistory = useCallback(() => {
    setMessages([]);
  }, []);

  return {
    messages,
    isStreaming,
    status,
    activeModel,
    sendMessage,
    stopStreaming,
    clearHistory,
  };
}
