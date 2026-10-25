"""OpenAI-compatible chat client for local model servers (vLLM, Ollama, llama.cpp).

Local-only by default: there is no cloud fallback. Tool calls are normalised so the
controller sees the same shape whether the server returned native `tool_calls` or
the model wrote a JSON tool call in plain text (common with small models).
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ModelError(RuntimeError):
    pass


class ContextOverflow(ModelError):
    """The server rejected the request, usually because the prompt is too long."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None  # None when the model sent unparseable JSON
    raw: str = ""
    error: str = ""             # explicit parse diagnostic; never dispatch this call


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall]
    native: bool                      # True: server-side tool_calls; False: parsed from text
    raw_tool_calls: list = field(default_factory=list)
    reasoning: str | None = None
    prompt_tokens: int | None = None
    finish_reason: str | None = None
    completion_tokens: int | None = None  # None means the backend did not report usage


class ChatClient:
    def __init__(self, base_url: str, model: str, *, temperature: float = 0.0,
                 max_tokens: int = 4096, timeout: float = 900, retries: int = 3,
                 allow_remote: bool = False, api_key: str = "local"):
        host = urllib.parse.urlparse(base_url).hostname
        if host not in LOCAL_HOSTS and not allow_remote:
            raise ValueError(f"Refusing non-local model server {host!r}; pass allow_remote=True to override.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.api_key = api_key
        # Local traffic must not go through an HTTP proxy configured for the internet.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, route: str, body: dict | None = None, timeout: float | None = None) -> dict:
        req = urllib.request.Request(
            self.base_url + route, data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        err: Exception | None = None
        for attempt in range(self.retries):
            try:
                with self._opener.open(req, timeout=timeout or self.timeout) as r:
                    return json.load(r)
            except urllib.error.HTTPError as exc:
                detail = exc.read(4000).decode(errors="replace")
                if exc.code == 400 and any(marker in detail.lower() for marker in (
                        "context_length_exceeded", "maximum context length", "context window",
                        "exceeds the context", "too many tokens", "prompt is too long")):
                    raise ContextOverflow(detail) from exc
                if 400 <= exc.code < 500 and exc.code not in (408, 429):
                    raise ModelError(f"HTTP {exc.code}: {detail}") from exc
                err = ModelError(f"HTTP {exc.code}: {detail}")
            except (OSError, ValueError) as exc:
                err = exc
            time.sleep(2 * (attempt + 1))
        raise ModelError(f"Model server unavailable after {self.retries} attempts: {err}")

    def models(self) -> list[str]:
        return [m.get("id", "") for m in self._request("/models", timeout=10).get("data", [])]

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_names: set[str] | None = None) -> Reply:
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        data = self._request("/chat/completions", body)
        choices = data.get("choices") or []
        if not choices:
            raise ModelError(f"No choices in response: {str(data)[:500]}")
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        raw_calls = msg.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raw_calls = [raw_calls]
        calls, normalized = [], []
        for i, c in enumerate(raw_calls):
            error = ""
            if not isinstance(c, dict) or not isinstance(c.get("function"), dict):
                error = "Native tool call must contain a function object."
                c = {}
            fn = c.get("function") or {}
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                error = error or "Tool name must be a nonempty string."
                name = "invalid_tool_call"
            raw_args = fn.get("arguments")
            args = None
            if isinstance(raw_args, dict):
                args = raw_args
            elif isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except ValueError as exc:
                    error = error or f"Invalid arguments JSON: {exc}"
            if not isinstance(args, dict):
                error = error or "Tool arguments must be a JSON object."
                args = None
            call_id = c.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = f"call_{i}"
            calls.append(ToolCall(call_id, name, args, str(raw_args), error))
            # Repair observations need a well-formed native transcript even when
            # the server emitted malformed call metadata.
            normalized.append({"id": call_id, "type": "function", "function": {
                "name": name, "arguments": raw_args if isinstance(raw_args, str) else json.dumps(raw_args)}})
        raw_calls = normalized
        native = bool(calls)
        if not calls and content:
            calls = parse_text_tool_calls(content, tool_names)
        return Reply(content=content, tool_calls=calls, native=native, raw_tool_calls=raw_calls,
                     reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
                     prompt_tokens=(data.get("usage") or {}).get("prompt_tokens"),
                     finish_reason=choices[0].get("finish_reason"),
                     completion_tokens=(data.get("usage") or {}).get("completion_tokens"))


_TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_text_tool_calls(text: str, tool_names: set[str] | None = None) -> list[ToolCall]:
    """Find JSON tool calls written as text: <tool_call>{...}</tool_call>, ```json {...}```,
    or a bare object with "name" and "arguments"/"parameters"."""
    calls: list[ToolCall] = []

    def invalid(chunk: str, message: str):
        calls.append(ToolCall(f"text_{len(calls)}", "invalid_tool_call", None,
                              chunk[:2000], message))

    def append_object(obj, chunk: str):
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            invalid(chunk, "Expected a tool-call object with a string name.")
            return
        args = obj.get("arguments", obj.get("parameters", {}))
        error = ""
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError as exc:
                error = f"Invalid arguments JSON: {exc}"
                args = None
        if not isinstance(args, dict) and not error:
            error = "Tool arguments must be a JSON object."
        if tool_names is not None and obj["name"] not in tool_names:
            error = f"Tool {obj['name']!r} is not available now."
        calls.append(ToolCall(f"text_{len(calls)}", obj["name"],
                              args if isinstance(args, dict) else None, chunk[:2000], error))

    # Explicit envelopes contain exactly one complete object. Do not scan their
    # contents for nested objects or discard unmatched envelopes elsewhere.
    if "<tool_call>" in text or "</tool_call>" in text:
        matches = list(_TAGGED.finditer(text))
        outside = _TAGGED.sub("", text)
        if "<tool_call>" in outside or "</tool_call>" in outside:
            invalid(outside, "Unclosed or unmatched tool_call tag.")
        elif re.search(r"[\[{]", outside):
            invalid(outside, "Tool JSON outside a tool_call envelope; use one consistent format.")
        for match in matches:
            chunk = match.group(1)
            try:
                append_object(json.loads(chunk), chunk)
            except ValueError as exc:
                invalid(chunk, f"Invalid tool-call JSON: {exc}")
        return calls

    fenced = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
    if "```" in text:
        matches = list(fenced.finditer(text))
        outside = fenced.sub("", text)
        if "```" in outside:
            invalid(text, "Unclosed tool-call code fence.")
        elif re.search(r"[\[{]", outside):
            invalid(outside, "Tool JSON outside a code fence; use one consistent format.")
        candidates = [m.group(1) for m in matches]
    else:
        # Retain the legacy prose-prefix / bare concatenated-object fallback.
        start = re.search(r"[\[{]", text)
        if start is None:
            return []
        candidates = [text[start.start():]]
    decoder = json.JSONDecoder()
    for chunk in candidates:
        pos = 0
        while pos < len(chunk):
            while pos < len(chunk) and chunk[pos].isspace():
                pos += 1
            if pos == len(chunk):
                break
            try:
                obj, end = decoder.raw_decode(chunk, pos)
            except ValueError as exc:
                invalid(chunk[pos:], f"Invalid tool-call JSON: {exc}")
                break
            append_object(obj, chunk[pos:end])
            pos = end
    return calls
