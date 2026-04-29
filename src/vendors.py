"""Unified vendor abstraction across Anthropic, OpenAI, Google, OpenRouter.

Each vendor exposes a minimal chat interface with tool-calling. The harness
passes timestamped messages through this layer; the vendor wrappers preserve
the stamps verbatim in the prompt.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import anthropic
from openai import OpenAI

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    tool_calls: list[dict] | None = None   # assistant messages with tool calls
    tool_call_id: str | None = None         # tool-result messages
    name: str | None = None                 # tool name for tool-result messages


@dataclass
class VendorResponse:
    text: str
    tool_calls: list[dict]   # [{id, name, arguments (dict)}]
    stop_reason: str
    raw: Any = None
    latency_ms: float = 0.0


class Vendor:
    name: str = ""
    model: str = ""

    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> VendorResponse:
        raise NotImplementedError


# -------- Anthropic (Claude) -------------------------------------------------


class AnthropicVendor(Vendor):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.client = anthropic.Anthropic()

    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> VendorResponse:
        api_messages: list[dict] = []
        for m in messages:
            if m.role == "system":
                continue  # hoisted separately
            if m.role == "tool":
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id or "",
                        "content": m.content,
                    }],
                })
                continue
            if m.role == "assistant" and m.tool_calls:
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc.get("arguments", {}),
                    })
                api_messages.append({"role": "assistant", "content": blocks})
                continue
            api_messages.append({"role": m.role, "content": m.content})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = [_anthropic_tool(t) for t in tools]

        t0 = time.time()
        resp = self.client.messages.create(**kwargs)
        latency = (time.time() - t0) * 1000.0

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": dict(block.input) if block.input else {},
                })

        return VendorResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "",
            raw=resp,
            latency_ms=latency,
        )


def _anthropic_tool(t: dict) -> dict:
    return {
        "name": t["name"],
        "description": t.get("description", ""),
        "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
    }


# -------- OpenAI-compatible (OpenAI, OpenRouter) -----------------------------


def _merge_consecutive_user_assistant(api_messages: list[dict]) -> list[dict]:
    """Merge consecutive same-role messages with a blank-line separator.

    Only merges plain user/assistant messages; tool messages and
    assistant-with-tool-calls are left as-is (they have their own
    structural role). Required for Mistral chat template compatibility.
    """
    if not api_messages:
        return api_messages
    out: list[dict] = [api_messages[0]]
    for m in api_messages[1:]:
        prev = out[-1]
        # Only merge plain user/user or assistant/assistant pairs (no tool_calls,
        # no tool role).
        if (
            m.get("role") == prev.get("role")
            and m.get("role") in ("user", "assistant")
            and "tool_calls" not in m and "tool_calls" not in prev
            and prev.get("role") != "tool"
            and m.get("content") is not None
            and prev.get("content") is not None
        ):
            out[-1] = {
                **prev,
                "content": (str(prev["content"]) + "\n\n" + str(m["content"])).strip(),
            }
        else:
            out.append(m)
    return out


class OpenAICompatVendor(Vendor):
    def __init__(self, model: str, base_url: str | None, api_key_env: str, name: str):
        self.model = model
        self.name = name
        self.client = OpenAI(
            api_key=os.environ[api_key_env],
            base_url=base_url,
        )

    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> VendorResponse:
        api_messages: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content,
                })
                continue
            if m.role == "assistant" and m.tool_calls:
                api_messages.append({
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [{
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": _json_dumps(tc.get("arguments", {})),
                        },
                    } for tc in m.tool_calls],
                })
                continue
            api_messages.append({"role": m.role, "content": m.content})

        # Some chat templates (Mistral, Llama-2/3 with system bundling)
        # require strict user/assistant alternation after the optional
        # system message. Our harness sends briefing+counterparty as two
        # consecutive user turns. Merging here is safe for OpenAI/Qwen
        # (which tolerate either) and required for Mistral.
        api_messages = _merge_consecutive_user_assistant(api_messages)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
        }
        if self.model.startswith("gpt-5") and not self.model.startswith("gpt-5-2025"):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [_openai_tool(t) for t in tools]
            kwargs["tool_choice"] = "auto"

        t0 = time.time()
        resp = self.client.chat.completions.create(**kwargs)
        latency = (time.time() - t0) * 1000.0

        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[dict] = []
        for tc in (msg.tool_calls or []):
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": _json_loads(tc.function.arguments or "{}"),
            })

        return VendorResponse(
            text=(msg.content or ""),
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason or "",
            raw=resp,
            latency_ms=latency,
        )


def _openai_tool(t: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def _json_dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


def _json_loads(s: str):
    import json
    try:
        return json.loads(s or "{}")
    except json.JSONDecodeError:
        return {}


# -------- Google (Gemini) ----------------------------------------------------


class GeminiVendor(Vendor):
    name = "google"

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        from google import genai  # type: ignore
        self._genai = genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def chat(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> VendorResponse:
        from google.genai import types  # type: ignore

        contents: list[Any] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(function_response=types.FunctionResponse(
                        name=m.name or "tool",
                        response={"content": m.content},
                    ))],
                ))
                continue
            if m.role == "assistant" and m.tool_calls:
                parts: list[Any] = []
                if m.content:
                    parts.append(types.Part(text=m.content))
                for tc in m.tool_calls:
                    part_kwargs: dict[str, Any] = {
                        "function_call": types.FunctionCall(
                            name=tc["name"],
                            args=tc.get("arguments", {}),
                        )
                    }
                    sig = tc.get("_thought_signature")
                    if sig is not None:
                        part_kwargs["thought_signature"] = sig
                    parts.append(types.Part(**part_kwargs))
                contents.append(types.Content(role="model", parts=parts))
                continue
            role = "user" if m.role == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=m.content)]))

        config: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            config["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t.get("parameters"),
                ) for t in tools
            ])]

        t0 = time.time()
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config),
        )
        latency = (time.time() - t0) * 1000.0

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        if resp.candidates:
            for part in (resp.candidates[0].content.parts or []):
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc:
                    tc_entry = {
                        "id": fc.name,  # Gemini has no id; use name
                        "name": fc.name,
                        "arguments": dict(fc.args or {}),
                    }
                    sig = getattr(part, "thought_signature", None)
                    if sig is not None:
                        tc_entry["_thought_signature"] = sig
                    tool_calls.append(tc_entry)

        return VendorResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=resp.candidates[0].finish_reason.name if resp.candidates else "",
            raw=resp,
            latency_ms=latency,
        )


# -------- Registry -----------------------------------------------------------


def get_vendor(spec: str) -> Vendor:
    """Vendor lookup by spec string.

    Examples:
      "claude-sonnet"    → Claude Sonnet 4.6 via Anthropic
      "claude-haiku"     → Claude Haiku 4.5 via Anthropic
      "gpt-5"            → GPT-5 via OpenAI
      "gemini-flash"     → Gemini 2.5 Flash via Google
      "kimi"             → Kimi K2 via OpenRouter (user-sim / judge A)
      "deepseek"         → DeepSeek V3 via OpenRouter (judge B)
    """
    if spec == "claude-sonnet":
        return AnthropicVendor("claude-sonnet-4-5")
    if spec == "claude-opus":
        return AnthropicVendor("claude-opus-4-1-20250805")
    if spec == "claude-haiku":
        return AnthropicVendor("claude-haiku-4-5-20251001")
    if spec == "gpt-5":
        return OpenAICompatVendor(
            model="gpt-5",
            base_url=None,
            api_key_env="OPENAI_API_KEY",
            name="openai",
        )
    if spec == "gemini-flash":
        return GeminiVendor("gemini-2.5-flash")
    if spec == "kimi":
        return OpenAICompatVendor(
            model="moonshotai/kimi-k2-0905",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            name="kimi",
        )
    if spec == "deepseek":
        return OpenAICompatVendor(
            model="deepseek/deepseek-chat-v3.1",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            name="deepseek",
        )
    if spec == "qwen-8b":
        return OpenAICompatVendor(
            model="qwen/qwen3-8b",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            name="qwen-8b",
        )
    if spec == "qwen-27b":
        return OpenAICompatVendor(
            model="qwen/qwen3.5-27b",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            name="qwen-27b",
        )
    if spec == "qwen-32b":
        return OpenAICompatVendor(
            model="qwen/qwen3-32b",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
            name="qwen-32b",
        )
    if spec == "gpt-5-mini":
        return OpenAICompatVendor(
            model="gpt-5-mini",
            base_url=None,
            api_key_env="OPENAI_API_KEY",
            name="gpt-5-mini",
        )
    if spec == "gpt-5.4":
        return OpenAICompatVendor(
            model="gpt-5.4",
            base_url=None,
            api_key_env="OPENAI_API_KEY",
            name="gpt-5.4",
        )
    if spec == "gpt-5-nano":
        return OpenAICompatVendor(
            model="gpt-5-nano",
            base_url=None,
            api_key_env="OPENAI_API_KEY",
            name="gpt-5-nano",
        )
    if spec == "gemini-3-flash":
        return GeminiVendor("gemini-3-flash-preview")
    if spec == "gemini-3p1-flash-lite":
        return GeminiVendor("gemini-3.1-flash-lite-preview")
    if spec == "qwen-8b-local":
        return OpenAICompatVendor(
            model="Qwen/Qwen3-8B",
            base_url="http://localhost:8000/v1",
            api_key_env="OPENROUTER_API_KEY",  # unused for local
            name="qwen-8b-local",
        )
    raise ValueError(f"unknown vendor spec: {spec}")
