"""Native Anthropic Messages API adapter behind the pilot's OpenAI-style client.

Anthropic's OpenAI-compatible endpoint silently drops ``cache_control``
markers (measured: $25.75/game with 0 cached tokens), so for
``provider="anthropic"`` the pilot talks to the Messages API directly. This
class exposes only the slice of ``AsyncOpenAI`` the pilot loop uses —
``client.chat.completions.create(...)`` — and returns genuine
``openai.types.chat.ChatCompletion`` objects, so the loop, cost tracking and
trace logging are untouched. Errors are re-raised as ``OpenAIError`` so the
loop's existing recovery path handles them.
"""

from __future__ import annotations

import json
from typing import Protocol

import anthropic
from openai import OpenAIError
from openai.types.chat import ChatCompletion

from magebench.common.log import get_logger

logger = get_logger(__name__)


class AnthropicNativeError(OpenAIError):
    """An Anthropic API failure, surfaced through the pilot's OpenAIError path."""


def _text_blocks(content: str | list | None) -> list[dict]:
    """OpenAI message content (str or parts) -> Anthropic text blocks, keeping cache_control."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        block: dict = {"type": "text", "text": part["text"]}
        if part.get("cache_control"):
            block["cache_control"] = part["cache_control"]
        blocks.append(block)
    return blocks


def _message_cache_control(msg: dict) -> dict | None:
    """A cache_control marker placed on the message or on any of its content parts."""
    marker = msg.get("cache_control")
    if isinstance(marker, dict):
        return marker
    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("cache_control"), dict):
                return dict(part["cache_control"])
    return None


def _mark_last(blocks: list[dict], cache_control: dict | None) -> list[dict]:
    if cache_control and blocks:
        blocks[-1] = {**blocks[-1], "cache_control": cache_control}
    return blocks


def convert_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """OpenAI chat messages -> (system blocks, Anthropic messages).

    - ``system`` messages become the top-level ``system`` param.
    - ``assistant`` messages with ``tool_calls`` become text + ``tool_use`` blocks.
    - ``tool`` results become ``tool_result`` blocks inside a ``user`` message;
      consecutive results are merged into one message, as the API requires.
    - Any ``cache_control`` marker on a message/part lands on that message's
      final block (the API caches the prefix up to and including that block).
    """
    system: list[dict] = []
    out: list[dict] = []

    def append(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for msg in messages:
        role = msg.get("role")
        cc = _message_cache_control(msg)
        if role == "system":
            system.extend(_mark_last(_text_blocks(msg.get("content")), cc))
        elif role == "user":
            append("user", _mark_last(_text_blocks(msg.get("content")), cc))
        elif role == "assistant":
            blocks = _text_blocks(msg.get("content"))
            tool_calls = msg.get("tool_calls")
            for call in tool_calls if tool_calls is not None else []:
                fn = call["function"]
                # The pilot already validated these arguments as JSON before recording them.
                args = json.loads(fn["arguments"]) if fn["arguments"] else {}
                blocks.append({"type": "tool_use", "id": call["id"], "name": fn["name"], "input": args})
            append("assistant", _mark_last(blocks, cc))
        elif role == "tool":
            content = msg.get("content")
            # The API rejects cache_control inside tool_result.content; the
            # marker (captured in `cc`) goes on the tool_result block itself.
            result_content: str | list[dict] = (
                content
                if isinstance(content, str)
                else [{k: v for k, v in b.items() if k != "cache_control"} for b in _text_blocks(content)]
            )
            block: dict = {"type": "tool_result", "tool_use_id": msg["tool_call_id"], "content": result_content}
            if cc:
                block["cache_control"] = cc
            append("user", [block])
        else:
            raise ValueError(f"unsupported message role: {role!r}")

    # The API rejects a conversation that doesn't start with a user turn.
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(begin)"}]})
    return system, out


def convert_tools(tools: list[dict] | None) -> list[dict]:
    """OpenAI function tools -> Anthropic tool definitions."""
    converted: list[dict] = []
    for tool in tools if tools is not None else []:
        fn = tool["function"]
        converted.append({"name": fn["name"], "description": fn["description"], "input_schema": fn["parameters"]})
    return converted


def convert_tool_choice(tool_choice: str | dict | None) -> dict | None:
    if tool_choice in (None, "auto"):
        return {"type": "auto"} if tool_choice == "auto" else None
    if tool_choice == "required":
        return {"type": "any"}
    if tool_choice == "none":
        return None
    if isinstance(tool_choice, dict):
        return {"type": "tool", "name": tool_choice["function"]["name"]}
    raise ValueError(f"unsupported tool_choice: {tool_choice!r}")


class _UsageLike(Protocol):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None


class _MessageLike(Protocol):
    """The slice of ``anthropic.types.Message`` we read (duck-typed for tests)."""

    id: str
    stop_reason: str | None
    content: list  # text / tool_use blocks
    usage: _UsageLike


# A non-streaming response always carries a stop_reason; unknown ones crash loudly.
_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


def to_chat_completion(response: _MessageLike, model: str) -> ChatCompletion:
    """Anthropic Message -> OpenAI ChatCompletion (with cached-token details)."""
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                }
            )
    assert response.stop_reason is not None, f"response {response.id} has no stop_reason"
    message: dict = {"role": "assistant", "content": "".join(text_parts) if text_parts else None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = response.usage
    # The API omits the cache buckets (None) when no caching happened.
    cache_read = usage.cache_read_input_tokens if usage.cache_read_input_tokens is not None else 0
    cache_write = usage.cache_creation_input_tokens if usage.cache_creation_input_tokens is not None else 0
    # OpenAI semantics: prompt_tokens is the whole prompt, cached_tokens the
    # subset served from cache. Anthropic reports the uncached remainder as
    # input_tokens, so add the cache buckets back.
    prompt_tokens = usage.input_tokens + cache_read + cache_write
    completion_tokens = usage.output_tokens
    payload = {
        "id": response.id,
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _STOP_REASONS[response.stop_reason],
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cache_read},
            # Not an OpenAI field; kept for cost analysis of cache writes.
            "cache_creation_input_tokens": cache_write,
        },
    }
    return ChatCompletion.model_validate(payload)


class _Completions:
    def __init__(self, client: anthropic.AsyncAnthropic):
        self._client = client

    async def create(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,  # OpenRouter-only knobs; ignored here
    ) -> ChatCompletion:
        del extra_body
        system, converted_messages = convert_messages(messages)
        request: dict = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": max_tokens or 1024,
        }
        if system:
            request["system"] = system
        converted_tools = convert_tools(tools)
        if converted_tools:
            request["tools"] = converted_tools
            choice = convert_tool_choice(tool_choice)
            if choice:
                request["tool_choice"] = choice
        try:
            response = await self._client.messages.create(**request)
        except anthropic.APIError as exc:
            raise AnthropicNativeError(f"{type(exc).__name__}: {exc}") from exc
        return to_chat_completion(response, model)


class _Chat:
    def __init__(self, client: anthropic.AsyncAnthropic):
        self.completions = _Completions(client)


class AnthropicNativeClient:
    """Drop-in for the ``AsyncOpenAI`` surface used by the pilot loop."""

    def __init__(self, api_key: str, timeout: float, max_retries: int = 1):
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.chat = _Chat(self._client)
