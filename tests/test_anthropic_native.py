"""Tests for the native Anthropic adapter (OpenAI-shaped in, OpenAI-shaped out)."""

import json
from types import SimpleNamespace

import pytest
from openai import OpenAIError

from magebench.pilot.anthropic_native import (
    AnthropicNativeError,
    convert_messages,
    convert_tool_choice,
    convert_tools,
    to_chat_completion,
)

CC = {"type": "ephemeral"}


def test_system_message_becomes_system_param_with_cache_control() -> None:
    system, messages = convert_messages(
        [
            {"role": "system", "content": [{"type": "text", "text": "rules", "cache_control": CC}]},
            {"role": "user", "content": "go"},
        ]
    )
    assert system == [{"type": "text", "text": "rules", "cache_control": CC}]
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "go"}]}]


def test_assistant_tool_calls_become_tool_use_blocks() -> None:
    _, messages = convert_messages(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "thinking",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "pass_priority", "arguments": "{}"}}
                ],
            },
        ]
    )
    assert messages[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "thinking"},
            {"type": "tool_use", "id": "call_1", "name": "pass_priority", "input": {}},
        ],
    }


def test_consecutive_tool_results_merge_into_one_user_message() -> None:
    _, messages = convert_messages(
        [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "x", "arguments": "{}"}},
                    {"id": "b", "type": "function", "function": {"name": "y", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "ra"},
            {"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "rb", "cache_control": CC}]},
        ]
    )
    assert messages[-1]["role"] == "user"
    results = messages[-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert results[0] == {"type": "tool_result", "tool_use_id": "a", "content": "ra"}
    # A marker inside a tool result lands on the tool_result block itself —
    # never inside its content (the API rejects that with a 400).
    assert results[1]["cache_control"] == CC
    assert results[1]["content"] == [{"type": "text", "text": "rb"}]
    assert all("cache_control" not in inner for inner in results[1]["content"])


def test_cache_control_marks_last_block_of_marked_user_message() -> None:
    _, messages = convert_messages(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b", "cache_control": CC}],
            }
        ]
    )
    blocks = messages[0]["content"]
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == CC


def test_conversation_is_forced_to_start_with_user_turn() -> None:
    _, messages = convert_messages([{"role": "assistant", "content": "hi"}])
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_tools_and_tool_choice_conversion() -> None:
    tools = convert_tools(
        [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]
    )
    assert tools == [{"name": "f", "description": "d", "input_schema": {"type": "object"}}]
    assert convert_tool_choice("auto") == {"type": "auto"}
    assert convert_tool_choice("required") == {"type": "any"}
    assert convert_tool_choice({"type": "function", "function": {"name": "f"}}) == {"type": "tool", "name": "f"}
    assert convert_tool_choice("none") is None


def _fake_response(**usage: int) -> SimpleNamespace:
    return SimpleNamespace(
        id="msg_1",
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="text", text="I'll pass."),
            SimpleNamespace(type="tool_use", id="toolu_1", name="pass_priority", input={"n": 1}),
        ],
        usage=SimpleNamespace(
            **{
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": None,
                "cache_creation_input_tokens": None,
                **usage,
            }
        ),
    )


def test_response_maps_to_chat_completion_with_cached_tokens() -> None:
    completion = to_chat_completion(
        _fake_response(cache_read_input_tokens=4000, cache_creation_input_tokens=100), "claude-haiku-4-5"
    )
    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "I'll pass."
    assert choice.message.tool_calls is not None
    call = choice.message.tool_calls[0]
    assert (call.id, call.function.name, json.loads(call.function.arguments)) == ("toolu_1", "pass_priority", {"n": 1})
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 10 + 4000 + 100
    assert completion.usage.completion_tokens == 5
    assert completion.usage.prompt_tokens_details is not None
    assert completion.usage.prompt_tokens_details.cached_tokens == 4000
    # The pilot's trace log dumps the whole response.
    assert completion.model_dump()["usage"]["prompt_tokens"] == 4110


def test_stop_reason_max_tokens_maps_to_length() -> None:
    resp = _fake_response()
    resp.stop_reason = "max_tokens"
    resp.content = [SimpleNamespace(type="text", text="trunc")]
    completion = to_chat_completion(resp, "m")
    assert completion.choices[0].finish_reason == "length"
    assert completion.choices[0].message.tool_calls is None
    assert completion.usage is not None
    assert completion.usage.prompt_tokens_details is not None
    assert completion.usage.prompt_tokens_details.cached_tokens == 0


def test_native_error_is_an_openai_error() -> None:
    with pytest.raises(OpenAIError):
        raise AnthropicNativeError("boom")
