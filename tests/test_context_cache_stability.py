"""The rendered prompt prefix must stay byte-stable between calls so prompt caching hits."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from magebench.pilot.pilot import _build_loop_messages, _mark_tail_cache_breakpoint
from magebench.pilot.pilot_rendering import (
    CONTEXT_CHUNK,
    CONTEXT_RECENT_COUNT,
    context_recent_start,
    render_context,
)
from magebench.pilot.pilot_state import PilotLoopState

SYSTEM_PROMPT = "You are a pilot."
CC = {"type": "ephemeral"}


def _pair(i: int) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": f"c{i}", "type": "function", "function": {"name": "pass_priority", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps({"i": i, "action_pending": False})},
    ]


def _history(n_pairs: int) -> list[dict]:
    history: list[dict] = [{"role": "user", "content": "Begin."}]
    for i in range(n_pairs):
        history.extend(_pair(i))
    return history


def test_recent_start_advances_only_in_chunks() -> None:
    assert context_recent_start(CONTEXT_RECENT_COUNT) == 0
    assert context_recent_start(CONTEXT_RECENT_COUNT + 1) == 0
    assert context_recent_start(CONTEXT_RECENT_COUNT + CONTEXT_CHUNK - 1) == 0
    assert context_recent_start(CONTEXT_RECENT_COUNT + CONTEXT_CHUNK) == CONTEXT_CHUNK
    assert context_recent_start(CONTEXT_RECENT_COUNT + 2 * CONTEXT_CHUNK + 3) == 2 * CONTEXT_CHUNK


def test_appending_within_a_chunk_keeps_the_rendered_prefix_identical() -> None:
    history = _history(40)  # 81 messages: past the first chunk boundary
    before = render_context(history, SYSTEM_PROMPT, "S", CC)
    history.extend(_pair(99))
    after = render_context(history, SYSTEM_PROMPT, "S", CC)
    assert after[: len(before)] == before
    assert len(after) == len(before) + 2


@pytest.mark.asyncio
async def test_state_summary_refreshes_only_at_chunk_boundaries() -> None:
    history = _history(30)  # 61 messages -> window start 20
    state = PilotLoopState(history=history)
    session = MagicMock()
    with patch("magebench.pilot.pilot._fetch_state_summary", new_callable=AsyncMock, return_value="S") as fetch:
        first = await _build_loop_messages(state, session, SYSTEM_PROMPT, CC)
        assert fetch.await_count == 1
        # Many calls inside the same chunk: no refetch, and the prefix is unchanged.
        for i in range(8):
            state.history.extend(_pair(100 + i))
            nxt = await _build_loop_messages(state, session, SYSTEM_PROMPT, CC)
            assert nxt[: len(first)] == first
        assert fetch.await_count == 1
        # Cross the next boundary: exactly one refetch.
        while context_recent_start(len(state.history)) == 20:
            state.history.extend(_pair(200 + len(state.history)))
        await _build_loop_messages(state, session, SYSTEM_PROMPT, CC)
        assert fetch.await_count == 2


@pytest.mark.asyncio
async def test_tail_breakpoint_marks_last_message_and_prefix_marker() -> None:
    state = PilotLoopState(history=_history(30))
    with patch("magebench.pilot.pilot._fetch_state_summary", new_callable=AsyncMock, return_value="S"):
        messages = await _build_loop_messages(state, MagicMock(), SYSTEM_PROMPT, CC)
    _mark_tail_cache_breakpoint(messages, state, CC)

    def marked(msg: dict) -> bool:
        content = msg.get("content")
        return isinstance(content, list) and any(b.get("cache_control") == CC for b in content if isinstance(b, dict))

    assert marked(messages[0])  # system
    assert state.cache_breakpoint_idx is not None
    assert marked(messages[state.cache_breakpoint_idx])  # stable-prefix marker
    assert marked(messages[-1])  # true tail
    assert sum(marked(m) for m in messages) == 3, "must stay within the 4-breakpoint limit"


def test_short_history_tail_breakpoint_marks_only_system_and_tail() -> None:
    state = PilotLoopState(history=_history(3))
    messages = render_context(state.history, SYSTEM_PROMPT, "", CC)
    state.cache_breakpoint_idx = len(messages) - 1
    _mark_tail_cache_breakpoint(messages, state, CC)
    tail = messages[-1]["content"]
    assert isinstance(tail, list) and tail[-1]["cache_control"] == CC
