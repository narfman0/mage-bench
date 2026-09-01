"""Tests for pilot recovery helpers (timeout and stall game-over detection)."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolResult, TextContent

from magebench.pilot.pilot_recovery import _handle_timeout, _recover_from_stall
from magebench.pilot.pilot_state import PilotLoopState

logger = MagicMock()


def _mock_tool_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)])


def _make_session(pass_priority_text: str) -> MagicMock:
    session = MagicMock()
    session.call_tool = AsyncMock(
        return_value=_mock_tool_result(pass_priority_text),
    )
    return session


# ---------------------------------------------------------------------------
# _handle_timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_timeout_detects_game_over():
    session = _make_session(json.dumps({"game_over": True}))
    state = PilotLoopState(history=[])
    game_log = MagicMock()

    result = await _handle_timeout(
        session,
        state,
        game_log,
        logger=logger,
        llm_request_timeout_secs=30,
        max_consecutive_timeouts=3,
        retries_before_auto_pass=0,
    )

    assert result is True
    game_log.emit.assert_any_call("auto_pilot_mode", reason="game_over")


@pytest.mark.asyncio
async def test_handle_timeout_detects_player_dead():
    session = _make_session(json.dumps({"player_dead": True}))
    state = PilotLoopState(history=[])
    game_log = MagicMock()

    result = await _handle_timeout(
        session,
        state,
        game_log,
        logger=logger,
        llm_request_timeout_secs=30,
        max_consecutive_timeouts=3,
        retries_before_auto_pass=0,
    )

    assert result is True
    game_log.emit.assert_any_call("auto_pilot_mode", reason="player_dead")


@pytest.mark.asyncio
async def test_handle_timeout_returns_false_on_normal_result():
    session = _make_session(json.dumps({"action_pending": False}))
    state = PilotLoopState(history=[{"role": "user", "content": "stale context"}])

    result = await _handle_timeout(
        session,
        state,
        None,
        logger=logger,
        llm_request_timeout_secs=30,
        max_consecutive_timeouts=3,
        retries_before_auto_pass=0,
    )

    assert result is False
    # Context must be reset even after a single timeout so the next LLM call
    # doesn't see stale game state from before the timeout recovery.
    assert len(state.history) == 1
    assert state.history[0]["content"] == "Continue playing. Call pass_priority."


@pytest.mark.asyncio
async def test_handle_timeout_detects_stop_reason_game_over():
    session = _make_session(json.dumps({"stop_reason": "game_over", "action_pending": False}))
    state = PilotLoopState(history=[])
    game_log = MagicMock()

    result = await _handle_timeout(
        session,
        state,
        game_log,
        logger=logger,
        llm_request_timeout_secs=30,
        max_consecutive_timeouts=3,
        retries_before_auto_pass=0,
    )

    assert result is True
    game_log.emit.assert_any_call("auto_pilot_mode", reason="game_over")


@pytest.mark.asyncio
async def test_handle_timeout_returns_false_on_tool_error():
    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=RuntimeError("bridge died"))
    state = PilotLoopState(history=[{"role": "user", "content": "stale context"}])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(asyncio, "sleep", AsyncMock())
        result = await _handle_timeout(
            session,
            state,
            None,
            logger=logger,
            llm_request_timeout_secs=30,
            max_consecutive_timeouts=3,
            retries_before_auto_pass=0,
        )

    assert result is False
    assert len(state.history) == 1
    assert state.history[0]["content"] == "Continue playing. Call pass_priority."


@pytest.mark.asyncio
async def test_handle_timeout_full_board_reset_after_max_consecutive():
    """After max_consecutive_timeouts, reset includes board context."""
    session = _make_session(json.dumps({"action_pending": False}))
    state = PilotLoopState(history=[{"role": "user", "content": "stale context"}])
    state.consecutive_timeouts = 2  # will become 3 inside _handle_timeout
    state.board_tracker.cursor = 42
    state.last_board = [{"zone": "battlefield"}]
    game_log = MagicMock()

    result = await _handle_timeout(
        session,
        state,
        game_log,
        logger=logger,
        llm_request_timeout_secs=30,
        max_consecutive_timeouts=3,
        retries_before_auto_pass=0,
    )

    assert result is False
    assert state.consecutive_timeouts == 0
    assert len(state.history) == 1
    # Full board reset clears board tracker and last_board
    assert state.board_tracker.cursor is None
    assert state.last_board is None
    game_log.emit.assert_any_call("context_reset", reason="repeated_timeouts")


# ---------------------------------------------------------------------------
# _recover_from_stall
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recover_from_stall_detects_game_over():
    call_count = 0

    async def fake_call_tool(name, _args):
        nonlocal call_count
        call_count += 1
        if name == "send_chat_message":
            return _mock_tool_result('{"success": true}')
        return _mock_tool_result(json.dumps({"game_over": True}))

    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=fake_call_tool)
    state = PilotLoopState(history=[])
    state.turns_without_progress = 5
    game_log = MagicMock()

    result = await _recover_from_stall(session, state, game_log, {"pass_priority"}, logger=logger)

    assert result is True
    game_log.emit.assert_any_call("auto_pilot_mode", reason="game_over")
    # Should not reset context when game ended
    assert state.history == []


@pytest.mark.asyncio
async def test_recover_from_stall_detects_player_dead():
    async def fake_call_tool(name, _args):
        if name == "send_chat_message":
            return _mock_tool_result('{"success": true}')
        return _mock_tool_result(json.dumps({"player_dead": True}))

    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=fake_call_tool)
    state = PilotLoopState(history=[])
    state.turns_without_progress = 5
    game_log = MagicMock()

    result = await _recover_from_stall(session, state, game_log, {"pass_priority"}, logger=logger)

    assert result is True
    game_log.emit.assert_any_call("auto_pilot_mode", reason="player_dead")


@pytest.mark.asyncio
async def test_recover_from_stall_returns_false_on_normal_result():
    async def fake_call_tool(name, _args):
        if name == "send_chat_message":
            return _mock_tool_result('{"success": true}')
        return _mock_tool_result(json.dumps({"action_pending": False}))

    session = MagicMock()
    session.call_tool = AsyncMock(side_effect=fake_call_tool)
    state = PilotLoopState(history=[])
    state.turns_without_progress = 5

    result = await _recover_from_stall(session, state, None, {"pass_priority"}, logger=logger)

    assert result is False
    # Should reset context when game continues
    assert len(state.history) == 1
    assert state.history[0]["content"] == "A new turn has started. Call pass_priority to continue."


@pytest.mark.asyncio
async def test_handle_timeout_retries_before_passing():
    """The first timeout must NOT pass on the player's behalf.

    Regression test for the bug seen in game_20260831_224345, where every single LLM
    timeout immediately fired pass_priority -- costing Kimi K3 three main phases, two
    attack steps and three blocking steps across turns 7-9. A transient timeout should
    cost latency, not a move.
    """
    session = _make_session(json.dumps({"action_pending": False}))
    state = PilotLoopState(history=[{"role": "user", "content": "stale context"}])
    game_log = MagicMock()

    result = await _handle_timeout(
        session,
        state,
        game_log,
        logger=logger,
        llm_request_timeout_secs=30,
        max_consecutive_timeouts=3,
        retries_before_auto_pass=2,
    )

    assert result is False
    # Nothing was sent to the game at all -- the pending decision is untouched.
    session.call_tool.assert_not_called()
    game_log.emit.assert_any_call("timeout_retry", attempt=1, of=3)
    assert state.consecutive_timeouts == 1


@pytest.mark.asyncio
async def test_handle_timeout_auto_passes_once_retry_budget_is_spent():
    """Once retries are exhausted the harness does pass -- and records that it did."""
    session = _make_session(json.dumps({"action_pending": False}))
    state = PilotLoopState(history=[{"role": "user", "content": "stale context"}])
    state.consecutive_timeouts = 2  # becomes 3, past a budget of 2
    game_log = MagicMock()

    result = await _handle_timeout(
        session,
        state,
        game_log,
        logger=logger,
        llm_request_timeout_secs=30,
        max_consecutive_timeouts=99,
        retries_before_auto_pass=2,
    )

    assert result is False
    # A move WAS made for the player, and it is recorded as the harness's, not the model's.
    assert session.call_tool.await_count >= 1
    emitted = [c.args[0] for c in game_log.emit.call_args_list if c.args]
    assert "harness_auto_pass" in emitted
