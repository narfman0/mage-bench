"""Golden: a seeded game resumes from its record (ReplayFeederCollector).

Three games on the same two bridges, same seed, same decks:

- game 0: the whole script, uninterrupted — the reference;
- game 1: the first half of the script, then a concede — its
  ``server_game_events.jsonl`` is the record a host would keep;
- game 2: created with ``replayFrom`` game 1's record while both bridges hold
  through the last recorded seq, then the second half of the script live.

The resumed game must feed every recorded decision without divergence and its
decisions and game log must equal game 0's: the record replays to the same
game, and live play continues from where the record stops.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from tests.golden_helpers import (
    DECK_BOLT_AND_BURN,
    DECK_FILLER,
    SpectatorProcess,
    _run_opponent_autopass,
    _run_replay_on_bridge,
)
from tests.golden_test_identities import golden_test

SEED = 3000007

MULLIGAN = [
    {"name": "pass_priority", "arguments": {}},
    {"name": "choose_action", "arguments": {"choice": "0"}},
    {"name": "pass_priority", "arguments": {}},
    {"name": "choose_action", "arguments": {"choice": "no"}},
]
# T1: Mountain (p14), Memnite (p13), decline to cast Lightning Bolt.
FIRST_HALF = [
    *MULLIGAN,
    {"name": "pass_priority", "arguments": {}},
    {"name": "choose_action", "arguments": {"choice": "p14"}},
    {"name": "pass_priority", "arguments": {}},
    {"name": "choose_action", "arguments": {"choice": "p13"}},
    {"name": "pass_priority", "arguments": {}},
    {"name": "choose_action", "arguments": {"choice": "no"}},
]
# T2: Badlands (p10), skip attacking, Bolt (p11) the opponent.
SECOND_HALF = [
    {"name": "pass_priority", "arguments": {}},
    {"name": "choose_action", "arguments": {"choice": "p10"}},
    {"name": "pass_priority", "arguments": {}},
    {"name": "choose_action", "arguments": {"choice": "no"}},
    {"name": "pass_priority", "arguments": {}},
    {"name": "choose_action", "arguments": {"choice": "p11"}},
    {"name": "choose_action", "arguments": {"choice": "1"}},
    {"name": "pass_priority", "arguments": {}},
    {"name": "get_game_state", "arguments": {}},
]


def _decisions(log: Path) -> list[tuple]:
    """(player, query type, response) per recorded decision — the replay key."""
    out = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("type") == "decision":
            r = e.get("response") or {}
            out.append(
                (e.get("player"), e.get("query_type"), r.get("type"), r.get("id"), r.get("name"), r.get("value"))
            )
    return out


def _last_decision_seq(log: Path) -> int:
    seqs = [
        int(json.loads(line)["seq"])
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("type") == "decision"
    ]
    return max(seqs) if seqs else 0


def _actions(log: Path) -> list[str]:
    """Game-action text with the per-run [abc] log refs stripped."""
    import re

    out = []
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("type") == "game_action":
            out.append(re.sub(r"\s*\[[0-9a-f]{3}\]", "", str(e.get("message") or "")))
    return out


def _wait_replay_done(game_dir: Path, timeout: float = 60.0) -> dict:
    status = game_dir / "replay_status.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if status.exists():
            data = json.loads(status.read_text(encoding="utf-8"))
            if data.get("done"):
                return data
        time.sleep(0.2)
    raise AssertionError(
        f"replay feeder did not finish within {timeout}s: {status.read_text() if status.exists() else 'no status'}"
    )


def _play(
    spectator: SpectatorProcess,
    project_root: Path,
    game_dir: Path,
    session_a,
    session_b,
    name_a: str,
    name_b: str,
    script_a: list[dict],
    *,
    replay_from: Path | None = None,
    hold_through: int = 0,
    concede: bool = True,
) -> None:
    """One seeded game: create the table, join both bridges, run the script."""
    game_dir.mkdir(parents=True, exist_ok=True)
    players_config = {
        "players": [
            {"type": "replay", "name": name_a, "deck": DECK_BOLT_AND_BURN},
            {"type": "replay", "name": name_b, "deck": DECK_FILLER},
        ],
        "gameType": "Two Player Duel",
        "deckType": "Constructed - Legacy",
    }
    spectator.start_game(game_dir, players_config, name_a, game_seed=SEED, replay_from=replay_from)
    table_id = spectator.wait_for_ready(game_dir)
    for session in (session_a, session_b):
        session.call_tool("hold_for_replay", {"through_game_seq": hold_through})

    errors: list[Exception] = []

    def _join(session, deck: str) -> None:
        try:
            session.call_tool("join_table", {"deck_path": str(project_root / deck), "table_id": table_id})
        except Exception as exc:  # noqa: BLE001 — surfaced below
            errors.append(exc)

    threads = [
        threading.Thread(target=_join, args=(session_a, DECK_BOLT_AND_BURN)),
        threading.Thread(target=_join, args=(session_b, DECK_FILLER)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors, f"bridge join failed: {errors}"
    spectator.wait_for_watching(game_dir)
    # Both bridges start acting right away, as a host's live loops would: a
    # held bridge's pass_priority must block until the feeder is done and the
    # first live query arrives, never answer a recorded one.

    b_errors: list[Exception] = []

    def _b() -> None:
        try:
            _run_opponent_autopass(session_b)
        except Exception as exc:  # noqa: BLE001 — benign once A concedes
            b_errors.append(exc)

    t_b = threading.Thread(target=_b)
    t_b.start()
    try:
        _run_replay_on_bridge(
            session_a,
            script_a,
            game_dir,
            name_a,
            skip_postscript=True,
            write_log=False,
            should_concede=False,
        )
    finally:
        if concede:
            session_a.call_tool("concede", {})
    t_b.join(timeout=120)
    spectator.wait_for_game_end(game_dir)
    if replay_from is not None:
        status = _wait_replay_done(game_dir)
        assert not status.get("divergence"), f"replay diverged: {status}"
        assert status["fed"] == status["total"], status


@golden_test("resume_from_record")
def test_resume_from_record(
    xmage_server,
    tmp_path,
    project_root,
    bridge_session,
    opponent_session,
    spectator_process,
):
    session_a, session_b = bridge_session.session, opponent_session.session
    assert session_a is not None and session_b is not None
    name_a, name_b = bridge_session.username, opponent_session.username
    common = dict(
        spectator=spectator_process,
        project_root=project_root,
        session_a=session_a,
        session_b=session_b,
        name_a=name_a,
        name_b=name_b,
    )

    # Game 0: the reference, played through in one sitting.
    full = tmp_path / "full"
    _play(game_dir=full, script_a=[*FIRST_HALF, *SECOND_HALF], **common)
    full_log = full / "server_game_events.jsonl"

    # Game 1: the first half, then the player "leaves" (concede): the record.
    part = tmp_path / "part"
    _play(game_dir=part, script_a=FIRST_HALF, **common)
    record = part / "server_game_events.jsonl"
    recorded = _decisions(record)
    hold = _last_decision_seq(record)
    assert recorded and hold > 0
    # The concede is not a decision; the record ends where the script stopped.
    assert recorded == _decisions(full_log)[: len(recorded)]

    # Game 2: resumed from the record, then the second half live.
    resumed = tmp_path / "resumed"
    offsets = {"A": bridge_session.capture_log_offsets(), "B": opponent_session.capture_log_offsets()}
    _play(game_dir=resumed, script_a=SECOND_HALF, replay_from=record, hold_through=hold, **common)
    # The hold did its job on both bridges: it was set before join_table
    # (which replaces the bridge's handler), held every recorded query, and
    # released on the first live one. Without this the bridges' own answers
    # race the feeder's, which is timing-dependent and can pass by luck.
    for who, offs in offsets.items():
        text = offs.bridge_log_path.read_bytes()[offs.bridge_log_offset :].decode("utf-8", errors="replace")
        assert "Replay hold set through game_seq" in text, f"bridge {who}: hold never set"
        assert text.count("] Holding ") > 0, f"bridge {who}: never held a callback during the replay"
        assert "Replay hold released at game_seq" in text, f"bridge {who}: hold never released"
    resumed_log = resumed / "server_game_events.jsonl"
    status = json.loads((resumed / "replay_status.json").read_text(encoding="utf-8"))
    assert status.get("divergence") is None, status
    assert (status["total"], status["fed"], status["remaining"], status["done"]) == (
        len(recorded),
        len(recorded),
        0,
        True,
    )

    # Same game: every decision (the fed ones and the live ones) and every
    # game action, in order, as if never interrupted.
    assert _decisions(resumed_log) == _decisions(full_log)
    assert _actions(resumed_log) == _actions(full_log)
    # A resumed game is itself resumable: its record is complete.
    assert _decisions(resumed_log)[: len(recorded)] == recorded
