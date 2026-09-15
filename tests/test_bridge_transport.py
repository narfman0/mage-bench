"""Tests for shared bridge launch argument assembly."""

import inspect
from pathlib import Path

from magebench.pilot import bridge_transport
from magebench.pilot.bridge_transport import bridge_launch_command, connect_bridge_http


def test_build_bridge_launch_args_for_sleepwalker() -> None:
    launch_args = bridge_transport.build_bridge_launch_args(
        server="example.org",
        port=17171,
        username="Sleeper",
        deck_path=Path("/tmp/decks/sleeper.dck"),
        heap_size_mb=512,
    )

    assert launch_args.jvm_args == (
        "--add-opens=java.base/java.io=ALL-UNNAMED -Xmx512m -Dxmage.bridge.server=example.org -Dxmage.bridge.port=17171"
    )
    assert launch_args.mvn_args == [
        "-q",
        "-Dxmage.bridge.username=Sleeper",
        "-Dxmage.bridge.deck=/tmp/decks/sleeper.dck",
        "exec:java",
    ]


def test_build_bridge_launch_args_for_pilot_with_logs() -> None:
    game_dir = Path("/tmp/game-001")

    launch_args = bridge_transport.build_bridge_launch_args(
        server="localhost",
        port=17171,
        username="Pilot",
        deck_path=Path("/tmp/decks/pilot.dck"),
        heap_size_mb=512,
        error_log_path=game_dir / "Pilot_errors.log",
        bridge_log_path=game_dir / "Pilot_bridge.jsonl",
        max_interactions_per_turn=9,
    )

    assert launch_args.jvm_args == (
        "--add-opens=java.base/java.io=ALL-UNNAMED -Xmx512m -Dxmage.bridge.server=localhost -Dxmage.bridge.port=17171"
    )
    assert launch_args.mvn_args == [
        "-q",
        "-Dxmage.bridge.username=Pilot",
        "-Dxmage.bridge.deck=/tmp/decks/pilot.dck",
        "-Dxmage.bridge.errorlog=/tmp/game-001/Pilot_errors.log",
        "-Dxmage.bridge.bridgelog=/tmp/game-001/Pilot_bridge.jsonl",
        "-Dxmage.bridge.maxInteractionsPerTurn=9",
        "exec:java",
    ]


def test_build_bridge_launch_args_for_replay_with_table_id() -> None:
    launch_args = bridge_transport.build_bridge_launch_args(
        server="localhost",
        port=17171,
        username="Replay",
        table_id="table-123",
        error_log_path=Path("/tmp/game-002/Replay_errors.log"),
    )

    assert launch_args.jvm_args == (
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "-Dxmage.bridge.server=localhost "
        "-Dxmage.bridge.port=17171 "
        "-Dxmage.bridge.tableId=table-123"
    )
    assert launch_args.mvn_args == [
        "-q",
        "-Dxmage.bridge.username=Replay",
        "-Dxmage.bridge.errorlog=/tmp/game-002/Replay_errors.log",
        "exec:java",
    ]


def test_build_bridge_launch_args_adds_darwin_ui_flag(monkeypatch) -> None:
    monkeypatch.setattr(bridge_transport.sys, "platform", "darwin")

    launch_args = bridge_transport.build_bridge_launch_args(
        server="localhost",
        port=17171,
        username="MacPilot",
    )

    assert launch_args.jvm_args == (
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "-Dxmage.bridge.server=localhost "
        "-Dxmage.bridge.port=17171 "
        "-Dapple.awt.UIElement=true"
    )


def test_bridge_launch_command_prefers_prebuilt_classpath():
    mvn_args = ["-q", "-Dxmage.bridge.username=Bot", "-Dxmage.bridge.deck=/d.dck", "exec:java"]
    jvm = "--add-opens=java.base/java.io=ALL-UNNAMED -Xmx512m -Dxmage.bridge.server=h -Dxmage.bridge.mcpPort=1"
    assert bridge_launch_command(mvn_args, jvm, None) == ["mvn", *mvn_args]
    cmd = bridge_launch_command(mvn_args, jvm, "/a.jar:/b/classes")
    assert cmd[0] == "java" and cmd[-1] == "mage.client.bridge.BridgeClient"
    assert "-cp" in cmd and cmd[cmd.index("-cp") + 1] == "/a.jar:/b/classes"
    assert "-Dxmage.bridge.username=Bot" in cmd and "-Dxmage.bridge.mcpPort=1" in cmd and "-Xmx512m" in cmd
    assert "exec:java" not in cmd and "-q" not in cmd


def test_connect_bridge_http_is_an_async_context_manager():
    cm = connect_bridge_http("http://127.0.0.1:1/mcp")
    assert hasattr(cm, "__aenter__") and hasattr(cm, "__aexit__")
    assert inspect.isasyncgenfunction(connect_bridge_http.__wrapped__)
