"""Golden prompt test: offer_mana_sources lets the player choose what pays."""

from tests.golden_helpers import (
    DECK_CRYSTAL_VEIN_FELLWAR,
    DECK_FILLER,
    run_golden_scenario,
)
from tests.golden_test_identities import golden_test


@golden_test("crystal_vein_manual_mana")
def test_crystal_vein_manual_mana(
    xmage_server,
    tmp_path,
    project_root,
    bridge_session,
    opponent_session,
    spectator_process,
):
    """Sacrifice Crystal Vein for {C}{C} to cast Fellwar Stone, keeping the Mountain up.

    Left to the auto-tapper, the Mountain is tapped first and the Vein's
    "1 or sacrifice for 2" question comes too late to keep it. With
    offer_mana_sources on, the Vein is a priority choice: activating it asks
    which ability, the sacrifice floats {C}{C}, and casting the Stone spends the
    pool before any land is touched. The tap, the ability and the cast are
    chained choose_actions: a pass_priority in between would leave the main
    phase and empty the pool.

    Opponent's 7 Mountains = p3-p9. TestPlayer's hand (alphabetical):
    Crystal Vein=p10, Fellwar Stone=p11, Mountain=p12..p16.
    """
    server, port = xmage_server
    # The toggle is handler config that survives join_table's fresh handler,
    # so a host sets it before the seat is even at a table.
    assert bridge_session.session is not None
    bridge_session.session.call_tool("offer_mana_sources", {"enabled": True})
    try:
        run_golden_scenario(
            server=server,
            port=port,
            project_root=project_root,
            game_dir=tmp_path / "crystal_vein_manual_mana",
            deck_a=DECK_CRYSTAL_VEIN_FELLWAR,
            deck_b=DECK_FILLER,
            script_a=[
                # Choose TestPlayer as starting player, keep hand.
                {"name": "pass_priority", "arguments": {}},
                {"name": "choose_action", "arguments": {"choice": "0"}},
                {"name": "pass_priority", "arguments": {}},
                {"name": "choose_action", "arguments": {"choice": "no"}},
                # T1: Play Mountain (p12).
                {"name": "pass_priority", "arguments": {}},
                {"name": "choose_action", "arguments": {"choice": "p12"}},
                # T1: nothing castable; the Mountain alone is offered as a mana source. Pass.
                {"name": "pass_priority", "arguments": {}},
                {"name": "choose_action", "arguments": {"choice": "no"}},
                # T2: Play Crystal Vein (p10); the next decision lists it and the
                # Mountain as mana sources beside the castable Stone.
                {"name": "pass_priority", "arguments": {}},
                {"name": "choose_action", "arguments": {"choice": "p10"}, "golden_blunder": True},
                # Tap the Vein for mana before casting: the engine asks which ability.
                {"name": "choose_action", "arguments": {"choice": "p10"}},
                # GAME_CHOOSE_ABILITY: index 1 = "{T}, Sacrifice Crystal Vein: Add {C}{C}."
                {"name": "choose_action", "arguments": {"choice": "1"}},
                # Cast Fellwar Stone (p11): {C}{C} in the pool pays {2}, Mountain untouched.
                {"name": "choose_action", "arguments": {"choice": "p11"}},
                {"name": "get_game_state", "arguments": {}},
            ],
            golden_name="crystal_vein_manual_mana",
            bridge_a=bridge_session,
            bridge_b=opponent_session,
            spectator=spectator_process,
        )
    finally:
        # The toggle is sticky per bridge and the bridge is shared by the suite.
        if bridge_session.session is not None:
            bridge_session.session.call_tool("offer_mana_sources", {"enabled": False})
