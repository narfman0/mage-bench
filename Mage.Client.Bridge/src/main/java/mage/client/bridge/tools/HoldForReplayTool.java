package mage.client.bridge.tools;

import mage.client.bridge.BridgeCallbackHandler;

import java.util.List;
import java.util.Map;

import static mage.client.bridge.tools.McpToolRegistry.example;
import static mage.client.bridge.tools.McpToolRegistry.json;

/**
 * Resume support: while the server's replay feeder answers a game's recorded
 * decisions, this bridge must not act on the queries it sees. Hold through
 * the last recorded game seq; the first query past it is live again.
 */
public class HoldForReplayTool {

    public static class Result {
        @ResultField(description = "The game seq this bridge now holds through (0 = not holding)")
        public Long through_game_seq;
    }

    @Tool(
        name = "hold_for_replay",
        description = "Ignore game queries with game_seq at or below the given value (the host's replay feeder answers them); "
            + "resume acting on the first query past it. 0 clears the hold."
    )
    public static Result execute(
            BridgeCallbackHandler handler,
            @Param(description = "Last recorded game_seq to hold through; 0 clears") Long through_game_seq) {
        long n = through_game_seq == null ? 0L : Math.max(0L, through_game_seq);
        handler.holdForReplay(n);
        var result = new Result();
        result.through_game_seq = n;
        return result;
    }

    public static List<Map<String, Object>> examples() {
        return List.of(example("Hold through seq 523", json("through_game_seq", 523)));
    }
}
