package mage.client.bridge.tools;

import mage.client.bridge.BridgeCallbackHandler;

import java.util.List;
import java.util.Map;

import static mage.client.bridge.tools.McpToolRegistry.example;
import static mage.client.bridge.tools.McpToolRegistry.json;

/**
 * Priority choices normally leave out permanents whose only plays are mana
 * abilities: the bridge auto-taps when a cost is paid, and listing every land
 * at every window would bury the real options. A human-driven seat wants them
 * (a player taps their own Crystal Vein for {C}{C} and keeps the Forest up);
 * the floating mana is spent first by the next payment.
 */
public class OfferManaSourcesTool {

    public static class Result {
        @ResultField(description = "Whether mana-only permanents are now listed among priority choices")
        public Boolean enabled;
    }

    @Tool(
        name = "offer_mana_sources",
        description = "Include permanents whose only plays are mana abilities in priority choices (action=\"mana\"); "
            + "choosing one activates the ability and the mana floats. Off by default; sticky for this bridge."
    )
    public static Result execute(
            BridgeCallbackHandler handler,
            @Param(description = "true to list mana sources, false to hide them again") Boolean enabled) {
        boolean on = Boolean.TRUE.equals(enabled);
        handler.setOfferManaSources(on);
        var result = new Result();
        result.enabled = on;
        return result;
    }

    public static List<Map<String, Object>> examples() {
        return List.of(example("Enable", json("enabled", true)));
    }
}
