package mage.client.bridge.processor;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.apache.log4j.Logger;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Appends every change of the projected game state to a JSONL file, so a
 * host can follow the board continuously instead of only at this player's
 * priority stops (the MCP tools are pull-only). One line per distinct
 * snapshot, in the exact shape get_game_state publishes, plus {@code moves}:
 * the cards that changed zone since the previous line, diffed here where
 * both states are at hand. A host that animates "card X went from hand to
 * the stack" needs that as an event, not two boards to compare.
 *
 * Enabled by {@code -Dxmage.bridge.stateLog=<path>}; off otherwise.
 */
final class BridgeStateLog {

    private static final Logger logger = Logger.getLogger(BridgeStateLog.class);
    private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().create();
    private static final String[] ZONES = {"hand", "battlefield", "graveyard", "exile", "command_zone"};

    /** Where a card was last seen: zone, controller, name. */
    private record Seen(String zone, String player, String name) {
    }

    private final BufferedWriter out;
    private final String username;
    private Long lastSnapshotId;
    private Map<String, Seen> lastSeen;

    private BridgeStateLog(BufferedWriter out, String username) {
        this.out = out;
        this.username = username;
    }

    static BridgeStateLog fromSystemProperties(String username) {
        String path = System.getProperty("xmage.bridge.stateLog");
        if (path == null || path.isBlank()) {
            return new BridgeStateLog(null, username);
        }
        try {
            Path p = Path.of(path);
            if (p.getParent() != null) {
                Files.createDirectories(p.getParent());
            }
            BufferedWriter w = Files.newBufferedWriter(p, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            logger.info("[" + username + "] state log: " + p);
            return new BridgeStateLog(w, username);
        } catch (IOException e) {
            logger.warn("[" + username + "] state log unavailable at " + path + ": " + e.getMessage());
            return new BridgeStateLog(null, username);
        }
    }

    boolean enabled() {
        return out != null;
    }

    /** Record a projected state if it differs from the last one recorded. */
    void record(BridgePublishedGameState state) {
        if (out == null || state == null || !state.available()) {
            return;
        }
        if (lastSnapshotId != null && lastSnapshotId.equals(state.snapshotId())) {
            return;
        }
        lastSnapshotId = state.snapshotId();
        Map<String, Seen> seen = index(state);
        List<Map<String, Object>> moves = lastSeen == null ? List.of() : diff(lastSeen, seen);
        lastSeen = seen;

        Map<String, Object> line = new LinkedHashMap<>();
        line.put("type", "board");
        line.put("game_seq", state.gameSeq());
        line.put("snapshot_id", state.snapshotId());
        line.put("turn", state.turn());
        line.put("phase", state.phase());
        line.put("step", state.step());
        line.put("active_player", state.activePlayer());
        line.put("priority_player", state.priorityPlayer());
        line.put("players", state.players());
        line.put("stack", state.stack());
        line.put("combat", state.combat());
        line.put("moves", moves);
        try {
            out.write(GSON.toJson(line));
            out.newLine();
            out.flush();
        } catch (IOException e) {
            logger.warn("[" + username + "] state log write failed: " + e.getMessage());
        }
    }

    /** Every card with an id, by id: which zone and whose. Spells on the
     *  stack keep their card id, so a cast reads hand → stack → battlefield;
     *  abilities on the stack belong to a source card and aren't cards. */
    private static Map<String, Seen> index(BridgePublishedGameState state) {
        Map<String, Seen> seen = new LinkedHashMap<>();
        for (Map<String, Object> player : state.players()) {
            String name = String.valueOf(player.get("name"));
            for (String zone : ZONES) {
                Object cards = player.get(zone);
                if (!(cards instanceof List<?> list)) {
                    continue;
                }
                for (Object o : list) {
                    if (o instanceof Map<?, ?> card && card.get("id") != null) {
                        seen.put(String.valueOf(card.get("id")),
                            new Seen(zone, name, String.valueOf(card.get("name"))));
                    }
                }
            }
        }
        for (Map<String, Object> item : state.stack()) {
            if (item.get("id") == null || item.get("source_card") != null) {
                continue;
            }
            seen.put(String.valueOf(item.get("id")),
                new Seen("stack", item.get("owner") != null ? String.valueOf(item.get("owner")) : null,
                    String.valueOf(item.get("name"))));
        }
        return seen;
    }

    /** Cards that changed zone or controller. {@code from}/{@code to} are null
     *  for a card not seen before / no longer visible (library, a hidden
     *  hand, a token that ceased to exist). */
    private static List<Map<String, Object>> diff(Map<String, Seen> before, Map<String, Seen> after) {
        List<Map<String, Object>> moves = new ArrayList<>();
        for (Map.Entry<String, Seen> e : after.entrySet()) {
            Seen prev = before.get(e.getKey());
            Seen now = e.getValue();
            if (prev == null) {
                moves.add(move(e.getKey(), now.name(), null, null, now.zone(), now.player()));
            } else if (!prev.zone().equals(now.zone()) || !java.util.Objects.equals(prev.player(), now.player())) {
                moves.add(move(e.getKey(), now.name(), prev.zone(), prev.player(), now.zone(), now.player()));
            }
        }
        for (Map.Entry<String, Seen> e : before.entrySet()) {
            if (!after.containsKey(e.getKey())) {
                Seen prev = e.getValue();
                moves.add(move(e.getKey(), prev.name(), prev.zone(), prev.player(), null, null));
            }
        }
        return pairRebirths(moves);
    }

    /** A spell on the stack and the permanent it resolves into carry different
     *  stable ids, so one resolution reads as "left the stack" + "appeared on
     *  the battlefield". Pair a departure with an arrival of the same name in
     *  the same diff into one move (id = the arrival's, the card as it is now). */
    private static List<Map<String, Object>> pairRebirths(List<Map<String, Object>> moves) {
        List<Map<String, Object>> gone = new ArrayList<>();
        List<Map<String, Object>> born = new ArrayList<>();
        List<Map<String, Object>> rest = new ArrayList<>();
        for (Map<String, Object> m : moves) {
            if (m.get("to") == null) {
                gone.add(m);
            } else if (m.get("from") == null) {
                born.add(m);
            } else {
                rest.add(m);
            }
        }
        for (Map<String, Object> b : born) {
            Map<String, Object> match = null;
            for (Map<String, Object> g : gone) {
                if (java.util.Objects.equals(g.get("name"), b.get("name"))) {
                    match = g;
                    break;
                }
            }
            if (match != null) {
                gone.remove(match);
                b.put("from", match.get("from"));
                b.put("from_player", match.get("from_player"));
                b.put("from_id", match.get("id"));
            }
            rest.add(b);
        }
        rest.addAll(gone);
        return rest;
    }

    private static Map<String, Object> move(String id, String name, String from, String fromPlayer, String to, String toPlayer) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", id);
        m.put("name", name);
        m.put("from", from);
        m.put("from_player", fromPlayer);
        m.put("to", to);
        m.put("to_player", toPlayer);
        return m;
    }
}
