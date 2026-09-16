package mage.collectors.services;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import mage.constants.ManaType;
import mage.game.Game;
import mage.game.events.PlayerQueryEvent;
import mage.players.Player;
import mage.util.ShortIdRegistry;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Resumes a game from its record: when {@code GameOptions.replayFrom} names a
 * recorded server event log, every player query is answered with the recorded
 * response for that player, in order, until the record runs out — then the
 * live players (bridges, held meanwhile) take over. Works because the game is
 * seeded ({@code gameSeed}) and object short ids are deterministic.
 * <p>
 * Each fed decision is checked against the record: same query type, the
 * recorded short id resolves, and the game log so far matches line for line
 * (engine refs stripped). Any mismatch is a <em>divergence</em>: feeding stops
 * and the status file says why, so the host can refuse the resume instead of
 * playing on from a different game. Status is written to
 * {@code <gameLogDir>/replay_status.json}.
 * <p>
 * Responses are handed to the player off the game thread: the query event is
 * raised on the game thread <em>before</em> it starts waiting, and
 * {@code HumanPlayer.setResponse*} blocks until the wait opens.
 */
public class ReplayFeederCollector extends EmptyDataCollector {

    public static final String SERVICE_CODE = "replayFeeder";
    private static final Logger logger = Logger.getLogger(ReplayFeederCollector.class);
    private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().create();

    private static final class Replay {
        final ReplayScript script;
        final Path statusPath;
        final ExecutorService feeder = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "replay-feeder");
            t.setDaemon(true);
            return t;
        });
        int fed = 0;
        boolean done = false;
        String divergence = null;
        /** Live username -> recorded username (by position in name order). */
        final Map<String, String> recordedName = new LinkedHashMap<>();

        Replay(ReplayScript script, Path statusPath) {
            this.script = script;
            this.statusPath = statusPath;
        }
    }

    private final Map<UUID, Replay> replays = new ConcurrentHashMap<>();

    @Override
    public String getServiceCode() {
        return SERVICE_CODE;
    }

    @Override
    public String getInitInfo() {
        return "answers recorded decisions for games with replayFrom";
    }

    @Override
    public void onGameStart(Game game) {
        String from = game.getOptions().replayFrom;
        if (from == null || from.isBlank()) {
            return;
        }
        String dir = game.getOptions().gameLogDir != null ? game.getOptions().gameLogDir : Paths.get(from).getParent().toString();
        Path status = Paths.get(dir, "replay_status.json");
        try {
            ReplayScript script = ReplayScript.load(Paths.get(from));
            Replay r = new Replay(script, status);
            // Usernames may differ from the record (a bridge pool hands out
            // fresh ones); both lists are in name order, which is also the
            // order player short ids were assigned in, so position maps them.
            java.util.List<String> live = new java.util.ArrayList<>();
            for (Player p : game.getState().getPlayers().values()) {
                live.add(p.getName());
            }
            java.util.Collections.sort(live);
            java.util.List<String> recorded = script.playerNames();
            if (!recorded.isEmpty() && recorded.size() != live.size()) {
                r.divergence = "record has " + recorded.size() + " players, this game has " + live.size();
                r.done = true;
            }
            for (int i = 0; i < live.size(); i++) {
                r.recordedName.put(live.get(i), i < recorded.size() ? recorded.get(i) : live.get(i));
            }
            replays.put(game.getId(), r);
            logger.info("Replay feeder: " + script.total() + " recorded decisions from " + from + " for game " + game.getId());
            writeStatus(r);
        } catch (IOException e) {
            logger.error("Replay feeder: cannot read " + from + ": " + e.getMessage());
            Replay r = new Replay(ReplayScript.parse(java.util.List.of()), status);
            r.divergence = "cannot read record: " + e.getMessage();
            r.done = true;
            replays.put(game.getId(), r);
            writeStatus(r);
        }
    }

    @Override
    public void onGameLog(Game game, String message, int gameSeq) {
        Replay r = replays.get(game.getId());
        if (r == null || r.done) {
            return;
        }
        String expected = r.script.nextLog();
        if (expected == null) {
            return; // past the recorded log: live play
        }
        String actual = ReplayScript.normalizeLog(message);
        // Log lines name players by username; the live ones may be new.
        for (Map.Entry<String, String> e : r.recordedName.entrySet()) {
            if (!e.getKey().equals(e.getValue())) {
                actual = actual.replace(e.getKey(), e.getValue());
            }
        }
        if (!expected.equals(actual)) {
            diverge(r, "game log differs at seq " + gameSeq + ": recorded \"" + expected + "\", live \"" + actual + "\"");
        }
    }

    @Override
    public void onPlayerQuery(Game game, PlayerQueryEvent event, int gameSeq) {
        Replay r = replays.get(game.getId());
        if (r == null || r.done) {
            return;
        }
        Player player = game.getPlayer(event.getPlayerId());
        if (player == null) {
            return;
        }
        if (r.script.exhausted()) {
            finish(r, "record exhausted; live players take over at seq " + gameSeq);
            return;
        }
        String recordedPlayer = r.recordedName.getOrDefault(player.getName(), player.getName());
        ReplayScript.Decision d = r.script.next(recordedPlayer);
        if (d == null) {
            diverge(r, "query for " + player.getName() + " (recorded as " + recordedPlayer + ") at seq " + gameSeq
                + " but the record has no more decisions for them");
            return;
        }
        String liveType = event.getQueryType().name();
        if (d.queryType() != null && !d.queryType().equals(liveType)) {
            diverge(r, "query type at seq " + gameSeq + " for " + player.getName() + ": recorded " + d.queryType() + ", live " + liveType);
            return;
        }
        Runnable answer;
        try {
            answer = responseFor(game, player, d);
        } catch (IllegalStateException e) {
            diverge(r, e.getMessage());
            return;
        }
        r.fed++;
        r.feeder.submit(() -> {
            try {
                answer.run();
            } catch (RuntimeException e) {
                logger.error("Replay feeder: answering failed at seq " + gameSeq + ": " + e, e);
            }
        });
        if (r.script.exhausted()) {
            finish(r, "all " + r.script.total() + " recorded decisions fed");
        } else {
            writeStatus(r);
        }
    }

    /** The recorded response as a call on the player, resolved against this
     *  game. Each answer is also reported to the collectors first, exactly as
     *  GameController.sendDirectPlayer* does for a client's answer, so the new
     *  record is complete and a resumed game can itself be resumed. */
    private static Runnable responseFor(Game game, Player player, ReplayScript.Decision d) {
        String type = d.responseType() == null ? "" : d.responseType();
        UUID playerId = player.getId();
        mage.collectors.DataCollectorServices collectors = mage.collectors.DataCollectorServices.getInstance();
        switch (type) {
            case "pass":
                return () -> {
                    collectors.onPlayerResponse(game, playerId, "uuid", null);
                    player.setResponseUUID(null);
                };
            case "uuid": {
                if (d.id() == null) {
                    return () -> {
                        collectors.onPlayerResponse(game, playerId, "uuid", null);
                        player.setResponseUUID(null);
                    };
                }
                ShortIdRegistry registry = game.getShortIdRegistry();
                UUID target = registry.tryResolve(d.id());
                if (target == null) {
                    throw new IllegalStateException("recorded choice " + d.id() + " (" + d.name() + ") at seq " + d.seq()
                        + " does not exist in this game");
                }
                return () -> {
                    collectors.onPlayerResponse(game, playerId, "uuid", target);
                    player.setResponseUUID(target);
                };
            }
            case "boolean": {
                Boolean v = d.value() != null && d.value().getAsBoolean();
                return () -> {
                    collectors.onPlayerResponse(game, playerId, "boolean", v);
                    player.setResponseBoolean(v);
                };
            }
            case "string": {
                String v = d.value() == null ? null : d.value().getAsString();
                return () -> {
                    collectors.onPlayerResponse(game, playerId, "string", v);
                    player.setResponseString(v);
                };
            }
            case "integer": {
                Integer v = d.value() == null ? null : d.value().getAsInt();
                return () -> {
                    collectors.onPlayerResponse(game, playerId, "integer", v);
                    player.setResponseInteger(v);
                };
            }
            case "manaType": {
                ManaType mana = d.color() == null ? null : ManaType.valueOf(d.color());
                return () -> {
                    collectors.onPlayerResponse(game, playerId, "manaType", mana);
                    player.setResponseManaType(playerId, mana);
                };
            }
            default:
                throw new IllegalStateException("recorded response type '" + type + "' at seq " + d.seq() + " cannot be replayed");
        }
    }

    private void diverge(Replay r, String why) {
        r.divergence = why;
        r.done = true;
        logger.error("Replay feeder: DIVERGENCE — " + why);
        writeStatus(r);
    }

    private void finish(Replay r, String note) {
        r.done = true;
        logger.info("Replay feeder: done — " + note);
        writeStatus(r);
    }

    @Override
    public void onGameEnd(Game game) {
        Replay r = replays.remove(game.getId());
        if (r != null) {
            r.feeder.shutdown();
        }
    }

    private static void writeStatus(Replay r) {
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("total", r.script.total());
        status.put("fed", r.fed);
        status.put("remaining", r.script.remaining());
        status.put("done", r.done);
        status.put("divergence", r.divergence);
        try {
            Path tmp = r.statusPath.resolveSibling(r.statusPath.getFileName() + ".tmp");
            Files.writeString(tmp, GSON.toJson(status), StandardCharsets.UTF_8);
            Files.move(tmp, r.statusPath, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            logger.warn("Replay feeder: cannot write " + r.statusPath + ": " + e.getMessage());
        }
    }
}
