package mage.client.bridge.processor;

import java.util.Set;
import java.util.UUID;

public record BridgeProjectionInputs(
        UUID currentPlayerId,
        Set<UUID> failedManaCasts,
        boolean offerManaSources
) {
    public boolean failedManaCast(UUID objectId) {
        return objectId != null && failedManaCasts.contains(objectId);
    }
}
