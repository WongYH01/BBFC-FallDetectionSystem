package com.bbfc.notification.core.port;

import java.util.Optional;

import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.Outcome;

/** Keypoint files kept for training, filed under their triage outcome once there is one. */
public interface KeypointStore {

    /**
     * Stored unlabelled when {@code outcome} is empty, straight under its label otherwise. The
     * extension is taken from {@code content.filename()}.
     *
     * @return the object key the file was stored under.
     */
    String storeKeypoints(String roomId, EventId eventId, Optional<Outcome> outcome, ClipContent content);

    /**
     * Moves an unlabelled file under its label. A key that is already labelled comes back
     * unchanged without touching storage, so calling this twice is harmless.
     *
     * @return the key the file now lives under.
     */
    String labelKeypoints(String key, Outcome outcome);
}
