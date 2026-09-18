package com.bbfc.notification.core.port;

import com.bbfc.notification.core.domain.EventId;

public interface ClipStore {

    /** @return the object key the clip was stored under. */
    String store(String roomId, EventId eventId, ClipContent content);

    /** Idempotent — tolerates a bucket that already exists. */
    void ensureBucket();
}
