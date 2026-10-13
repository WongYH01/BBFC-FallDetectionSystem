package com.bbfc.notification.core.port;

import java.time.Instant;
import java.util.List;
import java.util.Optional;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.EventId;

public interface AlertRepository {
    Alert save(Alert alert);
    Optional<Alert> findByEventId(EventId eventId);

    /** Takes a row-level write lock for the caller's transaction. Concurrent callers block. */
    Optional<Alert> findByEventIdForUpdate(EventId eventId);

    List<Alert> findDueForEscalation(Instant now, int limit);
}
