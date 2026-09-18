package com.bbfc.notification.core.port;

import java.time.Instant;
import java.util.List;
import java.util.Optional;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.EventId;

public interface AlertRepository {
    Alert save(Alert alert);
    Optional<Alert> findByEventId(EventId eventId);
    List<Alert> findDueForEscalation(Instant now, int limit);
}
