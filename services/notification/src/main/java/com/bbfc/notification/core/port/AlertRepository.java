package com.bbfc.notification.core.port;

import java.util.Optional;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.EventId;

public interface AlertRepository {
    Alert save(Alert alert);
    Optional<Alert> findByEventId(EventId eventId);
}
