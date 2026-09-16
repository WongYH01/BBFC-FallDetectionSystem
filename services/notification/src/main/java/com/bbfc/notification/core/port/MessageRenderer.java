package com.bbfc.notification.core.port;

import com.bbfc.notification.core.domain.Alert;

import java.time.Instant;

public interface MessageRenderer {
    String render(Alert alert, Instant at);
}
