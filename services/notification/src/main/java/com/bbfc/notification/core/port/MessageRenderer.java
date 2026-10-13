package com.bbfc.notification.core.port;

import com.bbfc.notification.core.domain.Alert;

import java.time.Instant;

public interface MessageRenderer {
    String render(Alert alert, Instant at);

    // These read every timestamp they need off the aggregate, so there is no "at" to get wrong.

    /** Replaces an alert message once it has been handled. */
    String renderAcknowledged(Alert alert);

    /** The follow-up question asked after acknowledgement. */
    String renderTriageQuestion(Alert alert);

    /** Replaces the follow-up question once it has been answered. */
    String renderTriaged(Alert alert);
}
