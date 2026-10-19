package com.bbfc.notification.core.service;

import com.bbfc.notification.core.domain.EventId;

/** Facts only, for the same reason as {@link AlertAcknowledgedEvent}. */
public record AlertTriagedEvent(EventId eventId) {
}
