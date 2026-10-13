package com.bbfc.notification.core.service;

import java.time.Instant;

import com.bbfc.notification.core.domain.EventId;

/**
 * Carries facts, not the aggregate. The repository maps every save through a fresh entity and
 * hands back a different Alert instance, so sharing a mutable one across the listener boundary
 * would let one full-state merge quietly revert the other's fields.
 */
public record AlertAcknowledgedEvent(EventId eventId, String acknowledgedBy, Instant at) {
}
