package com.bbfc.notification.domain;

public record EventId(String eventId) {
    public EventId {
        if (eventId == null || eventId.isBlank()) {
            throw new IllegalArgumentException("EventId must not be blank");
        }
    }
}
