package com.bbfc.notification.core.domain;

public class ClipAlreadyAttachedException extends IllegalStateException {

    public ClipAlreadyAttachedException(EventId eventId) {
        super("Clip already attached for " + eventId.eventId());
    }
}
