package com.bbfc.notification.core.domain;

public class KeypointsAlreadyAttachedException extends IllegalStateException {

    public KeypointsAlreadyAttachedException(EventId eventId) {
        super("Keypoints already attached for " + eventId.eventId());
    }
}
