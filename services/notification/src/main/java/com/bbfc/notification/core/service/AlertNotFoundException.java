package com.bbfc.notification.core.service;

import com.bbfc.notification.core.domain.EventId;

public class AlertNotFoundException extends RuntimeException {

    public AlertNotFoundException(EventId eventId) {
        super("No alert for " + eventId.eventId());
    }
}
