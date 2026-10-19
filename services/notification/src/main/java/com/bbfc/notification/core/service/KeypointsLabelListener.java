package com.bbfc.notification.core.service;

import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionalEventListener;

/**
 * Files the keypoints under their label once triage has committed.
 *
 * <p>Async because triage runs on the single Telegram polling thread: a slow or unreachable
 * bucket must not hold up everyone else's button taps.
 */
@Component
public class KeypointsLabelListener {

    private final KeypointsRecorder keypointsRecorder;

    public KeypointsLabelListener(KeypointsRecorder keypointsRecorder) {
        this.keypointsRecorder = keypointsRecorder;
    }

    @Async
    @TransactionalEventListener
    public void onAlertTriaged(AlertTriagedEvent event) {
        keypointsRecorder.labelIfReady(event.eventId());
    }
}
