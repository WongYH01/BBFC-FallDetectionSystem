package com.bbfc.notification.core.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.KeypointsFile;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.ClipStorageException;
import com.bbfc.notification.core.port.KeypointStore;

/**
 * Writes keypoint outcomes back under the row lock, like {@link ClipRecorder}.
 *
 * <p>Both the upload and the triage call {@link #labelIfReady}. Whichever takes the lock second
 * sees the other's result, so exactly one of them moves the file whatever order they land in.
 */
@Service
public class KeypointsRecorder {

    private static final Logger log = LoggerFactory.getLogger(KeypointsRecorder.class);

    private final AlertRepository alertRepository;
    private final KeypointStore keypointStore;

    public KeypointsRecorder(AlertRepository alertRepository, KeypointStore keypointStore) {
        this.alertRepository = alertRepository;
        this.keypointStore = keypointStore;
    }

    @Transactional
    public void markStored(EventId eventId, KeypointsFile keypoints) {
        Alert alert = load(eventId);
        alert.attachKeypoints(keypoints);
        alertRepository.save(alert);
    }

    /** REQUIRES_NEW: also called from an after-commit listener, where the old transaction is spent. */
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void labelIfReady(EventId eventId) {
        Alert alert = load(eventId);
        if (alert.outcome().isEmpty() || alert.keypoints().isEmpty()) {
            return;
        }
        String key = alert.keypoints().get().storageKey();
        String labelled;
        try {
            labelled = keypointStore.labelKeypoints(key, alert.outcome().get());
        } catch (ClipStorageException e) {
            log.warn("Keypoints for {} stay unlabelled at {}", eventId.eventId(), key, e);
            return;
        }
        if (!labelled.equals(key)) {
            alert.relocateKeypoints(labelled);
            alertRepository.save(alert);
        }
    }

    private Alert load(EventId eventId) {
        return alertRepository.findByEventIdForUpdate(eventId)
                .orElseThrow(() -> new AlertNotFoundException(eventId));
    }
}
