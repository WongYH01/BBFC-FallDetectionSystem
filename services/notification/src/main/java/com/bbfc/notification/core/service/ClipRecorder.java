package com.bbfc.notification.core.service;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.SkeletonClip;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.ClipDelivery;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

/**
 * Writes the clip outcomes back, one short transaction each.
 *
 * <p>Separate from {@link AttachClipService} on purpose. Saving writes the whole row, so holding a
 * loaded alert across the two uploads and saving it afterwards would quietly revert an
 * acknowledgement made in the meantime. Each method re-reads under a lock instead.
 */
@Service
public class ClipRecorder {

    private final AlertRepository alertRepository;

    public ClipRecorder(AlertRepository alertRepository) {
        this.alertRepository = alertRepository;
    }

    @Transactional
    public void markStored(EventId eventId, SkeletonClip clip) {
        Alert alert = load(eventId);
        alert.attachClip(clip);
        alertRepository.save(alert);
    }

    @Transactional
    public void markDelivered(EventId eventId, ClipDelivery delivery) {
        Alert alert = load(eventId);
        alert.recordClipDelivery(delivery.fileId(), delivery.messageId());
        alertRepository.save(alert);
    }

    private Alert load(EventId eventId) {
        return alertRepository.findByEventIdForUpdate(eventId)
                .orElseThrow(() -> new AlertNotFoundException(eventId));
    }
}
