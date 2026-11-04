package com.bbfc.notification.core.service;

import com.bbfc.notification.core.domain.UploadLimit;
import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.ClipAlreadyAttachedException;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.SkeletonClip;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.port.ClipStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.Clock;

/**
 * Stores the clip, then dispatches a Pending alert with it. A clip arriving after a text Dispatch
 * is stored but not posted.
 *
 * <p>Deliberately not transactional: the two uploads are independent, and a Telegram failure must
 * not undo the stored copy. Storage goes first because a delivery can later be retried from the
 * stored object, whereas a failed store leaves nothing behind.
 */
@Service
public class AttachClipService {

    private static final Logger log = LoggerFactory.getLogger(AttachClipService.class);

    private final AlertRepository alertRepository;
    private final ClipStore clipStore;
    private final ClipRecorder clipRecorder;
    private final AlertDispatcher alertDispatcher;
    private final UploadLimit uploadLimit;
    private final Clock clock;

    public AttachClipService(
            AlertRepository alertRepository, ClipStore clipStore, ClipRecorder clipRecorder,
            AlertDispatcher alertDispatcher, UploadLimit uploadLimit, Clock clock) {
        this.alertRepository = alertRepository;
        this.clipStore = clipStore;
        this.clipRecorder = clipRecorder;
        this.alertDispatcher = alertDispatcher;
        this.uploadLimit = uploadLimit;
        this.clock = clock;
    }

    /** @return true if the clip went out as the dispatch animation. */
    public boolean attach(EventId eventId, ClipContent content) {
        Alert alert = alertRepository.findByEventId(eventId)
                .orElseThrow(() -> new AlertNotFoundException(eventId));

        if (alert.clip().isPresent()) {
            throw new ClipAlreadyAttachedException(eventId);
        }
        if (content.sizeBytes() > uploadLimit.maxBytes()) {
            throw new ClipTooLargeException(content.sizeBytes(), uploadLimit.maxBytes());
        }

        String storageKey = clipStore.store(alert.room().roomId(), eventId, content);
        clipRecorder.markStored(eventId, SkeletonClip.stored(storageKey, clock.instant()));

        if (alertDispatcher.dispatchWithClip(eventId, content).dispatchCarriesClip()) {
            return true;
        }
        log.warn("Clip stored for {} but not posted: Telegram refused it or the alert went out as text", eventId.eventId());
        return false;
    }
}
