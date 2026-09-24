package com.bbfc.notification.core.service;

import com.bbfc.notification.config.ClipConfig;
import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.ClipAlreadyAttachedException;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.SkeletonClip;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.port.ClipStore;
import com.bbfc.notification.core.port.NotificationChannel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.Clock;
import java.util.Optional;

/**
 * Stores the clip, then sends it to Telegram.
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
    private final NotificationChannel notificationChannel;
    private final ClipConfig clipConfig;
    private final Clock clock;

    public AttachClipService(
            AlertRepository alertRepository, ClipStore clipStore, ClipRecorder clipRecorder,
            NotificationChannel notificationChannel, ClipConfig clipConfig, Clock clock) {
        this.alertRepository = alertRepository;
        this.clipStore = clipStore;
        this.clipRecorder = clipRecorder;
        this.notificationChannel = notificationChannel;
        this.clipConfig = clipConfig;
        this.clock = clock;
    }

    /** @return true if the clip also reached Telegram. */
    public boolean attach(EventId eventId, ClipContent content) {
        Alert alert = alertRepository.findByEventId(eventId)
                .orElseThrow(() -> new AlertNotFoundException(eventId));

        if (alert.clip().isPresent()) {
            throw new ClipAlreadyAttachedException(eventId);
        }
        if (content.sizeBytes() > clipConfig.maxBytes()) {
            throw new ClipTooLargeException(content.sizeBytes(), clipConfig.maxBytes());
        }

        String storageKey = clipStore.store(alert.room().roomId(), eventId, content);
        clipRecorder.markStored(eventId, SkeletonClip.stored(storageKey, clock.instant()));

        Optional<Long> replyTo = alert.dispatchMessageId();
        if (replyTo.isEmpty()) {
            log.warn("Clip stored for {} but there is no alert message to reply to", eventId.eventId());
            return false;
        }

        return notificationChannel.sendClipReply(replyTo.get(), content)
                .map(delivery -> {
                    clipRecorder.markDelivered(eventId, delivery);
                    return true;
                })
                .orElseGet(() -> {
                    log.warn("Clip stored for {} but Telegram did not accept it", eventId.eventId());
                    return false;
                });
    }
}
