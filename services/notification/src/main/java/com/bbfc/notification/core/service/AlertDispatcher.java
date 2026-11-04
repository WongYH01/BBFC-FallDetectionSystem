package com.bbfc.notification.core.service;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.AlertState;
import com.bbfc.notification.core.domain.EscalationPolicy;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.ClipContent;
import com.bbfc.notification.core.port.ClipDelivery;
import com.bbfc.notification.core.port.MessageRenderer;
import com.bbfc.notification.core.port.NotificationChannel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Clock;
import java.time.Instant;
import java.util.Optional;

/**
 * Sends each Pending alert's first message, exactly once.
 *
 * <p>The clip path and the tick both hold the row lock while they check the state and send, so
 * whichever gets there first dispatches and the other sees the alert already out.
 */
@Service
public class AlertDispatcher {

    private static final Logger log = LoggerFactory.getLogger(AlertDispatcher.class);
    private static final int BATCH_LIMIT = 50;

    private final AlertRepository alertRepository;
    private final NotificationChannel notificationChannel;
    private final MessageRenderer messageRenderer;
    private final EscalationPolicy escalationPolicy;
    private final Clock clock;

    public AlertDispatcher(
            AlertRepository alertRepository, NotificationChannel notificationChannel,
            MessageRenderer messageRenderer, EscalationPolicy escalationPolicy, Clock clock) {
        this.alertRepository = alertRepository;
        this.notificationChannel = notificationChannel;
        this.messageRenderer = messageRenderer;
        this.escalationPolicy = escalationPolicy;
        this.clock = clock;
    }

    /** Dispatches as text every Pending alert whose clip did not arrive in time. */
    @Scheduled(fixedDelay = 5000)
    @Transactional
    public void dispatchDueAlerts() {
        Instant now = clock.instant();
        for (Alert alert : alertRepository.findDueForDispatch(now, BATCH_LIMIT)) {
            long messageId;
            try {
                messageId = notificationChannel.sendAlert(alert.eventId(), messageRenderer.render(alert, alert.createdAt()));
            } catch (RuntimeException e) {
                // Left Pending for the next tick; rolling back the batch would re-send the ones already out.
                log.warn("Could not dispatch {}", alert.eventId().eventId(), e);
                continue;
            }
            alert.dispatch(messageId, false, now.plus(escalationPolicy.window()));
            alertRepository.save(alert);
        }
    }

    /**
     * Dispatches a Pending alert as its clip. The clip must already be attached.
     *
     * @return the alert as it now stands: dispatched with the clip, already dispatched by the
     *     tick, or still Pending if Telegram refused the animation.
     */
    @Transactional
    public Alert dispatchWithClip(EventId eventId, ClipContent content) {
        Alert alert = alertRepository.findByEventIdForUpdate(eventId)
                .orElseThrow(() -> new AlertNotFoundException(eventId));
        if (alert.state() != AlertState.PENDING) {
            return alert;
        }

        Optional<ClipDelivery> delivery = notificationChannel.sendAlertWithClip(
                eventId, messageRenderer.render(alert, alert.createdAt()), content);
        if (delivery.isEmpty()) {
            // shortcut: a refused animation waits for the tick's text dispatch at the deadline, until #49 dispatches text at once.
            return alert;
        }

        long messageId = delivery.get().messageId();
        alert.dispatch(messageId, true, clock.instant().plus(escalationPolicy.window()));
        alert.recordClipDelivery(delivery.get().fileId(), messageId);
        return alertRepository.save(alert);
    }
}
