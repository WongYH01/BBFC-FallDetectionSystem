package com.bbfc.notification.core.service;

import java.time.Instant;
import java.util.Optional;

import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.AlertState;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.MessageRenderer;
import com.bbfc.notification.core.port.NotificationChannel;

@Service
public class AcknowledgeAlertService {

    private final AlertRepository alertRepository;
    private final NotificationChannel notificationChannel;
    private final MessageRenderer messageRenderer;
    private final ApplicationEventPublisher eventPublisher;

    public AcknowledgeAlertService(
            AlertRepository alertRepository,
            NotificationChannel notificationChannel,
            MessageRenderer messageRenderer,
            ApplicationEventPublisher eventPublisher) {
        this.alertRepository = alertRepository;
        this.notificationChannel = notificationChannel;
        this.messageRenderer = messageRenderer;
        this.eventPublisher = eventPublisher;
    }

    /**
     * Decide, answer, then fan out. The answer goes out once the outcome is known but before the
     * Telegram edits, which are unbounded in number and would otherwise eat the callback's
     * validity window.
     */
    @Transactional
    public void acknowledge(String callbackQueryId, EventId eventId, String acknowledgedBy, Instant at) {
        Optional<Alert> found = alertRepository.findByEventIdForUpdate(eventId);
        if (found.isEmpty()) {
            notificationChannel.answerCallback(callbackQueryId, "That alert is no longer tracked.");
            return;
        }

        Alert alert = found.get();
        if (!alert.state().canTransitionTo(AlertState.ACKNOWLEDGED)) {
            notificationChannel.answerCallback(callbackQueryId, alreadyHandledMessage(alert));
            return;
        }

        alert.acknowledge(acknowledgedBy, at);
        alertRepository.save(alert);

        notificationChannel.answerCallback(callbackQueryId, "Acknowledged");

        String closed = messageRenderer.renderAcknowledged(alert);
        for (long messageId : alert.messageIds()) {
            notificationChannel.closeMessage(messageId, closed);
        }

        eventPublisher.publishEvent(new AlertAcknowledgedEvent(eventId, acknowledgedBy, at));
    }

    private static String alreadyHandledMessage(Alert alert) {
        return alert.acknowledgedBy()
                .map(by -> "Already handled by " + by)
                .orElse("That alert can no longer be acknowledged.");
    }
}
