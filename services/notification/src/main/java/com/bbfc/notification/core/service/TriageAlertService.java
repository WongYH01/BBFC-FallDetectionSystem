package com.bbfc.notification.core.service;

import java.time.Instant;
import java.util.Optional;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.AlertState;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.Outcome;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.MessageRenderer;
import com.bbfc.notification.core.port.NotificationChannel;

@Service
public class TriageAlertService {

    private final AlertRepository alertRepository;
    private final NotificationChannel notificationChannel;
    private final MessageRenderer messageRenderer;

    public TriageAlertService(
            AlertRepository alertRepository,
            NotificationChannel notificationChannel,
            MessageRenderer messageRenderer) {
        this.alertRepository = alertRepository;
        this.notificationChannel = notificationChannel;
        this.messageRenderer = messageRenderer;
    }

    /** Same decide-answer-fan-out shape as acknowledgement; triage buttons get double-tapped too. */
    @Transactional
    public void triage(String callbackQueryId, EventId eventId, Outcome outcome, Instant at) {
        Optional<Alert> found = alertRepository.findByEventIdForUpdate(eventId);
        if (found.isEmpty()) {
            notificationChannel.answerCallback(callbackQueryId, "That alert is no longer tracked.");
            return;
        }

        Alert alert = found.get();
        if (!alert.state().canTransitionTo(AlertState.TRIAGED)) {
            notificationChannel.answerCallback(callbackQueryId, alreadyRecordedMessage(alert));
            return;
        }

        alert.triage(outcome, at);
        alertRepository.save(alert);

        notificationChannel.answerCallback(callbackQueryId, "Recorded");

        alert.followUpMessageId().ifPresent(
                messageId -> notificationChannel.closeMessage(messageId, messageRenderer.renderTriaged(alert)));
    }

    private static String alreadyRecordedMessage(Alert alert) {
        return alert.outcome()
                .map(outcome -> "Already recorded as " + describe(outcome))
                .orElse("That alert has not been acknowledged yet.");
    }

    private static String describe(Outcome outcome) {
        return outcome == Outcome.GENUINE ? "a genuine fall" : "a false alarm";
    }
}
