package com.bbfc.notification.core.service;

import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.MessageRenderer;
import com.bbfc.notification.core.port.NotificationChannel;

@Component
public class FollowUpQuestionListener {

    private static final Logger log = LoggerFactory.getLogger(FollowUpQuestionListener.class);

    private final AlertRepository alertRepository;
    private final NotificationChannel notificationChannel;
    private final MessageRenderer messageRenderer;

    public FollowUpQuestionListener(
            AlertRepository alertRepository,
            NotificationChannel notificationChannel,
            MessageRenderer messageRenderer) {
        this.alertRepository = alertRepository;
        this.notificationChannel = notificationChannel;
        this.messageRenderer = messageRenderer;
    }

    /**
     * Runs synchronously inside the acknowledging transaction, which is what makes the
     * follow_up_message_id check-and-write a real "send once" guarantee rather than a race.
     * MANDATORY asserts that: with plain REQUIRED this would silently open its own transaction
     * and commit the follow-up independently of the acknowledgement.
     *
     * <p>Must stay void — a non-void listener republishes its return value as another event —
     * and must not let anything escape, because under MANDATORY an escaping exception marks the
     * caller's transaction rollback-only and the acknowledgement itself would be undone,
     * resuming escalation on an alert someone has already handled.
     */
    @EventListener
    @Transactional(propagation = Propagation.MANDATORY)
    public void onAlertAcknowledged(AlertAcknowledgedEvent event) {
        try {
            Optional<Alert> found = alertRepository.findByEventIdForUpdate(event.eventId());
            if (found.isEmpty()) {
                return;
            }
            Alert alert = found.get();
            if (alert.followUpMessageId().isPresent()) {
                return;
            }

            long messageId = notificationChannel.sendTriageQuestion(
                    alert.eventId(), messageRenderer.renderTriageQuestion(alert));
            alert.recordFollowUpMessage(messageId);
            alertRepository.save(alert);
        } catch (RuntimeException e) {
            log.error("Could not send the follow-up question for {}", event.eventId(), e);
        }
    }
}
