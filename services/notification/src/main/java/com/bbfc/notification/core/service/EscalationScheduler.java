package com.bbfc.notification.core.service;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.EscalationPolicy;
import com.bbfc.notification.core.port.AlertRepository;
import com.bbfc.notification.core.port.MessageRenderer;
import com.bbfc.notification.core.port.NotificationChannel;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.Clock;
import java.time.Instant;

@Service
public class EscalationScheduler {

    private static final int BATCH_LIMIT = 50;

    private final AlertRepository alertRepository;
    private final NotificationChannel notificationChannel;
    private final MessageRenderer messageRenderer;
    private final EscalationPolicy escalationPolicy;
    private final Clock clock;

    public EscalationScheduler(
            AlertRepository alertRepository, NotificationChannel notificationChannel,
            MessageRenderer messageRenderer, EscalationPolicy escalationPolicy,
            Clock clock) {
        this.alertRepository = alertRepository;
        this.notificationChannel = notificationChannel;
        this.messageRenderer = messageRenderer;
        this.escalationPolicy = escalationPolicy;
        this.clock = clock;
    }

    @Scheduled(fixedDelay = 5000)
    @Transactional
    public void escalateDueAlerts() {
        Instant now = clock.instant();
        for (Alert alert : alertRepository.findDueForEscalation(now, BATCH_LIMIT)) {
            if (alert.repeatCount() >= escalationPolicy.maxRepeats()) {
                alert.exhaust(now);
            } else {
                alert.escalate(now);
                alert.scheduleNextEscalation(now.plus(escalationPolicy.window()));
                notificationChannel.sendAlert(messageRenderer.render(alert, now));
            }
            alertRepository.save(alert);
        }
    }
}
