package com.bbfc.notification.core.service;

import com.bbfc.notification.config.ClockConfig;
import com.bbfc.notification.core.domain.*;
import com.bbfc.notification.core.port.MessageRenderer;
import com.bbfc.notification.core.port.NotificationChannel;
import org.springframework.stereotype.Service;

import com.bbfc.notification.core.port.AlertRepository;

import java.time.Clock;
import java.time.Instant;

@Service
public class DedupAlertService {
    private final AlertRepository alertRepository;
    private final NotificationChannel notificationChannel;
    private final MessageRenderer messageRenderer;
    private final EscalationPolicy escalationPolicy;
    private final Clock clock;


    public DedupAlertService(
            AlertRepository alertRepository, NotificationChannel notificationChannel,
            EscalationPolicy escalationPolicy, MessageRenderer messageRenderer, Clock clock){
        this.alertRepository = alertRepository;
        this.notificationChannel = notificationChannel;
        this.messageRenderer = messageRenderer;
        this.escalationPolicy = escalationPolicy;
        this.clock = clock;
    }

    public record Result(Alert alert, boolean created){
        
    }

    public Result handleAlert(EventId eventId, RoomRef roomRef, Confidence confidence){
        return alertRepository.findByEventId(eventId)
                .map(existing -> new Result(existing, false))
                .orElseGet(() -> {
                    Instant now = clock.instant();
                    Alert alert = Alert.dispatch(eventId, roomRef, confidence);
                    alert.scheduleNextEscalation(now.plus(escalationPolicy.window()));
                    Alert saved = alertRepository.save(alert);
                    notificationChannel.sendAlert(messageRenderer.render(saved, now));
                    return new Result(saved, true);
                });
    }


}
