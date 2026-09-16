package com.bbfc.notification.core.service;

import com.bbfc.notification.config.ClockConfig;
import com.bbfc.notification.core.port.MessageRenderer;
import com.bbfc.notification.core.port.NotificationChannel;
import org.springframework.stereotype.Service;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.RoomRef;
import com.bbfc.notification.core.port.AlertRepository;

import java.time.Clock;

@Service
public class DedupAlertService {
    private final AlertRepository alertRepository;
    private final NotificationChannel notificationChannel;
    private final MessageRenderer messageRenderer;
    private final Clock clock;


    public DedupAlertService(AlertRepository alertRepository, NotificationChannel notificationChannel, MessageRenderer messageRenderer, Clock clock){
        this.alertRepository = alertRepository;
        this.notificationChannel = notificationChannel;
        this.messageRenderer = messageRenderer;
        this.clock = clock;
    }

    public record Result(Alert alert, boolean created){
        
    }

    public Result handleAlert(EventId eventId, RoomRef roomRef, Confidence confidence){
        return alertRepository.findByEventId(eventId)
                .map(existing -> new Result(existing, false))
                .orElseGet(() -> {
                    Alert saved = alertRepository.save(Alert.dispatch(eventId, roomRef, confidence));
                    notificationChannel.sendAlert(messageRenderer.render(saved, clock.instant()));
                    return new Result(saved, true);
                });
    }


}
