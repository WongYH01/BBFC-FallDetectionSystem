package com.bbfc.notification.core.service;

import com.bbfc.notification.core.domain.*;
import org.springframework.stereotype.Service;

import com.bbfc.notification.core.port.AlertRepository;

import java.time.Clock;
import java.time.Instant;

@Service
public class DedupAlertService {
    private final AlertRepository alertRepository;
    private final DispatchPolicy dispatchPolicy;
    private final Clock clock;


    public DedupAlertService(AlertRepository alertRepository, DispatchPolicy dispatchPolicy, Clock clock){
        this.alertRepository = alertRepository;
        this.dispatchPolicy = dispatchPolicy;
        this.clock = clock;
    }

    public record Result(Alert alert, boolean created){
        
    }

    public Result handleAlert(EventId eventId, RoomRef roomRef, Confidence confidence){
        return alertRepository.findByEventId(eventId)
                .map(existing -> new Result(existing, false))
                .orElseGet(() -> {
                    Instant now = clock.instant();
                    Alert alert = Alert.receive(eventId, roomRef, confidence, now, now.plus(dispatchPolicy.clipWait()));
                    return new Result(alertRepository.save(alert), true);
                });
    }


}
