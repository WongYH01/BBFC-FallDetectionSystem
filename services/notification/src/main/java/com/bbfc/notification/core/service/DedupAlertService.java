package com.bbfc.notification.core.service;

import org.springframework.stereotype.Service;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.RoomRef;
import com.bbfc.notification.core.port.AlertRepository;

@Service
public class DedupAlertService {
    private final AlertRepository alertRepository;

    public DedupAlertService(AlertRepository alertRepository){
        this.alertRepository = alertRepository;
    }

    public record Result(Alert alert, boolean created){
        
    }

    public Result handleAlert(EventId eventId, RoomRef roomRef, Confidence confidence){
        return alertRepository.findByEventId(eventId)
                .map(existing -> new Result(existing, false))
                .orElseGet(() -> {
                    Alert alert = Alert.dispatch(eventId, roomRef, confidence);
                    return new Result(alertRepository.save(alert), true);
                });
    }


}
