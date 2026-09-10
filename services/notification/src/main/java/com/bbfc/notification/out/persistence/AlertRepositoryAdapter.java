package com.bbfc.notification.out.persistence;

import java.util.Optional;

import org.springframework.stereotype.Component;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.port.AlertRepository;

@Component
public class AlertRepositoryAdapter implements AlertRepository {

    private final AlertJPARepository alertJpaRepository;
    private final AlertMapper alertMapper;

    public AlertRepositoryAdapter(AlertJPARepository alertJpaRepository, AlertMapper alertMapper) {
        this.alertJpaRepository = alertJpaRepository;
        this.alertMapper = alertMapper;
    }

    @Override
    public Alert save(Alert alert) {
        AlertEntity savedEntity = alertJpaRepository.save(alertMapper.toEntity(alert));
        return alertMapper.toDomain(savedEntity);
    }

    @Override
    public Optional<Alert> findByEventId(EventId eventId) {
        return alertJpaRepository.findById(eventId.eventId())
        .map(alertMapper::toDomain);
    }
    
}
