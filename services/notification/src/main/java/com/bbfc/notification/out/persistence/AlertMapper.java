package com.bbfc.notification.out.persistence;

import org.springframework.stereotype.Component;

import com.bbfc.notification.core.domain.Alert;
import com.bbfc.notification.core.domain.Confidence;
import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.RoomRef;

@Component 
public class AlertMapper {

    // converts domain type to entity for DB calls
    public AlertEntity toEntity(Alert alert){
        AlertEntity parsedAlertEntity = new AlertEntity(
                alert.eventId().eventId(),
                alert.room().roomId(),
                alert.room().displayName(),
                alert.confidence().confidenceScore(),
                alert.state(),
                alert.acknowledgedBy().orElse(null),
                alert.acknowledgedAt().orElse(null),
                alert.escalatedAt().orElse(null),
                alert.outcome().orElse(null),
                alert.outcomeAt().orElse(null)
        );
        return parsedAlertEntity;
    }

    // convert entity back into domain for business logic
    public Alert toDomain(AlertEntity alertEntity){
        Alert parsedAlertDomain = Alert.load(
                new EventId(alertEntity.getEventId()),
                new RoomRef(alertEntity.getRoomId(), alertEntity.getRoomName()),
                new Confidence(alertEntity.getConfidence()),
                alertEntity.getState(),
                alertEntity.getAcknowledgedBy(),
                alertEntity.getAcknowledgedAt(),
                alertEntity.getEscalatedAt(),
                alertEntity.getOutcome(),
                alertEntity.getOutcomeAt()
        );
        return parsedAlertDomain;
    }
}
