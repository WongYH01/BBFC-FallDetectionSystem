package com.bbfc.notification.out.persistence;

import java.time.Instant;

import org.hibernate.annotations.JdbcTypeCode;
import org.hibernate.type.SqlTypes;

import com.bbfc.notification.core.domain.AlertState;
import com.bbfc.notification.core.domain.Outcome;

import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table (name="alerts")
public class AlertEntity {
    @Id
    private String eventId;

    private String roomId;
    private String roomName;
    private double confidence;
    private Instant createdAt;

    @Enumerated(EnumType.STRING)
    private AlertState state;

    private String acknowledgedBy;
    private Instant acknowledgedAt;
    private Instant escalatedAt;

    @Enumerated(EnumType.STRING)
    private Outcome outcome;
    private Instant outcomeAt;

    private int repeatCount;
    private Instant nextEscalationAt;

    private Long dispatchMessageId;

    @JdbcTypeCode(SqlTypes.ARRAY)
    private Long[] escalationMessageIds;

    private Long followUpMessageId;

    private String clipStorageKey;
    private Instant clipStoredAt;
    private String clipTelegramFileId;
    private Long clipMessageId;

    protected AlertEntity(){}

    public AlertEntity(
            String eventId, String roomId, String roomName,
            double confidence, Instant createdAt, AlertState state, String acknowledgedBy,
            Instant acknowledgedAt, Instant escalatedAt, Outcome outcome,
            Instant outcomeAt, int repeatCount, Instant nextEscalationAt,
            Long dispatchMessageId, Long[] escalationMessageIds, Long followUpMessageId,
            String clipStorageKey, Instant clipStoredAt, String clipTelegramFileId,
            Long clipMessageId) {
        this.eventId = eventId;
        this.roomId = roomId;
        this.roomName = roomName;
        this.confidence = confidence;
        this.createdAt = createdAt;
        this.state = state;
        this.acknowledgedBy = acknowledgedBy;
        this.acknowledgedAt = acknowledgedAt;
        this.escalatedAt = escalatedAt;
        this.outcome = outcome;
        this.outcomeAt = outcomeAt;
        this.repeatCount = repeatCount;
        this.nextEscalationAt = nextEscalationAt;
        this.dispatchMessageId = dispatchMessageId;
        this.escalationMessageIds = escalationMessageIds;
        this.followUpMessageId = followUpMessageId;
        this.clipStorageKey = clipStorageKey;
        this.clipStoredAt = clipStoredAt;
        this.clipTelegramFileId = clipTelegramFileId;
        this.clipMessageId = clipMessageId;
    }

    public String getEventId() {
        return eventId;
    }

    public String getRoomId() {
        return roomId;
    }

    public String getRoomName() {
        return roomName;
    }

    public double getConfidence() {
        return confidence;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public AlertState getState() {
        return state;
    }

    public String getAcknowledgedBy() {
        return acknowledgedBy;
    }

    public Instant getAcknowledgedAt() {
        return acknowledgedAt;
    }

    public Instant getEscalatedAt() {
        return escalatedAt;
    }

    public Outcome getOutcome() {
        return outcome;
    }

    public Instant getOutcomeAt() {
        return outcomeAt;
    }

    public int getRepeatCount() {
        return repeatCount;
    }

    public Instant getNextEscalationAt() {
        return nextEscalationAt;
    }

    public Long getDispatchMessageId() {
        return dispatchMessageId;
    }

    public Long[] getEscalationMessageIds() {
        return escalationMessageIds;
    }

    public Long getFollowUpMessageId() {
        return followUpMessageId;
    }

    public String getClipStorageKey() {
        return clipStorageKey;
    }

    public Instant getClipStoredAt() {
        return clipStoredAt;
    }

    public String getClipTelegramFileId() {
        return clipTelegramFileId;
    }

    public Long getClipMessageId() {
        return clipMessageId;
    }
}
