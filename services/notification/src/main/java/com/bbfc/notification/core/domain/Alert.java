package com.bbfc.notification.core.domain;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public final class Alert {
    private final EventId eventId;
    private final RoomRef roomRef;
    private final Confidence confidence;
    private final Instant createdAt;

    private AlertState alertState;
    private String acknowledgedBy;
    private Instant acknowledgedAt;
    private Instant escalatedAt;
    private Outcome outcome;
    private Instant outcomeAt;
    private int repeatCount;
    private Instant nextEscalationAt;
    private Long dispatchMessageId;
    private final List<Long> escalationMessageIds = new ArrayList<>();
    private Long followUpMessageId;

    private Alert(EventId eventId, RoomRef roomRef, Confidence confidence, Instant createdAt) {
        this.eventId = eventId;
        this.roomRef = roomRef;
        this.confidence = confidence;
        this.createdAt = createdAt;
        this.alertState = AlertState.DISPATCHED;
    }

    public static Alert dispatch(EventId eventId, RoomRef room, Confidence confidence, Instant createdAt) {
        return new Alert(eventId, room, confidence, createdAt);
    }

    public static Alert load(
            EventId eventId,
            RoomRef roomRef,
            Confidence confidence,
            Instant createdAt,
            AlertState alertState,
            String acknowledgedBy,
            Instant acknowledgedAt,
            Instant escalatedAt,
            Outcome outcome,
            Instant outcomeAt,
            int repeatCount,
            Instant nextEscalationAt,
            Long dispatchMessageId,
            List<Long> escalationMessageIds,
            Long followUpMessageId) {
        Alert alert = new Alert(eventId, roomRef, confidence, createdAt);
        alert.alertState = alertState;
        alert.acknowledgedBy = acknowledgedBy;
        alert.acknowledgedAt = acknowledgedAt;
        alert.escalatedAt = escalatedAt;
        alert.outcome = outcome;
        alert.outcomeAt = outcomeAt;
        alert.repeatCount = repeatCount;
        alert.nextEscalationAt = nextEscalationAt;
        alert.dispatchMessageId = dispatchMessageId;
        alert.escalationMessageIds.addAll(escalationMessageIds);
        alert.followUpMessageId = followUpMessageId;
        return alert;
    }

    private void transitionTo(AlertState targetState) {
        if (!alertState.canTransitionTo(targetState)) {
            throw new IllegalAlertTransitionException(alertState, targetState);
        }
        this.alertState = targetState;
    }

    public void acknowledge(String by, Instant at) {
        transitionTo(AlertState.ACKNOWLEDGED);
        this.acknowledgedBy = by;
        this.acknowledgedAt = at;
        this.nextEscalationAt = null;
    }

    public void escalate(Instant at){
        transitionTo(AlertState.ESCALATING);
        this.escalatedAt = at;
        this.repeatCount++;
    }

    public void exhaust(Instant at){
        transitionTo(AlertState.EXHAUSTED);
        this.nextEscalationAt = null;
    }

    public void scheduleNextEscalation(Instant at){
        this.nextEscalationAt = at;
    }

    public void triage(Outcome outcome, Instant at) {
        transitionTo(AlertState.TRIAGED);
        this.outcome = outcome;
        this.outcomeAt = at;
    }

    public void recordDispatchMessage(long messageId) {
        if (this.dispatchMessageId != null) {
            throw new IllegalStateException("Dispatch message already recorded for " + eventId);
        }
        this.dispatchMessageId = messageId;
    }

    public void recordEscalationMessage(long messageId) {
        this.escalationMessageIds.add(messageId);
    }

    public void recordFollowUpMessage(long messageId) {
        if (this.followUpMessageId != null) {
            throw new IllegalStateException("Follow-up message already recorded for " + eventId);
        }
        this.followUpMessageId = messageId;
    }

    /** Every message this alert has produced, in send order — the set to close out on acknowledgement. */
    public List<Long> messageIds() {
        List<Long> ids = new ArrayList<>();
        if (dispatchMessageId != null) {
            ids.add(dispatchMessageId);
        }
        ids.addAll(escalationMessageIds);
        return List.copyOf(ids);
    }

    public AlertState state() {
        return alertState;
    }

    public Optional<String> acknowledgedBy() {
        return Optional.ofNullable(acknowledgedBy);
    }

    public Optional<Outcome> outcome() {
        return Optional.ofNullable(outcome);
    }

    public Optional<Instant> acknowledgedAt() {
        return Optional.ofNullable(acknowledgedAt);
    }

    public Optional<Instant> escalatedAt() {
        return Optional.ofNullable(escalatedAt);
    }

    public Optional<Instant> outcomeAt() {
        return Optional.ofNullable(outcomeAt);
    }

    public int repeatCount(){
        return repeatCount;
    }

    public Optional<Instant> nextEscalationAt() {
        return Optional.ofNullable(nextEscalationAt);
    }

    public Optional<Long> dispatchMessageId() {
        return Optional.ofNullable(dispatchMessageId);
    }

    public List<Long> escalationMessageIds() {
        return List.copyOf(escalationMessageIds);
    }

    public Optional<Long> followUpMessageId() {
        return Optional.ofNullable(followUpMessageId);
    }

    public Instant createdAt() {
        return createdAt;
    }

    public EventId eventId() {
        return eventId;
    }

    public RoomRef room() {
        return roomRef;
    }

    public Confidence confidence() {
        return confidence;
    }

}
