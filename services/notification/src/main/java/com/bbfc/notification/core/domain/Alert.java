package com.bbfc.notification.core.domain;

import java.time.Instant;
import java.util.Optional;

public final class Alert {
    private final EventId eventId;
    private final RoomRef roomRef;
    private final Confidence confidence;

    private AlertState alertState;
    private String acknowledgedBy;
    private Instant acknowledgedAt;
    private Instant escalatedAt;
    private Outcome outcome;
    private Instant outcomeAt;
    private int repeatCount;
    private Instant nextEscalationAt;

    private Alert(EventId eventId, RoomRef roomRef, Confidence confidence) {
        this.eventId = eventId;
        this.roomRef = roomRef;
        this.confidence = confidence;
        this.alertState = AlertState.DISPATCHED;
    }

    public static Alert dispatch(EventId eventId, RoomRef room, Confidence confidence) {
        return new Alert(eventId, room, confidence);
    }

    public static Alert load(
            EventId eventId,
            RoomRef roomRef,
            Confidence confidence,
            AlertState alertState,
            String acknowledgedBy,
            Instant acknowledgedAt,
            Instant escalatedAt,
            Outcome outcome,
            Instant outcomeAt,
            int repeatCount,
            Instant nextEscalationAt) {
        Alert alert = new Alert(eventId, roomRef, confidence);
        alert.alertState = alertState;
        alert.acknowledgedBy = acknowledgedBy;
        alert.acknowledgedAt = acknowledgedAt;
        alert.escalatedAt = escalatedAt;
        alert.outcome = outcome;
        alert.outcomeAt = outcomeAt;
        alert.repeatCount = repeatCount;
        alert.nextEscalationAt = nextEscalationAt;
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
