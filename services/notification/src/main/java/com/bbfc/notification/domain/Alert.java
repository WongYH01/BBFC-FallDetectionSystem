package com.bbfc.notification.domain;

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

    private Alert(EventId eventId, RoomRef roomRef, Confidence confidence) {
        this.eventId = eventId;
        this.roomRef = roomRef;
        this.confidence = confidence;
        this.alertState = AlertState.DISPATCHED;
    }

    public static Alert dispatch(EventId eventId, RoomRef room, Confidence confidence) {
        return new Alert(eventId, room, confidence);
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
    }

    public void escalate(Instant at){
        transitionTo(AlertState.ESCALATING);
        this.escalatedAt = at;
    }

    public void exhaust(Instant at){
        transitionTo(AlertState.EXHAUSTED);
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
