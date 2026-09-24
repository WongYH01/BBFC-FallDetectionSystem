package com.bbfc.notification.core.domain;

import java.nio.charset.StandardCharsets;

/**
 * What a button tap means, and how it survives the round trip through Telegram's
 * {@code callback_data} field.
 */
public sealed interface CallbackCommand {

    /** Telegram's hard limit on callback_data. Longer buttons are rejected at send time. */
    int MAX_BYTES = 64;

    record Acknowledge(EventId eventId) implements CallbackCommand {}

    record Triage(EventId eventId, Outcome outcome) implements CallbackCommand {}

    static CallbackCommand parse(String raw) {
        if (raw == null || raw.isBlank()) {
            throw new IllegalArgumentException("callback_data must not be blank");
        }
        String[] parts = raw.split(":");
        return switch (parts[0]) {
            case "ack" -> {
                requireSegments(raw, parts, 2);
                yield new Acknowledge(new EventId(parts[1]));
            }
            case "triage" -> {
                requireSegments(raw, parts, 3);
                yield new Triage(new EventId(parts[1]), parseOutcome(parts[2]));
            }
            default -> throw new IllegalArgumentException("Unknown callback_data action: " + raw);
        };
    }

    default String encode() {
        String encoded = switch (this) {
            case Acknowledge a -> "ack:" + a.eventId().eventId();
            case Triage t -> "triage:" + t.eventId().eventId() + ":" + t.outcome().name().toLowerCase();
        };
        int bytes = encoded.getBytes(StandardCharsets.UTF_8).length;
        if (bytes > MAX_BYTES) {
            throw new IllegalArgumentException(
                    "callback_data is " + bytes + " bytes, exceeding Telegram's limit of " + MAX_BYTES
                            + ": " + encoded);
        }
        return encoded;
    }

    private static void requireSegments(String raw, String[] parts, int expected) {
        if (parts.length != expected) {
            throw new IllegalArgumentException("Malformed callback_data: " + raw);
        }
    }

    private static Outcome parseOutcome(String raw) {
        try {
            return Outcome.valueOf(raw.toUpperCase());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("Unknown triage outcome: " + raw, e);
        }
    }
}
