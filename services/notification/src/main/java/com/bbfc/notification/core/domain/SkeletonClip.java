package com.bbfc.notification.core.domain;

import java.time.Instant;

/**
 * The stored copy of a fall clip, plus the outcome of sending it to Telegram.
 *
 * <p>Storage and delivery are independent — a clip can be stored but undelivered, and that
 * partial state is recorded rather than treated as a failure.
 */
public record SkeletonClip(
        String storageKey,
        Instant storedAt,
        String telegramFileId,
        Long telegramMessageId) {

    public SkeletonClip {
        if (storageKey == null || storageKey.isBlank()) {
            throw new IllegalArgumentException("storageKey must not be blank");
        }
        if (storedAt == null) {
            throw new IllegalArgumentException("storedAt must not be null");
        }
    }

    public static SkeletonClip stored(String storageKey, Instant storedAt) {
        return new SkeletonClip(storageKey, storedAt, null, null);
    }

    public SkeletonClip delivered(String telegramFileId, long telegramMessageId) {
        return new SkeletonClip(storageKey, storedAt, telegramFileId, telegramMessageId);
    }

    public boolean isDelivered() {
        return telegramMessageId != null;
    }
}
