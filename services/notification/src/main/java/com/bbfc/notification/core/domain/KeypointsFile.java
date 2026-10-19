package com.bbfc.notification.core.domain;

import java.time.Instant;

/** The stored keypoint file for a fall, kept for training. Its key changes once triage labels it. */
public record KeypointsFile(String storageKey, Instant storedAt) {

    public KeypointsFile {
        if (storageKey == null || storageKey.isBlank()) {
            throw new IllegalArgumentException("storageKey must not be blank");
        }
        if (storedAt == null) {
            throw new IllegalArgumentException("storedAt must not be null");
        }
    }

    public KeypointsFile relocated(String newKey) {
        return new KeypointsFile(newKey, storedAt);
    }
}
