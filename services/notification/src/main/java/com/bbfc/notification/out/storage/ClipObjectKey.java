package com.bbfc.notification.out.storage;

import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.domain.Outcome;
import com.bbfc.notification.core.port.ClipStorageException;

import java.util.Locale;
import java.util.Optional;
import java.util.regex.Pattern;

final class ClipObjectKey {

    private static final Pattern SAFE_SEGMENT = Pattern.compile("[A-Za-z0-9._-]+");
    private static final String KEYPOINTS_INCOMING = "keypoints/incoming/";
    private static final String KEYPOINTS_LABELLED = "keypoints/labelled/";

    private ClipObjectKey() {
    }

    static String of(String roomId, EventId eventId) {
        return requireSafe(roomId, "roomId") + "/" + requireSafe(eventId.eventId(), "eventId") + ".mp4";
    }

    static String keypoints(String roomId, EventId eventId, Optional<Outcome> outcome, String filename) {
        String prefix = outcome.map(ClipObjectKey::labelledPrefix).orElse(KEYPOINTS_INCOMING);
        return prefix + requireSafe(roomId, "roomId") + "/" + requireSafe(eventId.eventId(), "eventId")
                + "." + requireSafe(extension(filename), "extension");
    }

    static boolean isUnlabelled(String key) {
        return key.startsWith(KEYPOINTS_INCOMING);
    }

    static String labelled(String unlabelledKey, Outcome outcome) {
        return labelledPrefix(outcome) + unlabelledKey.substring(KEYPOINTS_INCOMING.length());
    }

    private static String labelledPrefix(Outcome outcome) {
        return KEYPOINTS_LABELLED + outcome.name().toLowerCase(Locale.ROOT) + "/";
    }

    private static String extension(String filename) {
        int dot = filename == null ? -1 : filename.lastIndexOf('.');
        return dot < 0 ? null : filename.substring(dot + 1);
    }

    private static String requireSafe(String value, String field) {
        if (value == null || !SAFE_SEGMENT.matcher(value).matches()) {
            throw new ClipStorageException("Unsafe " + field + " for a storage key: " + value);
        }
        return value;
    }
}
