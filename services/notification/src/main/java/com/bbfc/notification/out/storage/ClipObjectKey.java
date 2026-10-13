package com.bbfc.notification.out.storage;

import com.bbfc.notification.core.domain.EventId;
import com.bbfc.notification.core.port.ClipStorageException;

import java.util.regex.Pattern;

final class ClipObjectKey {

    private static final Pattern SAFE_SEGMENT = Pattern.compile("[A-Za-z0-9._-]+");

    private ClipObjectKey() {
    }

    static String of(String roomId, EventId eventId) {
        return requireSafe(roomId, "roomId") + "/" + requireSafe(eventId.eventId(), "eventId") + ".mp4";
    }

    private static String requireSafe(String value, String field) {
        if (value == null || !SAFE_SEGMENT.matcher(value).matches()) {
            throw new ClipStorageException("Unsafe " + field + " for a storage key: " + value);
        }
        return value;
    }
}
