package com.bbfc.notification.core.port;

import java.io.IOException;
import java.io.InputStream;

/**
 * An uploaded clip, carried as a re-openable source rather than a stream or a byte array.
 *
 * <p>The clip goes to two destinations, so each adapter opens its own stream. Passing a single
 * {@link InputStream} would let the first consumer exhaust it and leave the second with nothing.
 */
public record ClipContent(String filename, String contentType, long sizeBytes, Source source) {

    @FunctionalInterface
    public interface Source {
        InputStream open() throws IOException;
    }

    public InputStream open() throws IOException {
        return source.open();
    }
}
